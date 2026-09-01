"""Three-pass matcher: exact payment_id -> candidate retrieval -> reasoner verdict."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import pandas as pd

from .ingest import Batch
from .reasoner import Reasoner, build_reasoner

MATCHED = "matched"
UNCERTAIN = "uncertain"
UNMATCHED = "unmatched"


@dataclass
class MatchConfig:
    amount_tolerance: float = 0.05  # ±5% window for candidate retrieval
    date_window_days: int = 3  # ±3 days around the settlement date
    top_k: int = 3
    accept_confidence: float = 0.80  # verdict confidence needed to auto-accept
    amount_mismatch_tolerance: float = 0.01  # >1% gross vs order amount => flagged
    delayed_settlement_days: int = 7


@dataclass
class MatchRow:
    settlement_id: str
    payment_id: str
    order_id: str | None
    match_status: str
    match_type: str  # exact | fuzzy | none
    confidence: float
    reason: str
    exception_reason: str | None
    reasoner: str
    gross_amount: float | None
    order_amount: float | None
    amount_delta: float | None
    fees: float | None
    tax: float | None
    net_amount: float | None
    settlement_date: str | None
    order_date: str | None
    settlement_lag_days: float | None
    order_status: str | None
    candidate_order_ids: str


@dataclass
class MatchOutcome:
    matches: pd.DataFrame
    unsettled_orders: pd.DataFrame
    config: MatchConfig
    reasoner_name: str
    llm_calls: int
    elapsed_seconds: float


def _candidates(settlement: pd.Series, orders: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    gross = settlement["gross_amount"]
    if pd.isna(gross):
        return orders.iloc[0:0]
    low, high = gross * (1 - config.amount_tolerance), gross * (1 + config.amount_tolerance)
    # a partial refund makes the settlement smaller than the order, so the upper
    # bound on the order amount is widened
    window = orders[(orders["amount"] >= low) & (orders["amount"] <= high * 1.6)].copy()
    lag = (settlement["settlement_date"] - window["order_date"]).dt.days
    window["day_gap"] = lag
    window = window[(lag >= -1) & (lag <= config.date_window_days + config.delayed_settlement_days)]
    if window.empty:
        return window
    window["amount_gap"] = (window["amount"] - gross).abs()
    window = window.sort_values(["amount_gap", "day_gap"])
    return window.head(config.top_k)


def _exception_for(settlement: pd.Series, order: pd.Series | None, config: MatchConfig,
                   *, duplicate: bool, recovered: bool) -> tuple[str | None, str | None]:
    if duplicate:
        return "duplicate_settlement", "second settlement seen for the same payment_id"
    if order is None:
        return None, None
    delta = settlement["gross_amount"] - order["amount"]
    if abs(delta) > config.amount_mismatch_tolerance * max(order["amount"], 1):
        direction = "short" if delta < 0 else "over"
        return "amount_mismatch", f"settled {direction} by {abs(delta):.2f} vs order amount"
    lag = (settlement["settlement_date"] - order["order_date"]).days
    if lag > config.delayed_settlement_days:
        return "date_drift", f"settled {lag} days after the order (delayed, not broken)"
    if recovered:
        return "orphan_settlement", "settlement had no payment_id; recovered by amount/date retrieval"
    return None, None


def run_matching(batch: Batch, config: MatchConfig | None = None,
                 reasoner: Reasoner | None = None, use_llm: bool = True) -> MatchOutcome:
    config = config or MatchConfig()
    reasoner = reasoner or build_reasoner(use_llm=use_llm)
    started = time.perf_counter()

    orders = batch.orders
    settlements = batch.settlements
    by_payment = {
        pid: row
        for pid, row in orders.set_index("razorpay_payment_id").iterrows()
        if isinstance(pid, str) and pid
    }

    rows: list[MatchRow] = []
    seen_payment_ids: set[str] = set()
    consumed_orders: set[str] = set()
    llm_calls = 0

    for _, stl in settlements.iterrows():
        pid = str(stl["payment_id"] or "")
        order: pd.Series | None = None
        match_type = "none"
        confidence = 0.0
        reason = ""
        candidate_ids: list[str] = []
        duplicate = False
        recovered = False
        reasoner_used = "rule"

        if pid and pid in by_payment:
            # --- pass 1: exact link ---------------------------------------
            order = by_payment[pid]
            order = order.copy()
            order["order_id"] = order.name if "order_id" not in order else order["order_id"]
            order_id = orders.loc[orders["razorpay_payment_id"] == pid, "order_id"].iloc[0]
            order["order_id"] = order_id
            match_type = "exact"
            confidence = 0.99
            reason = "exact payment_id link between settlement and order"
            duplicate = pid in seen_payment_ids
            seen_payment_ids.add(pid)
        else:
            # --- pass 2: candidate retrieval -------------------------------
            pool = orders[~orders["order_id"].isin(consumed_orders)]
            cands = _candidates(stl, pool, config)
            candidate_ids = cands["order_id"].tolist()
            # --- pass 3: reasoner verdict ----------------------------------
            verdict = reasoner.judge(
                {
                    "settlement_id": stl["settlement_id"],
                    "gross_amount": float(stl["gross_amount"]),
                    "net_amount": float(stl["net_amount"]),
                    "settlement_date": str(stl["settlement_date"].date())
                    if pd.notna(stl["settlement_date"]) else None,
                    "payment_id": pid or None,
                },
                [
                    {
                        "order_id": c["order_id"],
                        "amount": float(c["amount"]),
                        "order_date": str(c["order_date"].date()) if pd.notna(c["order_date"]) else None,
                        "status": c["status"],
                        "day_gap": int(c["day_gap"]) if pd.notna(c["day_gap"]) else None,
                    }
                    for _, c in cands.iterrows()
                ],
            )
            reasoner_used = verdict.source
            llm_calls += int(verdict.source == "llm")
            confidence = verdict.confidence
            reason = verdict.reason
            if verdict.verdict == "match" and verdict.confidence >= config.accept_confidence and verdict.order_id:
                order = orders.loc[orders["order_id"] == verdict.order_id].iloc[0]
                match_type = "fuzzy"
                recovered = True
            elif verdict.verdict == "uncertain" and verdict.order_id:
                order = None
                match_type = "fuzzy"

        if order is not None:
            consumed_orders.add(str(order["order_id"]))
            exception_reason, exception_note = _exception_for(
                stl, order, config, duplicate=duplicate, recovered=recovered
            )
            status = MATCHED if not duplicate else UNCERTAIN
        else:
            if pid and pid not in by_payment:
                exception_reason, exception_note = (
                    "no_order",
                    "settlement references a payment_id absent from orders (test/dummy?)",
                )
            elif candidate_ids:
                exception_reason, exception_note = (
                    "ambiguous_candidates",
                    f"{len(candidate_ids)} plausible order(s), none accepted above threshold",
                )
            else:
                exception_reason, exception_note = (
                    "orphan_settlement",
                    "no payment_id link and no candidate order in the amount/date window",
                )
            status = UNCERTAIN if candidate_ids else UNMATCHED

        rows.append(
            MatchRow(
                settlement_id=str(stl["settlement_id"]),
                payment_id=pid,
                order_id=str(order["order_id"]) if order is not None else None,
                match_status=status,
                match_type=match_type,
                confidence=round(float(confidence), 3),
                reason=reason if not exception_note else f"{reason}; {exception_note}",
                exception_reason=exception_reason,
                reasoner=reasoner_used,
                gross_amount=float(stl["gross_amount"]) if pd.notna(stl["gross_amount"]) else None,
                order_amount=float(order["amount"]) if order is not None else None,
                amount_delta=round(float(stl["gross_amount"] - order["amount"]), 2)
                if order is not None and pd.notna(stl["gross_amount"]) else None,
                fees=float(stl["fees"]) if pd.notna(stl["fees"]) else None,
                tax=float(stl["tax"]) if pd.notna(stl["tax"]) else None,
                net_amount=float(stl["net_amount"]) if pd.notna(stl["net_amount"]) else None,
                settlement_date=str(stl["settlement_date"].date()) if pd.notna(stl["settlement_date"]) else None,
                order_date=str(order["order_date"].date())
                if order is not None and pd.notna(order["order_date"]) else None,
                settlement_lag_days=float((stl["settlement_date"] - order["order_date"]).days)
                if order is not None and pd.notna(order["order_date"]) and pd.notna(stl["settlement_date"])
                else None,
                order_status=str(order["status"]) if order is not None else None,
                candidate_order_ids=",".join(candidate_ids),
            )
        )

    matches = pd.DataFrame([asdict(r) for r in rows])
    matched_ids = set(matches["order_id"].dropna())
    unsettled = orders[~orders["order_id"].isin(matched_ids)][
        ["order_id", "amount", "order_date", "status", "razorpay_payment_id"]
    ].copy()
    unsettled["exception_reason"] = "unsettled_order"
    unsettled["reason"] = "order has no settlement in this batch"

    return MatchOutcome(
        matches=matches,
        unsettled_orders=unsettled,
        config=config,
        reasoner_name=getattr(reasoner, "name", "unknown"),
        llm_calls=llm_calls,
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
