"""Reconciliation metrics: match rate, false-match rate, exception breakdown."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from .ingest import Batch
from .match import MATCHED, MatchOutcome


@dataclass
class Metrics:
    settlements: int
    orders: int
    matched: int
    uncertain: int
    unmatched: int
    match_rate: float
    exact_match_rate: float
    fuzzy_recovered: int
    exceptions: int
    unsettled_orders: int
    false_matches: int | None
    false_match_rate: float | None
    missed_matches: int | None
    elapsed_seconds: float
    throughput_per_second: float
    llm_calls: int
    reasoner: str

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(batch: Batch, outcome: MatchOutcome) -> Metrics:
    m = outcome.matches
    total = len(m)
    matched = int((m["match_status"] == MATCHED).sum())
    uncertain = int((m["match_status"] == "uncertain").sum())
    unmatched = total - matched - uncertain
    accepted = m[m["match_status"] == MATCHED]
    confident = accepted[accepted["confidence"] >= outcome.config.accept_confidence]

    false_matches = missed = None
    false_match_rate = None
    if batch.ground_truth is not None and not batch.ground_truth.empty:
        truth = batch.ground_truth.set_index("settlement_id")["true_order_id"].to_dict()
        judged = m[m["order_id"].notna()]
        false_matches = int(
            sum(1 for _, r in judged.iterrows() if truth.get(r["settlement_id"], "") != r["order_id"])
        )
        false_match_rate = round(false_matches / max(len(judged), 1), 4)
        missed = int(
            sum(
                1
                for _, r in m.iterrows()
                if truth.get(r["settlement_id"], "") and pd.isna(r["order_id"])
            )
        )

    elapsed = max(outcome.elapsed_seconds, 1e-6)
    return Metrics(
        settlements=total,
        orders=len(batch.orders),
        matched=matched,
        uncertain=uncertain,
        unmatched=unmatched,
        match_rate=round(len(confident) / max(total, 1), 4),
        exact_match_rate=round(int((m["match_type"] == "exact").sum()) / max(total, 1), 4),
        fuzzy_recovered=int(((m["match_type"] == "fuzzy") & (m["match_status"] == MATCHED)).sum()),
        exceptions=int(m["exception_reason"].notna().sum()),
        unsettled_orders=len(outcome.unsettled_orders),
        false_matches=false_matches,
        false_match_rate=false_match_rate,
        missed_matches=missed,
        elapsed_seconds=outcome.elapsed_seconds,
        throughput_per_second=round(total / elapsed, 2),
        llm_calls=outcome.llm_calls,
        reasoner=outcome.reasoner_name,
    )


def exception_breakdown(outcome: MatchOutcome) -> pd.DataFrame:
    m = outcome.matches
    exc = m[m["exception_reason"].notna()]
    rows = (
        exc.groupby("exception_reason")
        .agg(
            count=("settlement_id", "size"),
            settlement_ids=("settlement_id", lambda s: ", ".join(sorted(s)[:8])),
            example_reason=("reason", "first"),
        )
        .reset_index()
    )
    if len(outcome.unsettled_orders):
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "exception_reason": "unsettled_order",
                            "count": len(outcome.unsettled_orders),
                            "settlement_ids": ", ".join(
                                sorted(outcome.unsettled_orders["order_id"])[:8]
                            ),
                            "example_reason": "order has no settlement in this batch",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return rows.sort_values("count", ascending=False).reset_index(drop=True)


def financial_summary(outcome: MatchOutcome) -> dict[str, float]:
    m = outcome.matches
    return {
        "gross_amount": round(float(m["gross_amount"].sum()), 2),
        "fees": round(float(m["fees"].sum()), 2),
        "tax": round(float(m["tax"].sum()), 2),
        "net_payout": round(float(m["net_amount"].sum()), 2),
        "matched_gross": round(float(m.loc[m["match_status"] == MATCHED, "gross_amount"].sum()), 2),
        "exception_gross": round(float(m.loc[m["exception_reason"].notna(), "gross_amount"].sum()), 2),
    }


def render_text_report(metrics: Metrics, breakdown: pd.DataFrame, money: dict[str, float]) -> str:
    lines = [
        "SETTLEMENT RECONCILIATION REPORT",
        "=" * 64,
        f"settlements: {metrics.settlements}   orders: {metrics.orders}   reasoner: {metrics.reasoner}",
        f"matched: {metrics.matched}   uncertain: {metrics.uncertain}   unmatched: {metrics.unmatched}",
        f"match rate (conf >= threshold): {metrics.match_rate:.1%}   exact: {metrics.exact_match_rate:.1%}"
        f"   recovered by retrieval: {metrics.fuzzy_recovered}",
    ]
    if metrics.false_match_rate is not None:
        lines.append(
            f"false-match rate vs ground truth: {metrics.false_match_rate:.2%} "
            f"({metrics.false_matches} wrong, {metrics.missed_matches} missed)"
        )
    lines += [
        f"latency: {metrics.elapsed_seconds}s   throughput: {metrics.throughput_per_second} settlements/s"
        f"   llm calls: {metrics.llm_calls}",
        "",
        "MONEY (whole batch)",
        "-" * 64,
        f"gross {money['gross_amount']:,.2f}   fees {money['fees']:,.2f}   tax {money['tax']:,.2f}"
        f"   net payout {money['net_payout']:,.2f}",
        "",
        "EXCEPTIONS",
        "-" * 64,
        breakdown.to_string(index=False) if len(breakdown) else "none",
    ]
    return "\n".join(lines)
