"""Grounded Q&A over the reconciled batch.

Answers are computed from the reconciled dataframe (never free-form recall): a
retrieval step selects the relevant records, deterministic aggregates are
computed in pandas, and the LLM only phrases the answer over that evidence. Every
answer carries the record IDs it used, which is the audit trail.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from .llm import LLMClient
from .match import MATCHED, MatchOutcome
from .report import Metrics, exception_breakdown, financial_summary

SYSTEM_PROMPT = (
    "You are a settlement reconciliation analyst answering questions about ONE batch. "
    "Answer only from the EVIDENCE block; never invent numbers. Amounts are INR. "
    "Be concise (1-3 sentences), quote exact figures, and end with the record IDs you "
    "relied on, written plainly (e.g. 'Records: STL5571, STL5572') with no bracketed "
    "source markers. If the evidence does not contain the answer, say so plainly."
)

EXCEPTION_WORDS = {
    "orphan": "orphan_settlement",
    "duplicate": "duplicate_settlement",
    "mismatch": "amount_mismatch",
    "refund": "amount_mismatch",
    "delay": "date_drift",
    "drift": "date_drift",
    "late": "date_drift",
    "dummy": "no_order",
    "test": "no_order",
    "no matching order": "no_order",
    "unsettled": "unsettled_order",
    "ambiguous": "ambiguous_candidates",
}

ID_PATTERN = re.compile(r"\b(?:STL\d+|ORD\d+|pay_[A-Za-z0-9]+|UTR\d+)\b")


@dataclass
class Answer:
    question: str
    answer: str
    cited_records: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    source: str = "rules"


class ReconQA:
    def __init__(self, outcome: MatchOutcome, metrics: Metrics, client: LLMClient | None = None,
                 max_records: int = 25) -> None:
        self.outcome = outcome
        self.metrics = metrics
        self.matches = outcome.matches
        self.client = client or LLMClient()
        self.max_records = max_records

    # ---------------- retrieval ------------------------------------------
    def retrieve(self, question: str) -> pd.DataFrame:
        q = question.lower()
        m = self.matches

        explicit_ids = set(ID_PATTERN.findall(question))
        if explicit_ids:
            hit = m[
                m["settlement_id"].isin(explicit_ids)
                | m["order_id"].fillna("").isin(explicit_ids)
                | m["payment_id"].isin(explicit_ids)
            ]
            if len(hit):
                return hit

        for word, reason in EXCEPTION_WORDS.items():
            if word in q:
                if reason == "unsettled_order":
                    return self.outcome.unsettled_orders
                return m[m["exception_reason"] == reason]

        if any(w in q for w in ("unmatched", "no matching", "exception", "failed", "problem")):
            return m[m["exception_reason"].notna() | (m["match_status"] != MATCHED)]
        if any(w in q for w in ("uncertain", "review", "manual")):
            return m[m["match_status"] == "uncertain"]
        if "largest" in q or "biggest" in q or "top" in q:
            return m.sort_values("gross_amount", ascending=False).head(self.max_records)
        return m

    # ---------------- evidence -------------------------------------------
    def build_evidence(self, question: str) -> dict:
        subset = self.retrieve(question)
        money = financial_summary(self.outcome)
        numeric = [c for c in ("gross_amount", "fees", "tax", "net_amount") if c in subset.columns]
        cols = [
            c
            for c in (
                "settlement_id", "order_id", "payment_id", "match_status", "match_type",
                "confidence", "exception_reason", "gross_amount", "fees", "tax",
                "net_amount", "amount_delta", "settlement_date", "order_date",
                "settlement_lag_days", "order_status", "reason", "amount",
            )
            if c in subset.columns
        ]
        return {
            "batch_metrics": self.metrics.as_dict(),
            "batch_money_totals": money,
            "exception_breakdown": exception_breakdown(self.outcome).to_dict("records"),
            "retrieved_record_count": len(subset),
            "retrieved_subset_totals": {c: round(float(subset[c].sum()), 2) for c in numeric},
            "retrieved_records": subset[cols].head(self.max_records).to_dict("records"),
        }

    @staticmethod
    def _ids(evidence: dict) -> list[str]:
        ids = []
        for r in evidence["retrieved_records"]:
            ids.append(str(r.get("settlement_id") or r.get("order_id")))
        return ids

    # ---------------- answering ------------------------------------------
    def ask(self, question: str) -> Answer:
        evidence = self.build_evidence(question)
        ids = self._ids(evidence)
        if self.client.available:
            try:
                text = self.client.complete(
                    SYSTEM_PROMPT,
                    f"QUESTION: {question}\n\nEVIDENCE:\n{json.dumps(evidence, default=str, indent=2)}",
                ).strip()
                return Answer(question, text, ids, evidence, source="llm")
            except Exception as exc:  # noqa: BLE001 - fall back to deterministic answer
                fallback = self._rule_answer(question, evidence)
                return Answer(question, f"{fallback} (llm unavailable: {type(exc).__name__})",
                              ids, evidence, source="rules")
        return Answer(question, self._rule_answer(question, evidence), ids, evidence, source="rules")

    def _rule_answer(self, question: str, evidence: dict) -> str:
        q = question.lower()
        money = evidence["batch_money_totals"]
        subset_totals = evidence["retrieved_subset_totals"]
        n = evidence["retrieved_record_count"]
        met = evidence["batch_metrics"]

        if "tax" in q:
            return f"Tax deducted across the {n} retrieved record(s): INR {subset_totals.get('tax', 0):,.2f} (batch total INR {money['tax']:,.2f})."
        if "fee" in q or "commission" in q:
            return f"Fees across the {n} retrieved record(s): INR {subset_totals.get('fees', 0):,.2f} (batch total INR {money['fees']:,.2f})."
        if "net" in q or "payout" in q:
            return f"Net payout across the {n} retrieved record(s): INR {subset_totals.get('net_amount', 0):,.2f} (batch total INR {money['net_payout']:,.2f})."
        if "gross" in q or "total amount" in q:
            return f"Gross across the {n} retrieved record(s): INR {subset_totals.get('gross_amount', 0):,.2f} (batch total INR {money['gross_amount']:,.2f})."
        if "match rate" in q or "how many matched" in q:
            return (
                f"Match rate is {met['match_rate']:.1%} ({met['matched']} matched, "
                f"{met['uncertain']} uncertain, {met['unmatched']} unmatched of {met['settlements']} settlements)."
            )
        rows = evidence["retrieved_records"][:10]
        listed = ", ".join(str(r.get("settlement_id") or r.get("order_id")) for r in rows)
        return f"{n} record(s) match this question: {listed}{' ...' if n > len(rows) else ''}."


SAMPLE_QUESTIONS = [
    "What's the total tax deducted this batch?",
    "Which settlements have no matching order?",
    "How much did we pay in fees, and what was the net payout?",
    "Are there duplicate settlements for the same payment?",
    "Which settlements were delayed more than 7 days?",
    "What is the match rate for this batch?",
]
