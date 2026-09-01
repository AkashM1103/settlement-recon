"""Verdict layer: decide match / no_match / uncertain for fuzzy candidates.

The LLM sees one settlement plus its top candidate orders and returns a verdict,
a confidence score and a one-line reason. When no API key is available the same
interface is served by a deterministic scorer so the pipeline never hard-fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .llm import ClaudeClient

SYSTEM_PROMPT = (
    "You are a payments reconciliation analyst. Given one Razorpay settlement and a "
    "shortlist of candidate merchant orders, decide whether the settlement belongs to "
    "one of the candidates. Settlement gross_amount can be lower than the order amount "
    "when a partial refund happened. Settlement date is always on or after the order "
    "date, usually 1-4 days later; more than 7 days is delayed but still plausible. "
    "Reply with JSON only: "
    '{"verdict": "match|no_match|uncertain", "order_id": "<id or null>", '
    '"confidence": <0..1>, "reason": "<one line>"}'
)


@dataclass
class Verdict:
    verdict: str  # match | no_match | uncertain
    order_id: str | None
    confidence: float
    reason: str
    source: str  # llm | heuristic


class Reasoner(Protocol):
    name: str

    def judge(self, settlement: dict[str, Any], candidates: list[dict[str, Any]]) -> Verdict: ...


def _score(settlement: dict[str, Any], candidate: dict[str, Any]) -> float:
    amount_gap = abs(settlement["gross_amount"] - candidate["amount"]) / max(candidate["amount"], 1)
    day_gap = candidate.get("day_gap", 99)
    amount_score = max(0.0, 1 - amount_gap / 0.5)
    date_score = max(0.0, 1 - max(day_gap - 1, 0) / 14)
    return round(0.65 * amount_score + 0.35 * date_score, 3)


class HeuristicReasoner:
    """Deterministic fallback used when no LLM key is configured."""

    name = "heuristic"

    def judge(self, settlement: dict[str, Any], candidates: list[dict[str, Any]]) -> Verdict:
        if not candidates:
            return Verdict("no_match", None, 0.9, "no candidate order within amount/date window", self.name)

        ranked = sorted(candidates, key=lambda c: _score(settlement, c), reverse=True)
        best = ranked[0]
        confidence = _score(settlement, best)
        runner_up = _score(settlement, ranked[1]) if len(ranked) > 1 else 0.0
        margin = confidence - runner_up

        if confidence >= 0.85 and margin >= 0.05:
            reason = (
                f"amount within {abs(settlement['gross_amount'] - best['amount']):.2f} and "
                f"{best.get('day_gap')}d settlement lag of {best['order_id']}"
            )
            return Verdict("match", best["order_id"], confidence, reason, self.name)
        if confidence >= 0.6:
            return Verdict(
                "uncertain",
                best["order_id"],
                confidence,
                f"plausible but ambiguous against {len(candidates)} candidate(s); needs review",
                self.name,
            )
        return Verdict("no_match", None, round(1 - confidence, 3),
                       "no candidate close enough on amount and date", self.name)


class LLMReasoner:
    """Claude-backed reasoner; falls back per-call if the API errors."""

    name = "llm"

    def __init__(self, client: ClaudeClient | None = None) -> None:
        self.client = client or ClaudeClient()
        self.fallback = HeuristicReasoner()

    @property
    def available(self) -> bool:
        return self.client.available

    def judge(self, settlement: dict[str, Any], candidates: list[dict[str, Any]]) -> Verdict:
        if not self.available or not candidates:
            return self.fallback.judge(settlement, candidates)
        prompt = (
            "SETTLEMENT:\n"
            + json.dumps(settlement, default=str, indent=2)
            + "\n\nCANDIDATE ORDERS:\n"
            + json.dumps(candidates, default=str, indent=2)
        )
        try:
            data = self.client.complete_json(SYSTEM_PROMPT, prompt)
            verdict = str(data.get("verdict", "uncertain")).lower()
            if verdict not in {"match", "no_match", "uncertain"}:
                verdict = "uncertain"
            order_id = data.get("order_id") or None
            confidence = float(data.get("confidence", 0.5))
            reason = str(data.get("reason", "")).strip() or "model gave no reason"
            valid_ids = {c["order_id"] for c in candidates}
            if verdict == "match" and order_id not in valid_ids:
                return Verdict("uncertain", None, min(confidence, 0.5),
                               "model proposed an order outside the candidate set", self.name)
            return Verdict(verdict, order_id, round(max(0.0, min(confidence, 1.0)), 3), reason, self.name)
        except Exception as exc:  # noqa: BLE001 - degrade rather than abort the batch
            verdict = self.fallback.judge(settlement, candidates)
            return Verdict(verdict.verdict, verdict.order_id, verdict.confidence,
                           f"{verdict.reason} (llm error: {type(exc).__name__})", "heuristic")


def build_reasoner(use_llm: bool = True) -> Reasoner:
    if use_llm:
        reasoner = LLMReasoner()
        if reasoner.available:
            return reasoner
    return HeuristicReasoner()
