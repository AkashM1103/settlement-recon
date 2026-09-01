"""End-to-end pipeline wiring: ingest -> normalize -> match -> report -> Q&A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .ingest import Batch, load_batch, load_frames
from .match import MatchConfig, MatchOutcome, run_matching
from .normalize import normalize_batch
from .qa import ReconQA
from .report import Metrics, compute_metrics, exception_breakdown, financial_summary

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Reconciliation:
    batch: Batch
    outcome: MatchOutcome
    metrics: Metrics
    breakdown: pd.DataFrame
    money: dict[str, float]

    @property
    def matches(self) -> pd.DataFrame:
        return self.outcome.matches

    def qa(self) -> ReconQA:
        return ReconQA(self.outcome, self.metrics)


def reconcile(batch: Batch, config: MatchConfig | None = None, use_llm: bool = True) -> Reconciliation:
    clean = normalize_batch(batch)
    outcome = run_matching(clean, config=config, use_llm=use_llm)
    metrics = compute_metrics(clean, outcome)
    return Reconciliation(
        batch=clean,
        outcome=outcome,
        metrics=metrics,
        breakdown=exception_breakdown(outcome),
        money=financial_summary(outcome),
    )


def reconcile_paths(
    orders_path: str | Path = DATA_DIR / "orders.csv",
    settlements_path: str | Path = DATA_DIR / "settlements.csv",
    ground_truth_path: str | Path | None = DATA_DIR / "ground_truth.csv",
    config: MatchConfig | None = None,
    use_llm: bool = True,
) -> Reconciliation:
    batch = load_batch(orders_path, settlements_path, ground_truth_path)
    return reconcile(batch, config=config, use_llm=use_llm)


def reconcile_frames(orders: pd.DataFrame, settlements: pd.DataFrame,
                     ground_truth: pd.DataFrame | None = None,
                     config: MatchConfig | None = None, use_llm: bool = True) -> Reconciliation:
    return reconcile(load_frames(orders, settlements, ground_truth), config=config, use_llm=use_llm)
