import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from recon import MatchConfig, load_batch, reconcile  # noqa: E402
from recon.normalize import normalize_settlements  # noqa: E402
from recon.reasoner import HeuristicReasoner  # noqa: E402

DATA = ROOT / "data"


@pytest.fixture(scope="module")
def result():
    batch = load_batch(DATA / "orders.csv", DATA / "settlements.csv", DATA / "ground_truth.csv")
    return reconcile(batch, config=MatchConfig(), use_llm=False)


def test_every_settlement_gets_a_row(result):
    assert len(result.matches) == len(result.batch.settlements)
    assert result.matches["settlement_id"].is_unique


def test_match_rate_is_plausible(result):
    assert 0.80 <= result.metrics.match_rate <= 0.99


def test_no_false_matches_against_ground_truth(result):
    assert result.metrics.false_matches == 0


def test_all_injected_exception_types_are_detected(result):
    found = set(result.breakdown["exception_reason"])
    assert {"orphan_settlement", "amount_mismatch", "duplicate_settlement", "no_order",
            "date_drift", "unsettled_order"} <= found


def test_every_exception_has_a_reason(result):
    exc = result.matches[result.matches["exception_reason"].notna()]
    assert (exc["reason"].str.len() > 0).all()


def test_normalizer_fixes_currency_and_types():
    df = pd.DataFrame(
        [{"settlement_id": " STL1 ", "payment_id": "pay_a ", "gross_amount": "₹1,000.00",
          "fees": "23.60", "tax": "4.25", "net_amount": "972.15",
          "settlement_date": "2026-08-14", "utr": "UTR1"}]
    )
    out = normalize_settlements(df)
    assert out.loc[0, "settlement_id"] == "STL1"
    assert out.loc[0, "gross_amount"] == 1000.0
    assert bool(out.loc[0, "net_reconciles"])


def test_heuristic_reasoner_rejects_far_candidates():
    verdict = HeuristicReasoner().judge(
        {"gross_amount": 5000.0},
        [{"order_id": "ORD1", "amount": 12000.0, "day_gap": 12}],
    )
    assert verdict.verdict == "no_match"


def test_qa_answers_are_grounded_and_cited(result):
    qa = result.qa()
    answer = qa.ask("Which settlements have no matching order?")
    orphans = set(result.matches.loc[result.matches["exception_reason"] == "no_order", "settlement_id"])
    assert orphans and orphans <= set(answer.cited_records)
    tax = qa.ask("What's the total tax deducted this batch?")
    assert f"{result.money['tax']:,.2f}" in tax.answer
