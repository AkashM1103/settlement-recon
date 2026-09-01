"""Generate synthetic orders.csv + settlements.csv with deliberate messy cases.

Ground truth (which settlement really belongs to which order, and why a record is
an exception) is written to data/ground_truth.csv so the pipeline's false-match
rate can be measured.
"""

from __future__ import annotations

import argparse
import random
import string
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FEE_RATE = 0.0236  # 2.36% platform fee
TAX_RATE = 0.18  # GST on the fee

CLEAN_CASES = 60
NOISE_PLAN = {
    "orphan_settlement": 3,  # settlement without payment_id link
    "amount_mismatch": 3,  # partial refund not reflected on settlement
    "duplicate_settlement": 2,  # same payment_id settled twice
    "no_order": 2,  # test/dummy transaction, no order at all
    "date_drift": 3,  # settled far later than the order
    "unsettled_order": 3,  # order that never produced a settlement
}


def _payment_id(rng: random.Random) -> str:
    alphabet = string.ascii_letters + string.digits
    return "pay_" + "".join(rng.choice(alphabet) for _ in range(14))


def _fees_for(gross: float) -> tuple[float, float, float]:
    fees = round(gross * FEE_RATE, 2)
    tax = round(fees * TAX_RATE, 2)
    net = round(gross - fees - tax, 2)
    return fees, tax, net


def generate(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)

    orders: list[dict] = []
    settlements: list[dict] = []
    truth: list[dict] = []

    base_day = date(2026, 8, 1)
    order_seq = 10000
    stl_seq = 5500

    def new_order(status: str = "paid", day_offset: int | None = None) -> dict:
        nonlocal order_seq
        order_seq += 1
        offset = rng.randint(0, 20) if day_offset is None else day_offset
        return {
            "order_id": f"ORD{order_seq}",
            "amount": round(rng.uniform(199, 24999), 2),
            "currency": "INR",
            "order_date": (base_day + timedelta(days=offset)).isoformat(),
            "customer_ref": f"CUST{rng.randint(1, 99999):05d}",
            "status": status,
            "razorpay_payment_id": _payment_id(rng),
        }

    def new_settlement(order: dict, *, gross: float | None = None, lag: int = 2,
                       link_payment: bool = True) -> dict:
        nonlocal stl_seq
        stl_seq += 1
        gross_amount = order["amount"] if gross is None else gross
        fees, tax, net = _fees_for(gross_amount)
        settled = date.fromisoformat(order["order_date"]) + timedelta(days=lag)
        return {
            "settlement_id": f"STL{stl_seq}",
            "payment_id": order["razorpay_payment_id"] if link_payment else "",
            "gross_amount": gross_amount,
            "fees": fees,
            "tax": tax,
            "net_amount": net,
            "settlement_date": settled.isoformat(),
            "utr": f"UTR{rng.randint(1000000, 9999999)}",
        }

    def record(settlement: dict, order: dict | None, label: str, note: str) -> None:
        truth.append(
            {
                "settlement_id": settlement["settlement_id"],
                "true_order_id": order["order_id"] if order else "",
                "case": label,
                "note": note,
            }
        )

    # --- clean, unambiguous 1:1 pairs -------------------------------------
    for _ in range(CLEAN_CASES):
        order = new_order(status=rng.choices(["paid", "refunded"], weights=[0.93, 0.07])[0])
        orders.append(order)
        if order["status"] == "refunded":
            # refunded orders are still settled gross, then debited separately
            stl = new_settlement(order, lag=rng.randint(1, 4))
            settlements.append(stl)
            record(stl, order, "clean", "exact payment_id match")
        else:
            stl = new_settlement(order, lag=rng.randint(1, 4))
            settlements.append(stl)
            record(stl, order, "clean", "exact payment_id match")

    # --- orphan settlements: link dropped, order still exists -------------
    for _ in range(NOISE_PLAN["orphan_settlement"]):
        order = new_order()
        orders.append(order)
        stl = new_settlement(order, lag=rng.randint(1, 3), link_payment=False)
        settlements.append(stl)
        record(stl, order, "orphan_settlement", "payment_id missing; recoverable via amount+date")

    # --- amount mismatch: partial refund not reflected --------------------
    for _ in range(NOISE_PLAN["amount_mismatch"]):
        order = new_order(status="partial_refund")
        orders.append(order)
        refunded = round(order["amount"] * rng.uniform(0.2, 0.45), 2)
        stl = new_settlement(order, gross=round(order["amount"] - refunded, 2), lag=rng.randint(2, 5))
        settlements.append(stl)
        record(stl, order, "amount_mismatch", f"partial refund of {refunded} not reflected in order amount")

    # --- duplicate settlements for the same payment -----------------------
    for _ in range(NOISE_PLAN["duplicate_settlement"]):
        order = new_order()
        orders.append(order)
        first = new_settlement(order, lag=2)
        second = new_settlement(order, lag=3)
        settlements.extend([first, second])
        record(first, order, "clean", "first settlement for payment")
        record(second, order, "duplicate_settlement", "same payment_id settled twice")

    # --- settlement with no order at all (test/dummy) ---------------------
    for _ in range(NOISE_PLAN["no_order"]):
        ghost = new_order()  # not appended to orders
        stl = new_settlement(ghost, lag=1, link_payment=True)
        settlements.append(stl)
        record(stl, None, "no_order", "test/dummy transaction, no merchant order exists")

    # --- date drift: settled more than 7 days after the order -------------
    for _ in range(NOISE_PLAN["date_drift"]):
        order = new_order()
        orders.append(order)
        stl = new_settlement(order, lag=rng.randint(9, 21))
        settlements.append(stl)
        record(stl, order, "date_drift", "settled >7 days after order; delayed, not broken")

    # --- orders that never got settled ------------------------------------
    for _ in range(NOISE_PLAN["unsettled_order"]):
        orders.append(new_order(day_offset=rng.randint(18, 25)))

    rng.shuffle(orders)
    rng.shuffle(settlements)

    orders_df = pd.DataFrame(orders)
    # a little schema messiness for the normalizer to clean up
    orders_df.loc[orders_df.sample(frac=0.15, random_state=seed).index, "currency"] = "inr"
    orders_df["merchant"] = [fake.company() for _ in range(len(orders_df))]

    return orders_df, pd.DataFrame(settlements), pd.DataFrame(truth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    orders, settlements, truth = generate(args.seed)
    orders.to_csv(args.out / "orders.csv", index=False)
    settlements.to_csv(args.out / "settlements.csv", index=False)
    truth.to_csv(args.out / "ground_truth.csv", index=False)

    print(f"orders: {len(orders)} rows -> {args.out / 'orders.csv'}")
    print(f"settlements: {len(settlements)} rows -> {args.out / 'settlements.csv'}")
    print(truth["case"].value_counts().to_string())


if __name__ == "__main__":
    main()
