"""Read input tables and check their basic shape and IDs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ORDER_COLUMNS = {
    "order_id", "amount", "currency", "order_date", "customer_ref",
    "status", "razorpay_payment_id",
}
SETTLEMENT_COLUMNS = {
    "settlement_id", "payment_id", "gross_amount", "fees", "tax",
    "net_amount", "settlement_date", "utr",
}
ORDER_ALIASES = {
    "orderid": "order_id", "order_no": "order_id",
    "payment_id": "razorpay_payment_id", "rzp_payment_id": "razorpay_payment_id",
    "order_amount": "amount",
}
SETTLEMENT_ALIASES = {
    "settlementid": "settlement_id", "razorpay_payment_id": "payment_id",
    "gross": "gross_amount", "net": "net_amount",
}


@dataclass
class Batch:
    orders: pd.DataFrame
    settlements: pd.DataFrame
    ground_truth: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        payment_ids = self.settlements["payment_id"].astype(str).str.strip()
        payment_ids = payment_ids[payment_ids != ""]
        return {
            "orders": len(self.orders),
            "settlements": len(self.settlements),
            "settlements_missing_payment_id": int(
                (self.settlements["payment_id"].astype(str).str.strip() == "").sum()
            ),
            "duplicate_payment_ids": int(payment_ids.duplicated().sum()),
        }


def _standardize_columns(df: pd.DataFrame, aliases: dict[str, str], name: str) -> pd.DataFrame:
    """Make column names consistent and apply the documented simple aliases."""
    names = [str(column).strip().lower().replace(" ", "_") for column in df.columns]
    if len(names) != len(set(names)):
        raise ValueError(f"{name} has duplicate column names after cleanup")
    df = df.copy()
    df.columns = names
    for alias, standard in aliases.items():
        if alias in df.columns:
            if standard in df.columns:
                raise ValueError(f"{name} has both '{alias}' and '{standard}' columns")
            df = df.rename(columns={alias: standard})
    return df


def _check_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")
    if df.empty:
        raise ValueError(f"{name} has no data rows")


def validate_batch_ids(batch: Batch) -> None:
    """Reject IDs that would make a match ambiguous or a report unreliable."""
    checks = [
        (batch.orders, "order_id", "orders", True),
        (batch.orders, "razorpay_payment_id", "orders", False),
        (batch.settlements, "settlement_id", "settlements", True),
    ]
    for frame, column, table, required in checks:
        values = frame[column].astype(str).str.strip()
        if required and (values == "").any():
            raise ValueError(f"{table}.{column} cannot be blank")
        values = values[values != ""]
        duplicates = values[values.duplicated()].unique().tolist()
        if duplicates:
            examples = ", ".join(duplicates[:3])
            raise ValueError(f"{table}.{column} must be unique; duplicate value(s): {examples}")

    if batch.ground_truth is not None:
        _check_columns(batch.ground_truth, {"settlement_id", "true_order_id"}, "ground_truth")
        if batch.ground_truth["settlement_id"].duplicated().any():
            raise ValueError("ground_truth.settlement_id must be unique")


def load_batch(
    orders_path: str | Path,
    settlements_path: str | Path,
    ground_truth_path: str | Path | None = None,
) -> Batch:
    orders = pd.read_csv(orders_path, dtype=str, keep_default_na=False)
    settlements = pd.read_csv(settlements_path, dtype=str, keep_default_na=False)
    truth = None
    if ground_truth_path and Path(ground_truth_path).exists():
        truth = pd.read_csv(ground_truth_path, dtype=str, keep_default_na=False)
    return load_frames(orders, settlements, truth)


def load_frames(
    orders: pd.DataFrame,
    settlements: pd.DataFrame,
    ground_truth: pd.DataFrame | None = None,
) -> Batch:
    """Build a checked Batch from CSV uploads or in-memory tables."""
    orders = _standardize_columns(orders, ORDER_ALIASES, "orders")
    settlements = _standardize_columns(settlements, SETTLEMENT_ALIASES, "settlements")
    _check_columns(orders, ORDER_COLUMNS, "orders")
    _check_columns(settlements, SETTLEMENT_COLUMNS, "settlements")

    orders = orders.fillna("").astype(str)
    settlements = settlements.fillna("").astype(str)
    if ground_truth is not None:
        ground_truth = _standardize_columns(ground_truth, {}, "ground_truth")
        ground_truth = ground_truth.fillna("").astype(str)

    batch = Batch(orders=orders, settlements=settlements, ground_truth=ground_truth)
    validate_batch_ids(batch)
    return batch
