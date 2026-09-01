"""CSV ingestion with basic sanity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ORDER_COLUMNS = {
    "order_id",
    "amount",
    "currency",
    "order_date",
    "customer_ref",
    "status",
    "razorpay_payment_id",
}
SETTLEMENT_COLUMNS = {
    "settlement_id",
    "payment_id",
    "gross_amount",
    "fees",
    "tax",
    "net_amount",
    "settlement_date",
    "utr",
}


@dataclass
class Batch:
    orders: pd.DataFrame
    settlements: pd.DataFrame
    ground_truth: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "orders": len(self.orders),
            "settlements": len(self.settlements),
            "settlements_missing_payment_id": int(
                self.settlements["payment_id"].isna().sum()
                + (self.settlements["payment_id"].astype(str).str.strip() == "").sum()
            ),
            "duplicate_payment_ids": int(
                self.settlements["payment_id"].replace("", pd.NA).dropna().duplicated().sum()
            ),
        }


def _check_columns(df: pd.DataFrame, required: set[str], name: str) -> list[str]:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")
    return []


def load_batch(
    orders_path: str | Path,
    settlements_path: str | Path,
    ground_truth_path: str | Path | None = None,
) -> Batch:
    orders = pd.read_csv(orders_path, dtype=str, keep_default_na=False)
    settlements = pd.read_csv(settlements_path, dtype=str, keep_default_na=False)
    _check_columns(orders, ORDER_COLUMNS, "orders.csv")
    _check_columns(settlements, SETTLEMENT_COLUMNS, "settlements.csv")

    truth = None
    if ground_truth_path and Path(ground_truth_path).exists():
        truth = pd.read_csv(ground_truth_path, dtype=str, keep_default_na=False)

    warnings: list[str] = []
    if orders["order_id"].duplicated().any():
        warnings.append("duplicate order_id values present in orders.csv")
    if settlements["settlement_id"].duplicated().any():
        warnings.append("duplicate settlement_id values present in settlements.csv")

    return Batch(orders=orders, settlements=settlements, ground_truth=truth, warnings=warnings)


def load_frames(orders: pd.DataFrame, settlements: pd.DataFrame,
                ground_truth: pd.DataFrame | None = None) -> Batch:
    """Build a Batch from in-memory frames (used by the Streamlit uploader)."""
    _check_columns(orders, ORDER_COLUMNS, "orders")
    _check_columns(settlements, SETTLEMENT_COLUMNS, "settlements")
    return Batch(orders=orders.astype(str), settlements=settlements.astype(str),
                 ground_truth=ground_truth)
