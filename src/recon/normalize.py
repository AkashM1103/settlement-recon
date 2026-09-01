"""Schema normalization: types, currency casing, dates, id whitespace."""

from __future__ import annotations

import pandas as pd

from .ingest import Batch

ORDER_ALIASES = {
    "orderid": "order_id",
    "order_no": "order_id",
    "payment_id": "razorpay_payment_id",
    "rzp_payment_id": "razorpay_payment_id",
    "order_amount": "amount",
}
SETTLEMENT_ALIASES = {
    "settlementid": "settlement_id",
    "razorpay_payment_id": "payment_id",
    "gross": "gross_amount",
    "net": "net_amount",
}


def _rename(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})
    return df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})


def _to_float(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[₹,\s]", "", regex=True)
        .replace({"": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.strip(), errors="coerce", format="mixed").dt.normalize()


def normalize_orders(orders: pd.DataFrame) -> pd.DataFrame:
    df = _rename(orders.copy(), ORDER_ALIASES)
    df["order_id"] = df["order_id"].astype(str).str.strip()
    df["razorpay_payment_id"] = df["razorpay_payment_id"].astype(str).str.strip()
    df["customer_ref"] = df["customer_ref"].astype(str).str.strip()
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["amount"] = _to_float(df["amount"])
    df["order_date"] = _to_date(df["order_date"])
    return df


def normalize_settlements(settlements: pd.DataFrame) -> pd.DataFrame:
    df = _rename(settlements.copy(), SETTLEMENT_ALIASES)
    df["settlement_id"] = df["settlement_id"].astype(str).str.strip()
    df["payment_id"] = df["payment_id"].astype(str).str.strip().replace({"nan": "", "None": ""})
    df["utr"] = df["utr"].astype(str).str.strip()
    for col in ("gross_amount", "fees", "tax", "net_amount"):
        df[col] = _to_float(df[col])
    df["settlement_date"] = _to_date(df["settlement_date"])
    # derived integrity flag: does gross - fees - tax reconcile to net?
    df["net_reconciles"] = (
        (df["gross_amount"] - df["fees"] - df["tax"] - df["net_amount"]).abs() <= 0.05
    )
    return df


def normalize_batch(batch: Batch) -> Batch:
    return Batch(
        orders=normalize_orders(batch.orders),
        settlements=normalize_settlements(batch.settlements),
        ground_truth=batch.ground_truth,
        warnings=batch.warnings,
    )
