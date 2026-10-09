"""Normalize input values before reconciliation."""

from __future__ import annotations

import pandas as pd

from .ingest import Batch, ORDER_ALIASES, SETTLEMENT_ALIASES, validate_batch_ids


def _rename(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    names = {column: str(column).strip().lower().replace(" ", "_") for column in df.columns}
    df = df.rename(columns=names)
    changes = {old: new for old, new in aliases.items() if old in df.columns and new not in df.columns}
    return df.rename(columns=changes)


def _to_float(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[\u20b9,\s]", "", regex=True)
    cleaned = cleaned.replace({"": None, "nan": None, "None": None})
    return pd.to_numeric(cleaned, errors="coerce")


def _to_date(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    return pd.to_datetime(values, errors="coerce", format="mixed").dt.normalize()


def normalize_orders(orders: pd.DataFrame) -> pd.DataFrame:
    df = _rename(orders.copy(), ORDER_ALIASES)
    for column in ("order_id", "razorpay_payment_id", "customer_ref"):
        df[column] = df[column].astype(str).str.strip()
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
    for column in ("gross_amount", "fees", "tax", "net_amount"):
        df[column] = _to_float(df[column])
    df["settlement_date"] = _to_date(df["settlement_date"])
    expected_net = df["gross_amount"] - df["fees"] - df["tax"]
    df["net_reconciles"] = (expected_net - df["net_amount"]).abs() <= 0.05
    return df


def normalize_batch(batch: Batch) -> Batch:
    orders = normalize_orders(batch.orders)
    settlements = normalize_settlements(batch.settlements)
    clean = Batch(orders, settlements, batch.ground_truth, list(batch.warnings))
    validate_batch_ids(clean)

    required_values = {
        "orders": (orders, ["amount", "order_date"]),
        "settlements": (settlements, ["gross_amount", "fees", "tax", "net_amount", "settlement_date"]),
    }
    for name, (frame, columns) in required_values.items():
        invalid = frame[columns].isna().any(axis=1)
        if invalid.any():
            rows = [str(index + 2) for index, bad in enumerate(invalid.tolist()) if bad][:5]
            raise ValueError(f"{name} has an invalid amount or date on CSV row(s): {', '.join(rows)}")

    currencies = set(orders["currency"].dropna().unique())
    if currencies != {"INR"}:
        found = ", ".join(sorted(currencies)) or "blank"
        raise ValueError(f"This demo reports amounts in INR and cannot convert currency; found: {found}")

    bad_net_count = int((~settlements["net_reconciles"]).sum())
    if bad_net_count:
        clean.warnings.append(
            f"{bad_net_count} settlement row(s) do not satisfy gross - fees - tax = net"
        )
    return clean
