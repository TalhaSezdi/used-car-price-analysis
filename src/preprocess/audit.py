"""Initial dataset audit — run once before cleaning to document raw state."""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = Path(__file__).parents[2] / "data" / "raw" / "vehicles.csv"
AUDIT_OUT = Path(__file__).parents[2] / "docs" / "phase1_audit.md"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    return (
        pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        .sort_values("missing_pct", ascending=False)
    )


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])


def run_audit() -> None:
    print("Loading raw data...")
    df = load_raw()

    n_rows, n_cols = df.shape
    print(f"Shape: {n_rows:,} rows x {n_cols} columns")

    miss = missing_summary(df)
    num_summary = numeric_summary(df, ["price", "odometer", "year"])

    value_counts: dict[str, pd.Series] = {}
    for col in ["manufacturer", "condition", "fuel", "transmission", "drive",
                "type", "title_status", "cylinders", "paint_color", "state"]:
        if col in df.columns:
            value_counts[col] = df[col].value_counts(dropna=False).head(15)

    lines: list[str] = []
    lines.append("# Phase 1 Audit — Raw Dataset\n")
    lines.append(f"**Rows:** {n_rows:,}  \n**Columns:** {n_cols}  \n")
    lines.append("\n## Missing Values\n")
    lines.append(miss.to_markdown())
    lines.append("\n\n## Numeric Distributions (price, odometer, year)\n")
    lines.append(num_summary.to_markdown())
    lines.append("\n\n## Categorical Value Counts (top 15 each)\n")
    for col, vc in value_counts.items():
        lines.append(f"\n### {col}\n")
        lines.append(vc.to_markdown())
        lines.append("\n")

    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit written to {AUDIT_OUT}")

    print("\n--- Missing value summary (top 10) ---")
    print(miss.head(10).to_string())
    print("\n--- Price / Odometer / Year ---")
    print(num_summary.to_string())


if __name__ == "__main__":
    run_audit()
