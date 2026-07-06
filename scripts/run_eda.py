"""Entry point: generate all EDA figures and compute headline insights.

Usage:
    python scripts/run_eda.py

Saves PNGs to reports/figures/ and writes docs/phase2_insights.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.evaluation import plots
from src.evaluation.insights import compute_eda_insights

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "processed" / "cleaned.parquet"
FIG = ROOT / "reports" / "figures"
INSIGHTS = ROOT / "docs" / "phase2_insights.md"


def main() -> None:
    df = pd.read_parquet(DATA)
    print(f"Loaded {len(df):,} rows")
    FIG.mkdir(parents=True, exist_ok=True)

    figures = {
        "01_price_distribution.png": plots.plot_price_distribution,
        "02_depreciation.png": plots.plot_depreciation,
        "03_odometer_vs_price.png": plots.plot_odometer_vs_price,
        "04_price_by_category.png": plots.plot_price_by_category,
        "05_manufacturer_median_price.png": plots.plot_manufacturer_median_price,
        "06_state_median_price.png": plots.plot_state_median_price,
        "07_correlation_heatmap.png": plots.plot_correlation_heatmap,
        "08_value_heaping.png": plots.plot_value_heaping,
        "09_confound_check.png": plots.plot_confound_check,
        "10_age_odometer_interaction.png": plots.plot_age_odometer_interaction,
        "11_missingness_and_cardinality.png": plots.plot_missingness_and_cardinality,
    }
    for name, fn in figures.items():
        fig = fn(df, save_path=FIG / name)
        plt.close(fig)
        print(f"  saved {name}")

    INSIGHTS.write_text(compute_eda_insights(df), encoding="utf-8")
    print(f"Insights written to {INSIGHTS}")


if __name__ == "__main__":
    main()
