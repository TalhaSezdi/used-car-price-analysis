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

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "processed" / "cleaned.parquet"
FIG = ROOT / "reports" / "figures"
INSIGHTS = ROOT / "docs" / "phase2_insights.md"


def compute_insights(df: pd.DataFrame) -> str:
    top_brands = df["manufacturer"].value_counts().head(6).index
    med_by_brand = df.groupby("manufacturer")["price"].median()

    # depreciation: median price at age 1 vs age 10 for the market
    age1 = df[df["age"] == 1]["price"].median()
    age5 = df[df["age"] == 5]["price"].median()
    age10 = df[df["age"] == 10]["price"].median()

    state_med = df.groupby("state")["price"].median().sort_values(ascending=False)
    corr = df[["price", "age", "odometer", "mileage_per_year", "cylinders_num"]].corr()["price"]

    # experimental / behavioral angles
    price_end = df["price"].astype(int) % 1000
    pct_995 = (price_end == 995).mean()
    pct_round_price = (price_end == 0).mean()
    pct_round_odo = (df["odometer"].astype(int) % 1000 == 0).mean()

    d = df.copy()
    d["has_vin"] = d["VIN"].notna() & (d["VIN"].str.strip() != "")
    d["cond_missing"] = d["condition"].isna()
    d["age_bucket"] = pd.cut(d["age"], [0, 3, 6, 10, 15, 60],
                             labels=["0-3", "4-6", "7-10", "11-15", "16+"])
    vin_raw = d.groupby("has_vin")["price"].median()
    vin_ratio_raw = vin_raw[True] / vin_raw[False]
    vin_piv = d.pivot_table("price", "age_bucket", "has_vin", "median", observed=True)
    vin_ratio_old = (vin_piv.loc["11-15", True] / vin_piv.loc["11-15", False])
    cond_raw = d.groupby("cond_missing")["price"].median()
    cond_ratio_raw = cond_raw[True] / cond_raw[False]
    cond_piv = d.pivot_table("price", "age_bucket", "cond_missing", "median", observed=True)
    cond_ratio_old = (cond_piv.loc["11-15", True] / cond_piv.loc["11-15", False])

    # quant caveats
    age_year_corr = df[["age", "year"]].corr().iloc[0, 1]
    n_models = df["model"].nunique()
    rare_share = (df["model"].value_counts() < 5).mean()
    date_span = (df["posting_date"].max() - df["posting_date"].min()).days
    ak = df[df["state"] == "ak"]
    ak_4wd = (ak["drive"] == "4wd").mean()
    all_4wd = (df["drive"] == "4wd").mean()

    lines = [
        "# Phase 2 Insights — EDA\n",
        f"Dataset: {len(df):,} cleaned listings.\n",
        "## Headline findings (each maps to a figure in reports/figures/)\n",
        "### 1. Price is right-skewed -> log target is correct",
        f"- Raw price skew: {df['price'].skew():.2f}; log1p(price) skew: {df['log_price'].skew():.2f}.",
        "- Business meaning: modeling raw price would let a handful of expensive listings dominate the loss. "
        "Training on log price treats a $2k error on a $5k car and a $20k car proportionally.\n",
        "### 2. Depreciation is front-loaded",
        f"- Market median price: age 1 = ${age1:,.0f}, age 5 = ${age5:,.0f}, age 10 = ${age10:,.0f}.",
        f"- A car loses roughly {(1 - age5/age1)*100:.0f}% of value by year 5 and {(1 - age10/age1)*100:.0f}% by year 10.",
        "- Business meaning: the first few years carry the most pricing risk; accurate age handling matters most there.\n",
        "### 3. Mileage drives price, but non-linearly",
        f"- Correlation price~odometer: {corr['odometer']:.2f} (strongest single numeric driver).",
        "- The trend line drops steeply then flattens after ~150k miles.",
        "- Business meaning: below ~150k miles each mile costs real money; above it, mileage barely moves price.\n",
        "### 4. mileage_per_year is a weak linear signal",
        f"- Correlation price~mileage_per_year: {corr['mileage_per_year']:.2f} (near zero).",
        "- Business meaning: raw mileage and age each carry more signal than their ratio; keep the ratio only if a "
        "tree model finds interactions, don't rely on it linearly.\n",
        "### 5. Strong regional price spread",
        f"- Highest-median state: {state_med.index[0]} (${state_med.iloc[0]:,.0f}); "
        f"lowest: {state_med.index[-1]} (${state_med.iloc[-1]:,.0f}).",
        f"- Spread: {state_med.iloc[0] / state_med.iloc[-1]:.2f}x between the priciest and cheapest state medians.",
        "- Business meaning: location is a real pricing feature (arbitrage opportunity for a marketplace), not noise.\n",
        "### 6. Brand tiers are clear",
        f"- Cheapest volume brand median: {med_by_brand.loc[top_brands].idxmin()} "
        f"(${med_by_brand.loc[top_brands].min():,.0f}); "
        f"across all brands, luxury marques sit well above mass-market ones.",
        "- Business meaning: `manufacturer` (and `model`) are high-value categorical features; worth target encoding.\n",
        "## Experimental / behavioral findings\n",
        "### 7. Value heaping (figure 08)",
        f"- {pct_995:.0%} of prices end in 995 and {pct_round_price:.0%} end in 000 -> psychological pricing anchors.",
        f"- {pct_round_odo:.0%} of odometer readings are an exact round thousand -> sellers round their mileage.",
        "- Business meaning: odometer is not a precise continuous measurement; treat round-number heaping as noise, "
        "and expect list prices to cluster at .995/.999 rather than vary smoothly.\n",
        "### 8. Confound warning: several 'premiums' are mostly age (figure 09)",
        f"- Raw VIN premium: {vin_ratio_raw:.2f}x, but within the 11-15yr bucket it collapses to {vin_ratio_old:.2f}x.",
        f"- Raw missing-condition premium: {cond_ratio_raw:.2f}x, but within the 11-15yr bucket it is only "
        f"{cond_ratio_old:.2f}x -- it almost entirely reflects newer cars omitting the field.",
        "- Business meaning: univariate category 'premiums' (VIN, missing condition, and likely 4wd/diesel/color) are "
        "heavily confounded with age and body type. This is the core argument for a multivariate model over "
        "single-feature rules of thumb.\n",
        "### 9. Age x odometer interact (figure 10)",
        "- Median price over the 2D age x odometer grid falls along both axes jointly, not additively; a low-mileage "
        "old car and a high-mileage new car are priced very differently.",
        "- Business meaning: tree models (which capture interactions natively) should beat a purely additive linear "
        "model here; mileage_per_year was a weak proxy for exactly this interaction.\n",
        "### 10. Structured missingness + extreme model cardinality (figure 11)",
        f"- Missing fields co-occur (drive/type/paint_color/cylinders/size, phi up to ~0.5): missingness is NOT random.",
        f"- `model` has {n_models:,} unique values and {rare_share:.0%} appear in fewer than 5 listings.",
        "- Business meaning (Phase 3): impute with a 'missing' indicator rather than silent fill; and do NOT naive "
        "target-encode 20k models -- use smoothing / collapse the rare long tail to avoid overfitting.\n",
        "## Quant caveats (methodological honesty)\n",
        f"- **Perfect collinearity:** corr(age, year) = {age_year_corr:.2f} by construction. Use only ONE of them in "
        "the linear baseline; keeping both breaks coefficient interpretation.",
        f"- **State spread is confounded:** the highest-median state (AK) is {ak_4wd:.0%} 4wd vs {all_4wd:.0%} "
        "market-wide. The 3x 'regional' spread partly reflects a heavier truck/4wd mix, not pure geography -- same "
        "confound lesson as finding 8, applied to location.",
        f"- **Single-month snapshot:** all listings span just {date_span} days (Apr-May 2021). No seasonal/temporal "
        "modeling is possible or claimed; this also justifies a random (non-temporal) train/test split in Phase 3.",
    ]
    return "\n".join(lines)


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

    INSIGHTS.write_text(compute_insights(df), encoding="utf-8")
    print(f"Insights written to {INSIGHTS}")


if __name__ == "__main__":
    main()
