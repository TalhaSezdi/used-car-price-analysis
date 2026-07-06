"""Recon probe: is `description` worth extracting a leakage-free trim signal from?

Context: Phase 6C proved the price model systematically under-predicts rare,
expensive trims (e.g. predicted mid ~$30k vs actual ~$65k on the missed 50-150k
listings) -- a modeling gap, not a calibration gap. `description` is the only
field that might carry trim/package information the current features
(manufacturer, model, year, odometer, condition, ...) do not. Before spending
time building a leakage-free extractor, this probe checks three things:

  1. How much description text actually survives cleaning (missingness)?
  2. Do trim/package keywords in the text correlate with price WITHIN the same
     (manufacturer, model, year) group -- i.e. after controlling for the
     features the model already has? If not, there is no signal to extract.
  3. Do the specific listings the model most under-predicts (proxy: actual
     price >> point-model prediction, in the 50-150k band) actually MENTION a
     trim/package keyword in their description? If they mostly don't, the
     model's error is not a "missing trim signal" problem after all.

Read-only recon: does not modify the cleaning or training pipeline. Rebuilds a
version of the cleaned dataset that also retains `id`/`description` (dropped by
DataCleaner for the real pipeline) by temporarily relaxing DROP_COLS -- the
rest of the cleaning logic is untouched, so row order/content is identical to
data/processed/cleaned.parquet; this is checked explicitly below.

Usage:
    python scripts/probe_description_signal.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

import src.preprocess.cleaner as cleaner_mod
from src.features.engineer import FeatureEngineer
from src.models.dataset import (
    build_split, NUMERIC_FEATURES, LOW_CARD_FEATURES, HIGH_CARD_FEATURES,
)
from src.models.encoders import FeaturePreprocessor
from src.models.intervals import fit_median_model

ROOT = Path(__file__).parents[1]
RAW_PATH = ROOT / "data" / "raw" / "vehicles.csv"
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned.parquet"

# Common trim / package / condition-boosting keywords seen in Craigslist listings.
# Deliberately excludes anything price-like (handled separately / out of scope).
TRIM_KEYWORDS = [
    "denali", "platinum", "laramie", "limited", "sahara", "rubicon", "lariat",
    "sle", "slt", "xlt", "touring", "premium", "loaded", "leather", "sunroof",
    "navigation", "package", "trim", "sport", "se ", "ex ", "lx ", "lt ",
    "lariat", "king ranch", "big horn", "trd", "z71", "overland",
]


def rebuild_cleaned_with_text() -> pd.DataFrame:
    """Reproduce cleaned.parquet's exact rows, but keep id + description."""
    original_drop_cols = list(cleaner_mod.DROP_COLS)
    cleaner_mod.DROP_COLS = [c for c in original_drop_cols if c not in ("id", "description")]
    try:
        raw = pd.read_csv(RAW_PATH, low_memory=False)
        cleaner = cleaner_mod.DataCleaner()
        df = cleaner.fit_transform(raw)
    finally:
        cleaner_mod.DROP_COLS = original_drop_cols
    df = FeatureEngineer().fit_transform(df)
    return df


def main() -> None:
    df_text = rebuild_cleaned_with_text()
    df_ref = pd.read_parquet(CLEANED_PATH)

    # --- Alignment sanity check: same rows, same order, as the real pipeline ---
    assert len(df_text) == len(df_ref), (
        f"Row count mismatch: reconstructed={len(df_text)}, cleaned.parquet={len(df_ref)} "
        "-- DROP_COLS relaxation changed filtering logic, alignment is NOT safe."
    )
    check_cols = ["manufacturer", "model", "year", "price", "odometer"]
    a = df_text[check_cols].reset_index(drop=True)
    b = df_ref[check_cols].reset_index(drop=True)
    # NaN-safe equality: NaN != NaN by default, which would falsely flag rows
    # where a column (e.g. model) is legitimately missing in both.
    row_match = pd.Series(True, index=a.index)
    for col in check_cols:
        same = (a[col] == b[col]) | (a[col].isna() & b[col].isna())
        row_match &= same
    mismatches = int((~row_match).sum())
    print(f"Alignment check: {mismatches} mismatched rows out of {len(df_ref)} "
          f"({'OK' if mismatches == 0 else 'MISALIGNED -- DO NOT TRUST BELOW'})")
    if mismatches > 0:
        raise SystemExit(
            f"Aborting: {mismatches} rows do not align between the reconstructed "
            "description-bearing dataset and cleaned.parquet. Results below this "
            "point would be silently wrong (description text joined to the wrong "
            "row). Fix the reconstruction before trusting any downstream numbers."
        )

    # --- 1. Missingness ---
    desc = df_text["description"]
    n = len(desc)
    n_missing = int(desc.isna().sum())
    n_empty = int((desc.fillna("").str.strip() == "").sum())
    print(f"\n=== Description availability (n={n:,}) ===")
    print(f"Missing (NaN)      : {n_missing:,} ({n_missing/n*100:.2f}%)")
    print(f"Empty after strip  : {n_empty:,} ({n_empty/n*100:.2f}%)")
    print(f"Usable text        : {n - n_empty:,} ({(n - n_empty)/n*100:.2f}%)")

    lengths = desc.fillna("").str.len()
    print(f"Length (chars): median={lengths.median():.0f}, "
          f"p10={lengths.quantile(.1):.0f}, p90={lengths.quantile(.9):.0f}")

    # --- 2. Keyword presence vs price, WITHIN (manufacturer, model, year) group ---
    text_lower = desc.fillna("").str.lower()
    pattern = re.compile("|".join(re.escape(k) for k in TRIM_KEYWORDS))
    has_kw = text_lower.str.contains(pattern)
    print(f"\n=== Trim/package keyword presence ===")
    print(f"Any keyword found: {has_kw.mean()*100:.1f}% of listings")

    d = df_text[["manufacturer", "model", "year", "price"]].copy()
    d["has_kw"] = has_kw.values
    grp = d.groupby(["manufacturer", "model", "year"], observed=True)
    # Only groups with >=20 rows and a real split between kw/no-kw rows
    sizes = grp.size()
    valid_groups = sizes[sizes >= 20].index
    d_valid = d.set_index(["manufacturer", "model", "year"]).loc[valid_groups].reset_index()

    within_group_premium = (
        d_valid.groupby(["manufacturer", "model", "year", "has_kw"], observed=True)["price"]
        .median()
        .unstack("has_kw")
        .dropna()
    )
    if not within_group_premium.empty and True in within_group_premium.columns and False in within_group_premium.columns:
        premium_pct = ((within_group_premium[True] - within_group_premium[False])
                       / within_group_premium[False] * 100)
        print(f"Groups with both kw/no-kw rows (n>=20 total, same make/model/year): "
              f"{len(premium_pct)}")
        print(f"Median within-group price premium when a trim keyword is present: "
              f"{premium_pct.median():.1f}%")
        print(f"(25th/75th pct: {premium_pct.quantile(.25):.1f}% / "
              f"{premium_pct.quantile(.75):.1f}%)")
    else:
        print("Not enough same-make/model/year groups with both kw/no-kw rows to compare.")

    # --- 3. Do the model's most under-predicted expensive listings mention a trim kw? ---
    split = build_split(df_ref)
    prep = FeaturePreprocessor(
        numeric_cols=NUMERIC_FEATURES, low_card_cols=LOW_CARD_FEATURES,
        high_card_cols=HIGH_CARD_FEATURES, high_card_method="target",
    )
    Xt_train = prep.fit_transform(split.X_train_full, split.y_train_full)
    Xt_test = prep.transform(split.X_test)
    med_model = fit_median_model(Xt_train, split.y_train_full)
    pred_dollar = np.expm1(med_model.predict(Xt_test))
    price_test = split.price_test.values

    test_text = df_text.loc[split.X_test.index, "description"].fillna("").str.lower()
    test_has_kw = test_text.str.contains(pattern).values

    expensive = (price_test >= 50_000) & (price_test <= 150_000)
    underpriced = expensive & (pred_dollar < 0.6 * price_test)  # model badly under-predicts
    print(f"\n=== Underpredicted expensive listings (n={underpriced.sum()}) "
          f"vs all expensive (n={expensive.sum()}) ===")
    print(f"Keyword present, underpredicted subset : "
          f"{test_has_kw[underpriced].mean()*100:.1f}%")
    print(f"Keyword present, all expensive subset  : "
          f"{test_has_kw[expensive].mean()*100:.1f}%")
    print(f"Keyword present, overall test set       : {test_has_kw.mean()*100:.1f}%")

    print("\n--- Sample descriptions from underpredicted expensive listings ---")
    sample_idx = np.where(underpriced)[0][:5]
    ctx_cols = ["manufacturer", "model", "year", "price"]
    ctx = df_ref.loc[split.X_test.index].iloc[sample_idx][ctx_cols]
    texts = test_text.iloc[sample_idx]
    preds = pred_dollar[sample_idx]
    for (_, row), text, pred in zip(ctx.iterrows(), texts, preds):
        car = f"{int(row['year'])} {row['manufacturer']} {row['model']}"
        print(f"\n[{car}] actual=${row['price']:,.0f} predicted=${pred:,.0f}")
        print(f"  description: {text[:300]!r}")


if __name__ == "__main__":
    main()
