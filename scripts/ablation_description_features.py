"""Phase 7B, Ablation A4: does adding desc_* features improve the price model?

Question: Phase 6C found a ~16.5% within-(manufacturer, model, year) price
premium for listings mentioning a trim/equipment keyword -- signal the current
feature set cannot see. src.features.description.DescriptionFeatureExtractor
adds three leakage-free numeric features (desc_trim_luxury, desc_equip_count,
desc_len_log) to the cleaned dataset. This script trains the final LightGBM
pipeline WITH vs WITHOUT those features, same split, same everything else, and
reports whether it actually helps -- a negative result is recorded exactly like
a positive one (see Phase 6C's own gain-importance check).

Does NOT modify src/models/dataset.py's default NUMERIC_FEATURES. If this
ablation shows a real improvement, the follow-up step (separate, after user
approval) is to adopt the features as the default and re-ripple Phase 3/6
metrics. If not, the columns stay in the parquet, unused by default, and the
negative result is documented.

Usage:
    python scripts/ablation_description_features.py

Produces:
    docs/phase7_results.md (7B section)
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.dataset import (
    build_split, NUMERIC_FEATURES, LOW_CARD_FEATURES, HIGH_CARD_FEATURES,
)
from src.models.encoders import FeaturePreprocessor
from src.models.train import train_lgbm
from src.evaluation.metrics import metrics_table, error_by_segment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "processed" / "cleaned.parquet"
RESULTS = ROOT / "docs" / "phase7_results.md"

DESC_FEATURES = ["desc_trim_luxury", "desc_equip_count", "desc_len_log"]


def _build_matrix(df: pd.DataFrame, X_idx: pd.Index, numeric_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in numeric_cols + LOW_CARD_FEATURES + HIGH_CARD_FEATURES if c in df.columns]
    return df.loc[X_idx, cols].copy()


def main() -> None:
    df = pd.read_parquet(DATA)
    logger.info("Loaded %d rows", len(df))
    missing = [c for c in DESC_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Expected desc_* columns missing from cleaned.parquet: {missing}. "
            "Rerun scripts/clean_data.py after the Phase 7B FeatureEngineer change."
        )

    split = build_split(df)  # reuses Phase 3/6A split (X/y are rebuilt below with more cols)

    # NUMERIC_FEATURES now contains DESC_FEATURES by default (Phase 7B adoption).
    # For this ablation the "baseline" must strip them out; "extended" is just
    # the current default set (no double-add).
    baseline_numeric = [c for c in NUMERIC_FEATURES if c not in DESC_FEATURES]
    extended_numeric = list(NUMERIC_FEATURES)

    # Use train+val (80%) as the training corpus so this ablation reflects
    # the same 80/20 setup the final headline model is trained on.
    X_train_base = _build_matrix(df, split.X_train_full.index, baseline_numeric)
    X_test_base = _build_matrix(df, split.X_test.index, baseline_numeric)
    X_train_ext = _build_matrix(df, split.X_train_full.index, extended_numeric)
    X_test_ext = _build_matrix(df, split.X_test.index, extended_numeric)

    y_train, y_test = split.y_train_full, split.y_test
    price_test = split.price_test.values

    results = {}
    gain_desc = None

    for label, num_cols, X_tr, X_te in [
        ("baseline (no desc_*)", baseline_numeric, X_train_base, X_test_base),
        ("with desc_* features", extended_numeric, X_train_ext, X_test_ext),
    ]:
        prep = FeaturePreprocessor(
            numeric_cols=num_cols,
            low_card_cols=LOW_CARD_FEATURES,
            high_card_cols=HIGH_CARD_FEATURES,
            high_card_method="target",
        )
        Xt = prep.fit_transform(X_tr, y_train)
        Xv = prep.transform(X_te)
        model = train_lgbm(Xt, y_train.values, Xv, y_test.values, price_test)
        results[label] = model.metrics
        logger.info("%s: %s", label, model.metrics)

        if label == "with desc_* features":
            gain = model.model.booster_.feature_importance(importance_type="gain")
            names = model.model.booster_.feature_name()
            imp = pd.Series(gain, index=names)
            imp = imp / imp.sum() * 100
            gain_desc = imp[imp.index.isin(DESC_FEATURES)].sort_values(ascending=False)
            preds_ext = model.predictions

    comparison = metrics_table(results)
    print("\n=== Ablation A4: baseline vs with desc_* features ===")
    print(comparison.to_string())
    print("\n=== Gain share of desc_* features (with-desc model) ===")
    print(gain_desc.round(3).to_string())

    # --- Error by price segment, with-desc model (targets the 6C tail) ---
    test_df = df.loc[split.X_test.index].copy()
    price_bins = pd.cut(test_df["price"],
                        [0, 5000, 10000, 20000, 50000, 150000],
                        labels=["<5k", "5-10k", "10-20k", "20-50k", "50-150k"])
    err_price = error_by_segment(y_test.values, preds_ext, price_bins, price_test)
    print("\n=== Error by price segment (with-desc model) ===")
    print(err_price.to_string())

    rmse_base = results["baseline (no desc_*)"]["RMSE ($)"]
    rmse_ext = results["with desc_* features"]["RMSE ($)"]
    mape_base = results["baseline (no desc_*)"]["MAPE (%)"]
    mape_ext = results["with desc_* features"]["MAPE (%)"]
    pct_rmse = (rmse_ext - rmse_base) / rmse_base * 100

    write_results(comparison, gain_desc, err_price, rmse_base, rmse_ext, mape_base, mape_ext, pct_rmse)
    logger.info("Results written to %s", RESULTS)


def write_results(comparison, gain_desc, err_price, rmse_base, rmse_ext, mape_base, mape_ext, pct_rmse) -> None:
    total_gain = gain_desc.sum()
    if pct_rmse < -0.5 and mape_ext < mape_base:
        verdict = (
            f"**Verdict: real improvement.** RMSE drops {abs(pct_rmse):.2f}% "
            f"(${rmse_base:,.0f} -> ${rmse_ext:,.0f}), MAPE improves "
            f"({mape_base:.2f}% -> {mape_ext:.2f}%). desc_* features carry "
            f"{total_gain:.2f}% of total gain -- a real, if modest, signal. "
            "Recommendation: adopt as default features (separate follow-up step, "
            "requires re-rippling Phase 3/6 metrics)."
        )
    elif total_gain < 0.5:
        verdict = (
            f"**Verdict: negative result.** desc_* features carry only "
            f"{total_gain:.2f}% of total gain and RMSE moved {pct_rmse:+.2f}% "
            "(noise-level). The trim/equipment keyword signal, once boiled down "
            "to 3 leakage-free numeric features, does not add information beyond "
            "what manufacturer/model/year/odometer/target-encoding already "
            "captures. Recorded as a negative result -- exactly like Phase 6C's "
            "gain-importance check. The columns stay in cleaned.parquet but are "
            "NOT added to the default feature list."
        )
    else:
        verdict = (
            f"**Verdict: mixed / inconclusive.** RMSE moved {pct_rmse:+.2f}% and "
            f"desc_* features carry {total_gain:.2f}% of gain -- some signal, but "
            "not a clear win on the headline metrics. Not adopted as default; "
            "documented as a partial/negative result."
        )

    lines = [
        "## 7B. Description trim/equipment features\n",
        "**Method:** `src/features/description.py::DescriptionFeatureExtractor` adds "
        "`desc_trim_luxury` (0/1, curated trim-name keyword match), "
        "`desc_equip_count` (count of equipment keywords), `desc_len_log` "
        "(log1p description length). Stateless, row-wise, alphabetic-keyword-only "
        "(no digit extraction) -- computed in `FeatureEngineer` before the raw "
        "`description` column is dropped; the processed parquet never contains "
        "raw text. Ablation A4 (`scripts/ablation_description_features.py`) trains "
        "the final LightGBM pipeline with vs without these 3 columns, same "
        "train/test split (Phase 3/6A), same encoding pipeline.\n",
        "### Ablation A4: baseline vs with desc_* features\n",
        comparison.to_markdown(),
        "\n### Gain share of desc_* features (with-desc model)\n",
        gain_desc.round(3).to_markdown() if len(gain_desc) else "_(no gain from any desc_* feature)_",
        "\n### Error by price segment (with-desc model)\n",
        err_price.to_markdown(),
        "\n" + verdict + "\n",
    ]
    existing = RESULTS.read_text(encoding="utf-8")
    header = "## 7B. Description trim/equipment features"
    if header in existing:
        existing = existing.split(header)[0]
    existing = existing.rstrip("\n") + "\n\n"
    RESULTS.write_text(existing + "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
