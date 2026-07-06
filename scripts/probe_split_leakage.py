"""Phase 6A probe: are near-duplicate listings straddling the train/test split?

Reuses the exact Phase 3 split (src.models.dataset.build_split, same seed) and the
exact Phase 3 final model pipeline (target-encoded LightGBM). Does NOT modify the
pipeline -- this is a read-only diagnostic per CLAUDE.md's "prove it exists before
fixing it" rule.

A test row is flagged "contaminated" if a TRAIN row exists with:
  - identical (manufacturer, model, year), AND
  - |odometer_test - odometer_train| <= ODO_BAND miles, AND
  - relative price gap |p_test - p_train| / max(p_test, p_train) <= PRICE_BAND

Matching within each (manufacturer, model, year) group is done via sorted-odometer
+ searchsorted (O(n log n) per group), not a full O(n^2) cross join.

Usage:
    python scripts/probe_split_leakage.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.dataset import (
    build_split, select_features, NUMERIC_FEATURES, LOW_CARD_FEATURES,
    HIGH_CARD_FEATURES,
)
from src.models.encoders import FeaturePreprocessor
from src.models.train import train_lgbm
from src.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "processed" / "cleaned.parquet"

# (odometer_band_miles, price_band_fraction) sensitivity sweep
BANDS = [(250, 0.02), (500, 0.05), (1000, 0.10)]


def find_contaminated_test_rows(
    df: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    odo_band: float,
    price_band: float,
) -> np.ndarray:
    """Return a boolean mask over test_idx: True where a near-dup exists in train."""
    key_cols = ["manufacturer", "model", "year"]
    train_df = df.loc[train_idx, key_cols + ["odometer", "price"]].copy()
    test_df = df.loc[test_idx, key_cols + ["odometer", "price"]].copy()

    train_df["_key"] = list(zip(*[train_df[c] for c in key_cols]))
    test_df["_key"] = list(zip(*[test_df[c] for c in key_cols]))

    train_groups = train_df.groupby("_key")
    flagged = np.zeros(len(test_df), dtype=bool)

    test_by_key: dict = {}
    for pos, key in enumerate(test_df["_key"].values):
        test_by_key.setdefault(key, []).append(pos)

    for key, test_positions in test_by_key.items():
        if key not in train_groups.groups:
            continue
        tr_sub = train_df.loc[train_groups.groups[key]]
        if tr_sub.empty:
            continue
        tr_odo = tr_sub["odometer"].to_numpy()
        tr_price = tr_sub["price"].to_numpy()
        order = np.argsort(tr_odo)
        tr_odo_sorted = tr_odo[order]
        tr_price_sorted = tr_price[order]

        for pos in test_positions:
            te_odo = test_df["odometer"].iat[pos]
            te_price = test_df["price"].iat[pos]
            if pd.isna(te_odo):
                lo_i, hi_i = 0, len(tr_odo_sorted)
            else:
                lo_i = np.searchsorted(tr_odo_sorted, te_odo - odo_band, side="left")
                hi_i = np.searchsorted(tr_odo_sorted, te_odo + odo_band, side="right")
            if lo_i >= hi_i:
                continue
            cand_price = tr_price_sorted[lo_i:hi_i]
            gap = np.abs(te_price - cand_price) / np.maximum(
                np.maximum(te_price, cand_price), 1
            )
            if np.any(gap <= price_band):
                flagged[pos] = True

    return flagged


def main() -> None:
    df = pd.read_parquet(DATA)
    logger.info("Loaded %d rows", len(df))

    split = build_split(df)
    # Use train+val as the training corpus so this probe reflects the same
    # 80/20 setup the final headline model is trained on.
    train_idx = split.X_train_full.index
    test_idx = split.X_test.index

    # --- Sensitivity sweep: contamination rate at 3 band widths ---
    print("\n=== Contamination rate sensitivity ===")
    sweep_rows = []
    default_mask = None
    for odo_band, price_band in BANDS:
        mask = find_contaminated_test_rows(df, train_idx, test_idx, odo_band, price_band)
        share = mask.mean() * 100
        sweep_rows.append({
            "odo_band_mi": odo_band,
            "price_band_pct": price_band * 100,
            "contaminated_test_rows": int(mask.sum()),
            "share_pct": round(share, 3),
        })
        if (odo_band, price_band) == (500, 0.05):
            default_mask = mask
    sweep_df = pd.DataFrame(sweep_rows)
    print(sweep_df.to_string(index=False))

    # --- Train the Phase 3 final pipeline (target-encoded LightGBM) once ---
    logger.info("Training Phase 3 final LightGBM pipeline for metric comparison...")
    prep = FeaturePreprocessor(
        numeric_cols=NUMERIC_FEATURES,
        low_card_cols=LOW_CARD_FEATURES,
        high_card_cols=HIGH_CARD_FEATURES,
        high_card_method="target",
    )
    Xt = prep.fit_transform(split.X_train_full, split.y_train_full)
    Xv = prep.transform(split.X_test)
    lgbm = train_lgbm(Xt, split.y_train_full.values, Xv, split.y_test.values, split.price_test.values)

    y_test_log = split.y_test.values
    preds_log = lgbm.predictions
    price_test = split.price_test.values

    # --- Metric comparison: full / contaminated / clean (default 500mi / 5% band) ---
    print("\n=== Metrics: full test vs contaminated subset vs clean subset (500mi / 5%) ===")
    full_m = compute_metrics(y_test_log, preds_log, price_test)
    contam_m = (
        compute_metrics(y_test_log[default_mask], preds_log[default_mask], price_test[default_mask])
        if default_mask.sum() > 0 else {}
    )
    clean_m = compute_metrics(
        y_test_log[~default_mask], preds_log[~default_mask], price_test[~default_mask]
    )
    report = pd.DataFrame({
        "full_test": full_m,
        "contaminated_subset": contam_m,
        "clean_subset": clean_m,
    }).T
    report.insert(0, "n_rows", [len(y_test_log), int(default_mask.sum()), int((~default_mask).sum())])
    print(report.round(2).to_string())

    rmse_full = full_m["RMSE ($)"]
    rmse_clean = clean_m["RMSE ($)"]
    pct_diff = (rmse_clean - rmse_full) / rmse_full * 100
    print(f"\nClean-subset RMSE vs full-test RMSE: {pct_diff:+.2f}%")
    print(f"Contaminated share (default band): {default_mask.mean() * 100:.3f}%")

    print("\n=== Decision gate ===")
    gate_pass = (default_mask.mean() < 0.01) and (abs(pct_diff) < 2.0)
    if gate_pass:
        print("PASS: contamination < 1% and clean-subset RMSE within 2% of full test.")
        print("-> No group-aware re-split needed. Document as a defense note.")
    else:
        print("FAIL: gate criteria not met.")
        print("-> Proceed to Step A2: group-aware re-split and metric regeneration.")


if __name__ == "__main__":
    main()
