"""Phase 6C root-cause probe: why Mondrian does not fix ACTUAL-price-segment coverage.

Context: Mondrian CQR (per-bin conformal corrections on the predicted-band
midpoint) was expected to close the 6B coverage gap on the price tails. It did
not (50-150k actual segment: 70.2% -> 69.6%). This probe proves why, testing:

H1: Mondrian DOES hold ~90% coverage per PREDICTED-price bin -- its actual,
    feature-conditional guarantee. (Confirmed: 88.7-90.8% per bin.)
H2: The missed 50-150k actual-price rows are rows the point model badly
    UNDER-predicts (predicted mid ~$30k vs actual ~$65k, medians) -- they sit
    inside bins dominated by genuinely mid-priced cars, so no per-bin widening
    reaches them without destroying interval usefulness for everyone else.
H3: Finer (10-bin) or tail-focused binning does not help ACTUAL-price segments
    either -- the problem is bin misalignment caused by model bias, not bin
    coarseness. (Confirmed: 50-150k stays ~69% under every binning.)

Conclusion recorded in docs/phase6_results.md: slicing coverage by the ACTUAL
price conditions on the outcome; exact outcome-conditional coverage is provably
unattainable in finite samples (Vovk 2012; Foygel Barber et al. 2021). The
achievable and delivered guarantee is feature-conditional (per predicted bin,
per age bucket). The residual tail miscoverage is a Phase 3 point-model bias on
rare expensive trims -- a modeling problem, not a calibration problem.

Usage:
    python scripts/probe_mondrian_conditional_coverage.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.dataset import (
    build_split, split_calibration,
    NUMERIC_FEATURES, LOW_CARD_FEATURES, HIGH_CARD_FEATURES,
)
from src.models.encoders import FeaturePreprocessor
from src.models.intervals import _fit_lgbm_quantile

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA = Path(__file__).parents[1] / "data" / "processed" / "cleaned.parquet"
ALPHA = 0.10


def corrections_for_bins(mid_cal, scores, edges, alpha):
    bins_cal = np.clip(np.digitize(mid_cal, edges), 0, len(edges))
    corr = np.empty(len(edges) + 1)
    for b in range(len(edges) + 1):
        m = bins_cal == b
        n_b = int(m.sum())
        q = min(1.0, (1 - alpha) * (1 + 1 / max(n_b, 1)))
        corr[b] = np.quantile(scores[m], q) if n_b > 0 else 0.0
    return corr


def main():
    df = pd.read_parquet(DATA)
    split = build_split(df)
    X_fit, X_calib, y_fit, y_calib = split_calibration(split.X_train_full, split.y_train_full)

    prep = FeaturePreprocessor(
        numeric_cols=NUMERIC_FEATURES, low_card_cols=LOW_CARD_FEATURES,
        high_card_cols=HIGH_CARD_FEATURES, high_card_method="target",
    )
    Xt_fit = prep.fit_transform(X_fit, y_fit)
    Xt_calib = prep.transform(X_calib)
    Xt_test = prep.transform(split.X_test)

    lo_m = _fit_lgbm_quantile(Xt_fit, y_fit, ALPHA / 2)
    hi_m = _fit_lgbm_quantile(Xt_fit, y_fit, 1 - ALPHA / 2)

    lo_cal, hi_cal = lo_m.predict(Xt_calib), hi_m.predict(Xt_calib)
    lo_te, hi_te = lo_m.predict(Xt_test), hi_m.predict(Xt_test)
    y_cal = np.asarray(y_calib)
    price_te = split.price_test.values

    mid_cal = (lo_cal + hi_cal) / 2
    mid_te = (lo_te + hi_te) / 2
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)

    price_bins = pd.cut(
        pd.Series(price_te),
        [0, 5000, 10000, 20000, 50000, 150000],
        labels=["<5k", "5-10k", "10-20k", "20-50k", "50-150k"],
    )

    def eval_binning(name, edges):
        corr = corrections_for_bins(mid_cal, scores, edges, ALPHA)
        b_te = np.clip(np.digitize(mid_te, edges), 0, len(edges))
        c = corr[b_te]
        lo_d, hi_d = np.expm1(lo_te - c), np.expm1(hi_te + c)
        inside = (price_te >= lo_d) & (price_te <= hi_d)
        print(f"\n--- {name}: overall={inside.mean():.4f} ---")
        print("coverage by ACTUAL price segment:")
        print(pd.Series(inside).groupby(price_bins, observed=False).mean().round(4).to_string())
        print("coverage by PREDICTED bin (the actual Mondrian guarantee):")
        print(pd.Series(inside).groupby(pd.Series(b_te), observed=False).mean().round(4).to_string())
        return inside, b_te

    edges5 = np.quantile(mid_cal, np.linspace(0, 1, 6)[1:-1])
    inside5, bte5 = eval_binning("5 equal-freq bins (6C setup)", edges5)

    edges10 = np.quantile(mid_cal, np.linspace(0, 1, 11)[1:-1])
    eval_binning("10 equal-freq bins", edges10)

    edges_tail = np.quantile(mid_cal, [0.05, 0.20, 0.80, 0.95])
    eval_binning("tail-focused bins (5/20/80/95 pct)", edges_tail)

    expensive = (price_bins == "50-150k").values
    missed_exp = expensive & ~inside5
    print(f"\n--- H2: missed 50-150k rows (n={missed_exp.sum()}) by predicted bin (5-bin) ---")
    print(pd.Series(bte5[missed_exp]).value_counts().sort_index().to_string())
    print(f"\nAll 50-150k rows (n={expensive.sum()}) by predicted bin:")
    print(pd.Series(bte5[expensive]).value_counts().sort_index().to_string())
    pred_mid_dollar = np.expm1(mid_te[missed_exp])
    print(f"\nMissed 50-150k rows: predicted band midpoint ($) quartiles: "
          f"{np.percentile(pred_mid_dollar, [25, 50, 75]).round(0)}")
    print(f"Their actual prices ($) quartiles: "
          f"{np.percentile(price_te[missed_exp], [25, 50, 75]).round(0)}")


if __name__ == "__main__":
    main()
