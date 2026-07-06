"""Shared constants for the used-car pipeline.

Single source of truth for values that were previously redefined
independently in multiple modules (``RANDOM_STATE`` appeared in 5 files) or
retyped as inline literals in multiple scripts (price/age segment bins).
Every value here is copied verbatim from what was already the de facto
default across the codebase -- this module only relocates and names them,
it does not change any default.
"""

from __future__ import annotations

RANDOM_STATE: int = 42
"""Seed used by every stochastic operation (splits, CV folds, model fits)."""

PRICE_SEGMENT_BINS: list[float] = [0, 5_000, 10_000, 20_000, 50_000, 150_000]
"""Bin edges for price-segment error analysis (train.py, predict_intervals.py,
ablation_description_features.py)."""

PRICE_SEGMENT_LABELS: list[str] = ["<5k", "5-10k", "10-20k", "20-50k", "50-150k"]
"""Labels matching ``PRICE_SEGMENT_BINS`` (5 labels for 5 bins)."""

AGE_BUCKET_BINS: list[float] = [0, 3, 6, 10, 15, 60]
"""Bin edges for age-bucket analysis (plots.py, train.py, predict_intervals.py)."""

AGE_BUCKET_LABELS: list[str] = ["0-3", "4-6", "7-10", "11-15", "16+"]
"""Base labels matching ``AGE_BUCKET_BINS``. Some call sites (train.py,
predict_intervals.py) display these with a "yr" suffix -- derive that at the
call site (``[l + "yr" for l in AGE_BUCKET_LABELS]``) rather than hardcoding a
second label list here."""

ANOMALY_Z_THRESHOLD: float = 3.5
"""Robust z-score flag threshold for ResidualAnomalyDetector, also used as the
default reference line in plot_anomaly_overview."""

INTERVAL_ALPHA: float = 0.10
"""1 - alpha is the nominal conformal prediction interval coverage (90%)."""

LGBM_QUANTILE_N_ESTIMATORS: int = 3000
"""Default n_estimators for the quantile LightGBM models in models/intervals.py."""
