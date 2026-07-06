"""Tests for src/anomaly/detector.py."""

import numpy as np
import pandas as pd
import pytest

from src.anomaly.detector import (
    IsolationForestDetector,
    ResidualAnomalyDetector,
    fat_tail_comparison,
    in_sample_residual_std,
    is_strong_residual,
    tier_anomalies,
)


def test_residual_detector_flags_known_outlier():
    rng = np.random.RandomState(42)
    residual = rng.normal(0, 0.05, 200)
    residual[0] = 5.0  # obvious outlier
    y_log = residual
    pred_log = np.zeros(200)

    det = ResidualAnomalyDetector(z_threshold=3.5)
    det.fit(y_log, pred_log)
    scored = det.score(y_log, pred_log)
    assert scored["residual_flag"].iloc[0]
    assert not scored["residual_flag"].iloc[1:].all()


def test_residual_detector_direction_sign():
    y_log = np.array([-1.0, 1.0])
    pred_log = np.array([0.0, 0.0])
    det = ResidualAnomalyDetector().fit(y_log, pred_log)
    scored = det.score(y_log, pred_log)
    assert scored["direction"].iloc[0] == "underpriced"
    assert scored["direction"].iloc[1] == "overpriced"


def test_residual_detector_mad_zero_fallback_to_std():
    residual = np.array([1.0, 1.0, 1.0, 1.0, 5.0])  # MAD is 0 except for the last
    det = ResidualAnomalyDetector()
    det.fit(residual, np.zeros(5))
    assert det.mad_ > 0
    scored = det.score(residual, np.zeros(5))
    assert np.isfinite(scored["residual_z"]).all()


def test_residual_detector_fit_then_score_reproducible():
    rng = np.random.RandomState(1)
    y_log = rng.normal(0, 1, 100)
    pred_log = rng.normal(0, 1, 100)
    det = ResidualAnomalyDetector()
    det.fit(y_log, pred_log)
    s1 = det.score(y_log, pred_log)
    s2 = det.score(y_log, pred_log)
    pd.testing.assert_frame_equal(s1, s2)


def test_isolation_forest_detector_flags_structural_outlier():
    rng = np.random.RandomState(42)
    n = 300
    X = pd.DataFrame({
        "age": rng.uniform(1, 15, n),
        "odometer": rng.uniform(10_000, 150_000, n),
    })
    # Impossible combination: very old car with near-zero odometer.
    X.loc[0, "age"] = 20
    X.loc[0, "odometer"] = 100

    det = IsolationForestDetector(contamination=0.05)
    out = det.fit_score(X)
    assert out["if_score"].iloc[0] > out["if_score"].median()


def test_isolation_forest_detector_handles_nan_via_median_impute():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"age": rng.uniform(1, 15, 50), "odometer": rng.uniform(1_000, 100_000, 50)})
    X.loc[0, "age"] = np.nan
    det = IsolationForestDetector()
    out = det.fit_score(X)  # must not raise
    assert not out["if_score"].isna().any()


def test_in_sample_residual_std_smaller_than_noise_when_overfittable():
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({"age": rng.uniform(1, 20, n)})
    y = pd.Series(10.0 - 0.1 * X["age"] + rng.normal(0, 1.0, n))
    std = in_sample_residual_std(X, y, numeric_cols=["age"], low_card_cols=[], high_card_cols=[], n_estimators=50)
    assert std >= 0
    assert std < 1.0  # in-sample fit should reduce residual std below raw noise std


def test_is_strong_residual_both_conditions_required():
    abs_z = np.array([6.0, 6.0, 1.0])
    abs_pct = np.array([90.0, 10.0, 90.0])
    result = is_strong_residual(abs_z, abs_pct, z_strong=5.0, pct_strong=85.0)
    assert list(result) == [True, False, False]


def test_tier_anomalies_categories():
    residual_z = pd.Series([6.0, 4.0, 1.0, 1.0])
    residual_flag = pd.Series([True, True, False, False])
    residual_pct = pd.Series([90.0, 90.0, 5.0, 5.0])
    if_flag = pd.Series([True, True, True, False])

    out = tier_anomalies(residual_z, residual_flag, residual_pct, if_flag)
    assert out["priority"].iloc[0] == "HIGH (strong mispriced + structural)"
    assert out["priority"].iloc[1] == "moderate mispriced + structural"
    assert out["priority"].iloc[2] == "structural only"
    assert out["priority"].iloc[3] == "normal"


def test_fat_tail_comparison_counts_and_gaussian_expectation():
    abs_z = pd.Series([1.0] * 90 + [6.0] * 10)
    result = fat_tail_comparison(abs_z, thresholds=[5.0])
    threshold, cnt, pct, gauss, gauss_pct = result[0]
    assert cnt == 10
    assert gauss < cnt  # Gaussian expects far fewer than the fat-tailed synthetic data has
