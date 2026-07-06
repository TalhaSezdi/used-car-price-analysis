"""Tests for src/evaluation/metrics.py."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_metrics,
    coverage,
    coverage_by_segment,
    error_by_segment,
    gain_importance_table,
    metrics_table,
)


def test_compute_metrics_from_log_scale_matches_manual_expm1():
    y_true_log = np.log1p(np.array([1000.0, 2000.0, 3000.0]))
    y_pred_log = np.log1p(np.array([1100.0, 1900.0, 3100.0]))
    m = compute_metrics(y_true_log, y_pred_log)

    y_true_dollar = np.expm1(y_true_log)
    y_pred_dollar = np.expm1(y_pred_log)
    expected_mae = np.mean(np.abs(y_true_dollar - y_pred_dollar))
    assert abs(m["MAE ($)"] - expected_mae) < 1e-6


def test_compute_metrics_price_actual_override_used_over_expm1():
    y_true_log = np.log1p(np.array([1000.0, 2000.0]))
    y_pred_log = np.log1p(np.array([1000.0, 2000.0]))
    price_actual = np.array([999.0, 2001.0])  # deliberately different from expm1(y_true_log)
    m = compute_metrics(y_true_log, y_pred_log, price_actual=price_actual)
    assert m["MAE ($)"] > 0  # would be ~0 if price_actual were ignored


def test_compute_metrics_mape_excludes_zero_price_rows():
    y_true_log = np.log1p(np.array([0.0, 1000.0]))
    y_pred_log = np.log1p(np.array([0.0, 1100.0]))
    m = compute_metrics(y_true_log, y_pred_log)
    assert not np.isnan(m["MAPE (%)"])


def test_metrics_table_rounds_to_2_decimals():
    results = {"model_a": {"RMSE ($)": 123.456789, "MAE ($)": 45.6789}}
    table = metrics_table(results)
    assert table.index.name == "Model"
    assert table.loc["model_a", "RMSE ($)"] == 123.46


def test_error_by_segment_sorted_by_mae_descending():
    y_true_log = np.log1p(np.array([1000.0, 1000.0, 5000.0, 5000.0]))
    y_pred_log = np.log1p(np.array([1200.0, 1200.0, 5100.0, 5100.0]))
    segment = pd.Series(["a", "a", "b", "b"])
    agg = error_by_segment(y_true_log, y_pred_log, segment)
    assert list(agg.index) == ["a", "b"]  # segment "a" has the larger MAE


def test_error_by_segment_pct_error_nan_for_zero_price():
    y_true_log = np.log1p(np.array([0.0]))
    y_pred_log = np.log1p(np.array([100.0]))
    segment = pd.Series(["a"])
    agg = error_by_segment(y_true_log, y_pred_log, segment)
    assert np.isnan(agg.loc["a", "MAPE"])


def test_coverage_matches_manual_count():
    price_actual = np.array([100.0, 200.0, 300.0, 400.0])
    lo = np.array([90.0, 90.0, 310.0, 350.0])
    hi = np.array([110.0, 150.0, 320.0, 450.0])
    # covered: row0 (yes), row1 (no, 200>150), row2 (no, 300<310), row3 (yes)
    assert coverage(price_actual, lo, hi) == 0.5


def test_coverage_by_segment_groups_correctly():
    price_actual = np.array([100.0, 100.0, 500.0, 500.0])
    lo = np.array([90.0, 90.0, 600.0, 600.0])
    hi = np.array([110.0, 110.0, 700.0, 700.0])
    segment = pd.Series(["a", "a", "b", "b"])
    result = coverage_by_segment(price_actual, lo, hi, segment)
    assert result.loc["a", "coverage"] == 1.0
    assert result.loc["b", "coverage"] == 0.0
    assert result.loc["a", "count"] == 2


def test_gain_importance_table_sums_to_100_and_sorted_descending():
    import lightgbm as lgb

    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 200), "b": rng.uniform(0, 1, 200)})
    y = X["a"] * 10 + rng.normal(0, 0.01, 200)  # "a" should dominate gain
    model = lgb.LGBMRegressor(n_estimators=20, verbose=-1).fit(X, y)

    imp = gain_importance_table(model, top_n=None)
    assert abs(imp.sum() - 100.0) < 1e-6
    assert imp.index[0] == "a"
    assert list(imp.values) == sorted(imp.values, reverse=True)


def test_gain_importance_table_respects_top_n():
    import lightgbm as lgb

    rng = np.random.RandomState(42)
    X = pd.DataFrame({f"f{i}": rng.uniform(0, 1, 100) for i in range(5)})
    y = pd.Series(rng.uniform(0, 1, 100))
    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1).fit(X, y)

    imp = gain_importance_table(model, top_n=2)
    assert len(imp) == 2
