"""Evaluation metrics for price prediction, reported in dollar scale."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    price_actual: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE, R2 in dollar scale after expm1 inverse.

    Parameters
    ----------
    y_true_log : ground truth in log1p(price) scale
    y_pred_log : predictions in log1p(price) scale
    price_actual : optional raw dollar prices (if not provided, expm1 is used)
    """
    y_true_dollar = np.expm1(y_true_log)
    y_pred_dollar = np.expm1(y_pred_log)

    if price_actual is not None:
        y_true_dollar = np.asarray(price_actual)

    rmse = np.sqrt(mean_squared_error(y_true_dollar, y_pred_dollar))
    mae = mean_absolute_error(y_true_dollar, y_pred_dollar)
    r2 = r2_score(y_true_dollar, y_pred_dollar)

    nonzero = y_true_dollar > 0
    mape = np.mean(np.abs(
        (y_true_dollar[nonzero] - y_pred_dollar[nonzero]) / y_true_dollar[nonzero]
    )) * 100

    return {"RMSE ($)": rmse, "MAE ($)": mae, "MAPE (%)": mape, "R2": r2}


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a comparison DataFrame from {model_name: metrics_dict}."""
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df.round(2)


def error_by_segment(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    segment_series: pd.Series,
    price_actual: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute MAE and MAPE within each segment (brand, age bucket, etc.)."""
    y_true_dollar = np.expm1(y_true_log)
    y_pred_dollar = np.expm1(y_pred_log)
    if price_actual is not None:
        y_true_dollar = np.asarray(price_actual)

    err = np.abs(y_true_dollar - y_pred_dollar)
    pct_err = np.where(
        y_true_dollar > 0,
        err / np.where(y_true_dollar > 0, y_true_dollar, 1) * 100,
        np.nan,
    )

    df = pd.DataFrame({
        "abs_error": err,
        "pct_error": pct_err,
        "segment": segment_series.values,
    })
    agg = df.groupby("segment", observed=False).agg(
        MAE=("abs_error", "mean"),
        MAPE=("pct_error", "mean"),
        count=("abs_error", "count"),
    ).sort_values("MAE", ascending=False)
    return agg.round(2)


def gain_importance_table(model, top_n: int | None = 15) -> pd.Series:
    """Compute gain-based LightGBM feature importance as % of total gain.

    Gain-based (loss reduction), not split-count -- split-count inflates
    high-cardinality features and continuous ones regardless of real
    predictive value.

    Args:
        model: A fitted LightGBM sklearn wrapper (must expose `.booster_`).
        top_n: If given, only the top N features (by gain) are returned.

    Returns:
        pd.Series: Feature name -> % of total gain, sorted descending.
    """
    gain = model.booster_.feature_importance(importance_type="gain")
    imp = pd.Series(gain, index=model.booster_.feature_name())
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    if top_n is not None:
        imp = imp.head(top_n)
    return imp


def coverage(price_actual: np.ndarray, lo_dollar: np.ndarray, hi_dollar: np.ndarray) -> float:
    """Compute empirical coverage: fraction of actuals falling inside [lo, hi].

    Args:
        price_actual: Actual listed price per row.
        lo_dollar: Interval lower bound per row (dollar scale).
        hi_dollar: Interval upper bound per row (dollar scale).

    Returns:
        float: Fraction of rows where lo <= actual <= hi.
    """
    price_actual = np.asarray(price_actual)
    return float(np.mean((price_actual >= lo_dollar) & (price_actual <= hi_dollar)))


def coverage_by_segment(
    price_actual: np.ndarray,
    lo_dollar: np.ndarray,
    hi_dollar: np.ndarray,
    segment: pd.Series,
) -> pd.DataFrame:
    """Compute coverage and median interval width within each segment.

    Args:
        price_actual: Actual listed price per row.
        lo_dollar: Interval lower bound per row (dollar scale).
        hi_dollar: Interval upper bound per row (dollar scale).
        segment: Segment label per row (e.g. age bucket, price bucket).

    Returns:
        pd.DataFrame: One row per segment with columns `coverage`,
        `median_width`, `count`.
    """
    inside = (np.asarray(price_actual) >= lo_dollar) & (np.asarray(price_actual) <= hi_dollar)
    width = hi_dollar - lo_dollar
    df = pd.DataFrame({
        "inside": inside,
        "width": width,
        "segment": segment.values,
    })
    return (
        df.groupby("segment", observed=False)
        .agg(coverage=("inside", "mean"), median_width=("width", "median"), count=("inside", "count"))
        .round(4)
    )
