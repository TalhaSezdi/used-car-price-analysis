"""Anomaly detection: residual-based (price model) and Isolation Forest."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)

MAD_SCALE: float = 1.4826  # makes MAD a consistent estimator of std under normality


class ResidualAnomalyDetector:
    """Flag listings priced far from their model-predicted value.

    Works in LOG space so the residual is roughly a percentage error and is
    comparable across the whole price range (a $5k gap is huge on a $6k car,
    trivial on an $80k car -- dollar residuals are heteroscedastic).

    Robust z-score uses median + MAD instead of mean + std, because the very
    outliers we are hunting inflate the mean and std and mask themselves.

    Residual sign (actual_log - pred_log):
      - negative -> listed BELOW model value -> "underpriced": scam / too-good
        -to-be-true / hidden defect.
      - positive -> listed ABOVE model value -> "overpriced": data-entry error,
        spam, or a genuinely rare/special trim.

    Args:
        z_threshold: Absolute robust-z above which a row is flagged.
    """

    def __init__(self, z_threshold: float = 3.5):
        self.z_threshold = z_threshold
        self.median_: float = 0.0
        self.mad_: float = 0.0

    def fit(self, y_log: np.ndarray, pred_log: np.ndarray) -> "ResidualAnomalyDetector":
        """Fit the residual median and MAD (median absolute deviation).

        Args:
            y_log: Actual target values (log-price scale).
            pred_log: Predicted values (log-price scale), ideally out-of-fold.

        Returns:
            ResidualAnomalyDetector: self.
        """
        residual = np.asarray(y_log) - np.asarray(pred_log)
        self.median_ = float(np.median(residual))
        self.mad_ = float(np.median(np.abs(residual - self.median_)))
        if self.mad_ == 0:
            self.mad_ = float(np.std(residual)) or 1.0
        return self

    def score(self, y_log: np.ndarray, pred_log: np.ndarray) -> pd.DataFrame:
        """Compute the robust z-score, flag, and direction per row.

        Args:
            y_log: Actual target values (log-price scale).
            pred_log: Predicted values (log-price scale).

        Returns:
            pd.DataFrame: Columns `residual_log`, `residual_z`,
            `residual_flag`, `direction` ("underpriced"/"overpriced").
        """
        residual = np.asarray(y_log) - np.asarray(pred_log)
        robust_z = (residual - self.median_) / (MAD_SCALE * self.mad_)
        flag = np.abs(robust_z) > self.z_threshold
        direction = np.where(residual < 0, "underpriced", "overpriced")
        return pd.DataFrame({
            "residual_log": residual,
            "residual_z": robust_z,
            "residual_flag": flag,
            "direction": direction,
        })


class IsolationForestDetector:
    """Unsupervised structural-anomaly detector on the numeric feature space.

    Catches listings whose attribute COMBINATION is odd (e.g. a 20-year-old car
    with 2,000 miles, or an impossible age/odometer/price mix) independent of
    the price model.

    Why Isolation Forest and not LOF / One-Class SVM: IF is ~O(n log n) and
    scales to ~200k rows in seconds via random subsampling; LOF is ~O(n^2) on
    distances and One-Class SVM is impractical at this size. IF also needs no
    feature scaling (it splits on random thresholds).

    Args:
        contamination: Expected proportion of anomalies (passed to IsolationForest).
        n_estimators: Number of trees in the forest.
    """

    def __init__(self, contamination: float = 0.01, n_estimators: int = 200):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model_ = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.feature_cols_: list[str] = []

    def fit_score(self, X_numeric: pd.DataFrame) -> pd.DataFrame:
        """Fit on the numeric matrix and return score + flag per row.

        NaNs are median-imputed (IF cannot take NaN). if_score is oriented so
        HIGHER = MORE anomalous.

        Args:
            X_numeric: Numeric feature matrix (structural features only).

        Returns:
            pd.DataFrame: Columns `if_score` (higher = more anomalous) and
            `if_flag` (True where IsolationForest predicts an outlier).
        """
        self.feature_cols_ = list(X_numeric.columns)
        X = X_numeric.fillna(X_numeric.median())
        self.model_.fit(X)
        # score_samples: higher = more normal. Negate so higher = more anomalous.
        if_score = -self.model_.score_samples(X)
        if_flag = self.model_.predict(X) == -1
        return pd.DataFrame({
            "if_score": if_score,
            "if_flag": if_flag,
        }, index=X_numeric.index)


def in_sample_residual_std(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    low_card_cols: list[str],
    high_card_cols: list[str],
    n_estimators: int = 600,
) -> float:
    """Fit one LightGBM on ALL rows and predict in-sample -- the leakage-risk demo.

    Compared against out-of-fold residual std (see models.train.oof_log_predictions)
    to show how much smaller (optimistically biased) in-sample residuals are.

    Args:
        X: Feature matrix (all rows, e.g. from models.dataset.select_features).
        y: Target (log-price scale).
        numeric_cols: Numeric feature columns for FeaturePreprocessor.
        low_card_cols: Low-cardinality categorical columns for FeaturePreprocessor.
        high_card_cols: High-cardinality categorical columns for FeaturePreprocessor.
        n_estimators: Fixed boosting round count (no early stopping).

    Returns:
        float: Standard deviation of the in-sample residual (log scale).
    """
    import lightgbm as lgb

    from src.models.encoders import FeaturePreprocessor
    from src.models.train import LGBM_PARAMS

    prep = FeaturePreprocessor(
        numeric_cols=numeric_cols,
        low_card_cols=low_card_cols,
        high_card_cols=high_card_cols,
        high_card_method="target",
    )
    Xt = prep.fit_transform(X, y)
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **LGBM_PARAMS)
    model.fit(Xt, y)
    resid = y.values - model.predict(Xt)
    return float(np.std(resid))


def is_strong_residual(
    abs_z: np.ndarray, abs_pct: np.ndarray, z_strong: float = 5.0, pct_strong: float = 85.0
) -> np.ndarray:
    """The STRONG-tier anomaly rule: extreme by BOTH z-score and pct deviation.

    Survives the model-error confound: with a model MAPE of ~37%, a deviation
    this large in both senses is far outside the plausible noise band.

    Args:
        abs_z: Absolute robust residual z-score per row.
        abs_pct: Absolute residual percentage deviation per row.
        z_strong: Z-score threshold.
        pct_strong: Percentage-deviation threshold.

    Returns:
        np.ndarray: Boolean array, True where both thresholds are exceeded.
    """
    return (np.asarray(abs_z) > z_strong) & (np.asarray(abs_pct) > pct_strong)


def tier_anomalies(
    residual_z: pd.Series,
    residual_flag: pd.Series,
    residual_pct: pd.Series,
    if_flag: pd.Series,
    z_strong: float = 5.0,
    pct_strong: float = 85.0,
) -> pd.DataFrame:
    """Tier residual + Isolation Forest flags into a single priority label per row.

    STRONG (extreme by both z and pct) is far outside the plausible model-noise
    band; MODERATE (flagged but not STRONG) may just be model error on a rare
    car and needs human review rather than auto-action.

    Args:
        residual_z: Robust residual z-score per row.
        residual_flag: Boolean residual flag per row (|z| > detector's threshold).
        residual_pct: Residual percentage deviation per row.
        if_flag: Boolean Isolation Forest flag per row.
        z_strong: Z-score threshold for the STRONG tier.
        pct_strong: Percentage-deviation threshold for the STRONG tier.

    Returns:
        pd.DataFrame: Columns `strong_resid`, `moderate_resid`, `priority`
        (one of "HIGH (strong mispriced + structural)", "strong mispriced",
        "moderate mispriced + structural", "moderate mispriced (may be model
        error)", "structural only", or "normal").
    """
    abs_z = np.abs(residual_z)
    abs_pct = np.abs(residual_pct)
    strong_resid = is_strong_residual(abs_z, abs_pct, z_strong, pct_strong)
    moderate_resid = np.asarray(residual_flag) & ~strong_resid
    if_flag_arr = np.asarray(if_flag)

    conditions = [
        strong_resid & if_flag_arr,
        strong_resid,
        moderate_resid & if_flag_arr,
        moderate_resid,
        if_flag_arr,
    ]
    choices = [
        "HIGH (strong mispriced + structural)",
        "strong mispriced",
        "moderate mispriced + structural",
        "moderate mispriced (may be model error)",
        "structural only",
    ]
    priority = np.select(conditions, choices, default="normal")
    index = residual_z.index if isinstance(residual_z, pd.Series) else None
    return pd.DataFrame({
        "strong_resid": strong_resid,
        "moderate_resid": moderate_resid,
        "priority": priority,
    }, index=index)


def fat_tail_comparison(
    abs_z: pd.Series, thresholds: list[float] = (3.5, 5, 7, 10)
) -> list[tuple[float, int, float, float, float]]:
    """Compare observed flag counts at each threshold against the Gaussian expectation.

    Shows the residual distribution's tails are much fatter than a Gaussian
    reference -- the flag threshold is an operational (capacity) choice, not a
    rarity claim.

    Args:
        abs_z: Absolute robust residual z-score per row.
        thresholds: Z-score thresholds to compare.

    Returns:
        list[tuple]: One tuple per threshold: (threshold, observed_count,
        observed_pct, gaussian_expected_count, gaussian_expected_pct).
    """
    from scipy.stats import norm

    n = len(abs_z)
    result = []
    for t in thresholds:
        cnt = int((abs_z > t).sum())
        gauss = 2 * norm.sf(t) * n
        result.append((t, cnt, cnt / n * 100, gauss, gauss / n * 100))
    return result
