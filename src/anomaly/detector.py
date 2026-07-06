"""Anomaly detection: residual-based (price model) and Isolation Forest."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

RANDOM_STATE: int = 42
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
    """

    def __init__(self, z_threshold: float = 3.5):
        self.z_threshold = z_threshold
        self.median_: float = 0.0
        self.mad_: float = 0.0

    def fit(self, y_log: np.ndarray, pred_log: np.ndarray) -> "ResidualAnomalyDetector":
        residual = np.asarray(y_log) - np.asarray(pred_log)
        self.median_ = float(np.median(residual))
        self.mad_ = float(np.median(np.abs(residual - self.median_)))
        if self.mad_ == 0:
            self.mad_ = float(np.std(residual)) or 1.0
        return self

    def score(self, y_log: np.ndarray, pred_log: np.ndarray) -> pd.DataFrame:
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
