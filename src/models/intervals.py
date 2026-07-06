"""Prediction intervals via split-conformal quantile regression (CQR).

Point predictions (Phase 3) give one number. A marketplace wants a range:
"this car is worth $14k-$19k" is a more honest, more usable statement than
"$16,400" when the model's MAPE is ~37%. This module adds that range with a
finite-sample coverage guarantee, instead of an ad-hoc +/- band.

Method: Romano, Patterson & Candes (2019), "Conformalized Quantile
Regression". Train lower/upper LightGBM quantile regressors on TRAIN, then
calibrate the band width on a held-out CALIBRATION set (carved from TRAIN,
never TEST -- see src.models.dataset.split_calibration) so the interval has
valid marginal coverage under exchangeability, regardless of how well the
raw quantile models are calibrated on their own.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def _quantile_params(alpha: float) -> dict:
    from src.models.train import LGBM_PARAMS

    params = dict(LGBM_PARAMS)
    params.update(objective="quantile", alpha=alpha)
    return params


def _fit_lgbm_quantile(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float,
    n_estimators: int = 3000,
    random_state: int = RANDOM_STATE,
):
    """Fit one LightGBM quantile regressor, early-stopping on a TRAIN-carved split.

    Mirrors src.models.train._fit_lgbm: the validation split for early
    stopping comes out of X_train, never out of calibration or test.
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=random_state
    )
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **_quantile_params(alpha))
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


class ConformalIntervalModel:
    """Split-conformal quantile regression for a (1 - alpha) prediction interval.

    Operates in log-price space (the model's native target scale); use
    predict_interval_dollar for the expm1-converted, business-facing output.
    """

    def __init__(self, alpha: float = 0.10, n_estimators: int = 3000):
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.lower_alpha = alpha / 2
        self.upper_alpha = 1 - alpha / 2
        self.model_lo_ = None
        self.model_hi_ = None
        self.correction_: float = 0.0

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
    ) -> "ConformalIntervalModel":
        self.model_lo_ = _fit_lgbm_quantile(
            X_train, y_train, self.lower_alpha, self.n_estimators
        )
        self.model_hi_ = _fit_lgbm_quantile(
            X_train, y_train, self.upper_alpha, self.n_estimators
        )

        lo_cal = self.model_lo_.predict(X_calib)
        hi_cal = self.model_hi_.predict(X_calib)
        y_cal = np.asarray(y_calib)

        # CQR nonconformity score: how far outside the raw band the true
        # value falls (negative = inside the band).
        scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)

        n = len(y_cal)
        # Finite-sample-corrected quantile level (Romano et al. 2019, eq. 6).
        q_level = min(1.0, (1 - self.alpha) * (1 + 1 / n))
        self.correction_ = float(np.quantile(scores, q_level))
        logger.info(
            "Conformal calibration (alpha=%.3f): correction=%.4f (log scale)",
            self.alpha, self.correction_,
        )
        return self

    def predict_interval_log(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated (lo, hi) in log-price scale."""
        lo = self.model_lo_.predict(X) - self.correction_
        hi = self.model_hi_.predict(X) + self.correction_
        return lo, hi

    def predict_interval_dollar(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated (lo, hi) in dollar scale (expm1 of the log interval)."""
        lo, hi = self.predict_interval_log(X)
        return np.expm1(lo), np.expm1(hi)

    def raw_interval_log(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Uncalibrated quantile band (no conformal correction) -- for comparison."""
        return self.model_lo_.predict(X), self.model_hi_.predict(X)


class MondrianConformalIntervalModel(ConformalIntervalModel):
    """Group-conditional (Mondrian) CQR: one conformal correction per group.

    Standard split-conformal guarantees MARGINAL coverage only; Phase 6B showed
    it under-covers the price tails. Mondrian conformal fits the correction
    separately within each group, restoring the coverage guarantee per group.

    Groups are bins of the raw quantile-band midpoint (log scale) -- a quantity
    the model itself produces for any row, so binning works identically at
    calibration and inference time. Binning on the ACTUAL price would be both
    unavailable at inference ("what is this car worth" has no price yet) and
    invalid (conditioning calibration on the ground truth breaks
    exchangeability).
    """

    def __init__(self, alpha: float = 0.10, n_estimators: int = 3000, n_bins: int = 5):
        super().__init__(alpha=alpha, n_estimators=n_estimators)
        self.n_bins = n_bins
        self.bin_edges_: np.ndarray | None = None
        self.corrections_: np.ndarray | None = None

    def _midpoint(self, X: pd.DataFrame) -> np.ndarray:
        lo, hi = self.raw_interval_log(X)
        return (lo + hi) / 2

    def _bin_index(self, mid: np.ndarray) -> np.ndarray:
        # Interior edges only; values outside the calibration range fall into
        # the first/last bin.
        return np.clip(np.digitize(mid, self.bin_edges_), 0, self.n_bins - 1)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
    ) -> "MondrianConformalIntervalModel":
        super().fit(X_train, y_train, X_calib, y_calib)  # fits quantile models + global correction

        mid_cal = self._midpoint(X_calib)
        self.bin_edges_ = np.quantile(
            mid_cal, np.linspace(0, 1, self.n_bins + 1)[1:-1]
        )
        bins_cal = self._bin_index(mid_cal)

        lo_cal = self.model_lo_.predict(X_calib)
        hi_cal = self.model_hi_.predict(X_calib)
        y_cal = np.asarray(y_calib)
        scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)

        corrections = np.empty(self.n_bins)
        for b in range(self.n_bins):
            mask = bins_cal == b
            n_b = int(mask.sum())
            q_level = min(1.0, (1 - self.alpha) * (1 + 1 / n_b))
            corrections[b] = np.quantile(scores[mask], q_level)
            logger.info(
                "Mondrian bin %d (alpha=%.3f): n=%d, correction=%.4f",
                b, self.alpha, n_b, corrections[b],
            )
        self.corrections_ = corrections
        return self

    def predict_interval_log(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lo = self.model_lo_.predict(X)
        hi = self.model_hi_.predict(X)
        corr = self.corrections_[self._bin_index((lo + hi) / 2)]
        return lo - corr, hi + corr

    def predict_interval_dollar_global(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Standard (global-correction) CQR interval from the same fitted models.

        super().fit already computed the global correction, so the standard
        variant comes for free -- no need to train a second pair of quantile
        models just to compare Mondrian against it.
        """
        lo, hi = super().predict_interval_log(X)
        return np.expm1(lo), np.expm1(hi)


def fit_median_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 3000):
    """Single alpha=0.5 quantile LightGBM, used as the point estimate for reporting."""
    return _fit_lgbm_quantile(X_train, y_train, alpha=0.5, n_estimators=n_estimators)


def coverage(price_actual: np.ndarray, lo_dollar: np.ndarray, hi_dollar: np.ndarray) -> float:
    """Empirical coverage: fraction of actuals falling inside [lo, hi]."""
    price_actual = np.asarray(price_actual)
    return float(np.mean((price_actual >= lo_dollar) & (price_actual <= hi_dollar)))


def coverage_by_segment(
    price_actual: np.ndarray,
    lo_dollar: np.ndarray,
    hi_dollar: np.ndarray,
    segment: pd.Series,
) -> pd.DataFrame:
    """Coverage and median width within each segment (age bucket, price bucket, ...)."""
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
