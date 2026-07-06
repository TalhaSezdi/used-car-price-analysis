"""Leakage-safe encoding and imputation for the model pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

RANDOM_STATE: int = 42


class SafeTargetEncoder(BaseEstimator, TransformerMixin):
    """KFold out-of-fold target encoder with smoothing for high-card categoricals.

    Leakage safety:
      - fit_transform (TRAIN): each row is encoded using ONLY the other folds'
        targets, so a row never sees its own label. This is the whole point of
        KFold target encoding -- without it, the encoded feature contains the
        target and the model overfits it.
      - transform (TEST / new data): uses a smoothed mapping fitted on the FULL
        training set. Unseen categories fall back to the global mean.

    Smoothing shrinks rare-category means toward the global mean (Bayesian).
    """

    def __init__(self, cols: list[str], n_folds: int = 5, smoothing: int = 20):
        self.cols = cols
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.mapping_: dict[str, dict] = {}
        self.global_mean_: float = 0.0

    def _smoothed_means(self, x_col: pd.Series, y: pd.Series) -> pd.Series:
        stats = y.groupby(x_col).agg(["sum", "count"])
        return (
            (stats["sum"] + self.smoothing * self.global_mean_)
            / (stats["count"] + self.smoothing)
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SafeTargetEncoder":
        """Fit the full-train mapping used by transform (for TEST data)."""
        self.global_mean_ = float(y.mean())
        for col in self.cols:
            self.mapping_[col] = self._smoothed_means(X[col], y).to_dict()
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit the mapping AND return OOF-encoded TRAIN data (no self-leakage)."""
        from sklearn.model_selection import KFold

        self.global_mean_ = float(y.mean())
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_STATE)
        X_enc = X.copy()

        for col in self.cols:
            oof = pd.Series(np.nan, index=X.index, dtype=float)
            for train_idx, val_idx in kf.split(X):
                means = self._smoothed_means(
                    X[col].iloc[train_idx], y.iloc[train_idx]
                )
                oof.iloc[val_idx] = (
                    X[col].iloc[val_idx].map(means)
                    .fillna(self.global_mean_).values
                )
            X_enc[col] = oof.values
            # full-train smoothed mapping, used later by transform on test
            self.mapping_[col] = self._smoothed_means(X[col], y).to_dict()

        return X_enc

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace categories with their training-set frequency."""

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.mapping_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyEncoder":
        for col in self.cols:
            freq = X[col].value_counts(normalize=True)
            self.mapping_[col] = freq.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.mapping_[col]).fillna(0.0)
        return X


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Full preprocessing: impute, encode low-card, encode high-card.

    Parameters
    ----------
    high_card_cols : columns for target/frequency encoding
    low_card_cols : columns for one-hot encoding
    numeric_cols : columns to impute with median + missing indicator
    high_card_method : 'target' or 'frequency' (for ablation A3)
    """

    def __init__(
        self,
        numeric_cols: list[str],
        low_card_cols: list[str],
        high_card_cols: list[str],
        high_card_method: str = "target",
    ):
        self.numeric_cols = numeric_cols
        self.low_card_cols = low_card_cols
        self.high_card_cols = high_card_cols
        self.high_card_method = high_card_method
        self.medians_: dict[str, float] = {}
        self.missing_cols_: list[str] = []
        self.ohe_cols_: list[str] = []
        self.encoder_ = None
        self.train_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeaturePreprocessor":
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit all state and return the TRAIN matrix with OOF target encoding.

        Must be used for the training split. Using fit() + transform() on the
        same training data would apply the leaky full-train mapping instead of
        the out-of-fold encoding.
        """
        for col in self.numeric_cols:
            if col in X.columns:
                self.medians_[col] = X[col].median()

        X = self._impute_numeric(X)

        hc = [c for c in self.high_card_cols if c in X.columns]
        for col in hc:
            X[col] = X[col].fillna("missing")
        if self.high_card_method == "target" and y is not None and hc:
            self.encoder_ = SafeTargetEncoder(cols=hc)
            X = self.encoder_.fit_transform(X, y)  # OOF, no self-leakage
        elif self.high_card_method == "frequency" and hc:
            self.encoder_ = FrequencyEncoder(cols=hc)
            self.encoder_.fit(X)
            X = self.encoder_.transform(X)

        X = self._onehot(X)
        self.train_columns_ = list(X.columns)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._impute_numeric(X)
        hc = [c for c in self.high_card_cols if c in X.columns]
        for col in hc:
            X[col] = X[col].fillna("missing")
        if self.encoder_ is not None:
            X = self.encoder_.transform(X)  # full-train mapping (safe on test)
        X = self._onehot(X)
        for col in self.train_columns_:
            if col not in X.columns:
                X[col] = 0
        return X[self.train_columns_]

    def _impute_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.numeric_cols:
            if col in X.columns:
                miss_flag = f"{col}_missing"
                X[miss_flag] = X[col].isna().astype(int) if X[col].isna().any() else 0
                X[col] = X[col].fillna(self.medians_.get(col, 0))
        return X

    def _onehot(self, X: pd.DataFrame) -> pd.DataFrame:
        low = [c for c in self.low_card_cols if c in X.columns]
        if low:
            for col in low:
                X[col] = X[col].fillna("missing")
            X = pd.get_dummies(X, columns=low, drop_first=True, dtype=int)
        return X
