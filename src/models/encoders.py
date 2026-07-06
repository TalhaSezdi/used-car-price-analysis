"""Leakage-safe encoding and imputation for the model pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        cols: list[str],
        n_folds: int = 5,
        smoothing: int = 20,
        random_state: int = RANDOM_STATE,
    ):
        self.cols = cols
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.random_state = random_state
        self.mapping_: dict[str, dict] = {}
        self.global_mean_: float = 0.0

    def _smoothed_means(self, x_col: pd.Series, y: pd.Series) -> pd.Series:
        stats = y.groupby(x_col).agg(["sum", "count"])
        return (
            (stats["sum"] + self.smoothing * self.global_mean_)
            / (stats["count"] + self.smoothing)
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SafeTargetEncoder":
        """Fit the full-train mapping used by transform (for TEST data).

        Args:
            X: Training feature matrix.
            y: Training target.

        Returns:
            SafeTargetEncoder: self.
        """
        self.global_mean_ = float(y.mean())
        for col in self.cols:
            self.mapping_[col] = self._smoothed_means(X[col], y).to_dict()
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit the mapping AND return OOF-encoded TRAIN data (no self-leakage).

        Args:
            X: Training feature matrix.
            y: Training target.

        Returns:
            pd.DataFrame: `X` with `self.cols` replaced by out-of-fold encodings.
        """
        from sklearn.model_selection import KFold

        self.global_mean_ = float(y.mean())
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
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
        """Encode using the full-train smoothed mapping (safe on TEST data).

        Args:
            X: Feature matrix to encode.

        Returns:
            pd.DataFrame: `X` with `self.cols` replaced. Unseen categories
            fall back to the global mean.
        """
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
        """Compute each category's training-set frequency.

        Args:
            X: Training feature matrix.
            y: Unused; present for scikit-learn API compatibility.

        Returns:
            FrequencyEncoder: self.
        """
        for col in self.cols:
            freq = X[col].value_counts(normalize=True)
            self.mapping_[col] = freq.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode using the fitted frequency mapping.

        Args:
            X: Feature matrix to encode.

        Returns:
            pd.DataFrame: `X` with `self.cols` replaced. Unseen categories
            fall back to 0.0.
        """
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.mapping_[col]).fillna(0.0)
        return X


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Full preprocessing: impute, encode low-card, encode high-card.

    Args:
        numeric_cols: Columns to impute with median + missing indicator.
        low_card_cols: Columns for one-hot encoding.
        high_card_cols: Columns for target/frequency encoding.
        high_card_method: 'target' or 'frequency' (for ablation A3).
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

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeaturePreprocessor":
        """Fit via fit_transform and discard the OOF-encoded output.

        Warning:
            Do NOT use fit() followed by transform() on the same training
            data -- transform() applies the leaky full-train mapping, not
            the out-of-fold encoding fit_transform() produces. Always call
            fit_transform() directly on the training split.

        Args:
            X: Training feature matrix.
            y: Training target.

        Returns:
            FeaturePreprocessor: self.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit all state and return the TRAIN matrix with OOF target encoding.

        Must be used for the training split. Using fit() + transform() on the
        same training data would apply the leaky full-train mapping instead of
        the out-of-fold encoding.

        Args:
            X: Training feature matrix.
            y: Training target.

        Returns:
            pd.DataFrame: Fully encoded/imputed training matrix.
        """
        for col in self.numeric_cols:
            if col in X.columns:
                self.medians_[col] = X[col].median()

        X = self._impute_numeric(X)

        hc = self._fillna_high_card(X)
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
        """Encode/impute using state fitted on TRAIN (safe on TEST data).

        Args:
            X: Feature matrix to transform.

        Returns:
            pd.DataFrame: Transformed matrix with columns aligned to the
            training matrix's columns (missing ones added as all-zero).
        """
        X = self._impute_numeric(X)
        self._fillna_high_card(X)
        if self.encoder_ is not None:
            X = self.encoder_.transform(X)  # full-train mapping (safe on test)
        X = self._onehot(X)
        for col in self.train_columns_:
            if col not in X.columns:
                X[col] = 0
        return X[self.train_columns_]

    def _fillna_high_card(self, X: pd.DataFrame) -> list[str]:
        """Fill missing values in the present high-cardinality columns.

        Args:
            X: Feature matrix, modified in place.

        Returns:
            list[str]: `self.high_card_cols` filtered to columns present in `X`.
        """
        hc = [c for c in self.high_card_cols if c in X.columns]
        for col in hc:
            X[col] = X[col].fillna("missing")
        return hc

    def _impute_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Median-impute numeric columns, adding a missing-value indicator per column.

        Args:
            X: Feature matrix.

        Returns:
            pd.DataFrame: Copy of `X` with numeric columns imputed and
            `{col}_missing` indicator columns added.
        """
        X = X.copy()
        for col in self.numeric_cols:
            if col in X.columns:
                miss_flag = f"{col}_missing"
                X[miss_flag] = X[col].isna().astype(int) if X[col].isna().any() else 0
                X[col] = X[col].fillna(self.medians_.get(col, 0))
        return X

    def _onehot(self, X: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode the present low-cardinality columns (drop_first=True).

        Args:
            X: Feature matrix.

        Returns:
            pd.DataFrame: `X` with low-cardinality columns one-hot encoded.
        """
        low = [c for c in self.low_card_cols if c in X.columns]
        if low:
            for col in low:
                X[col] = X[col].fillna("missing")
            X = pd.get_dummies(X, columns=low, drop_first=True, dtype=int)
        return X
