"""Feature engineering for the used-car dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REFERENCE_YEAR: int = 2021
AGE_MIN: int = 1


class FeatureEngineer:
    """Derives model-ready features from the cleaned dataset.

    Must be called AFTER DataCleaner. Does not touch the train/test split;
    encoders are intentionally left to the model pipeline to avoid leakage.

    Args:
        reference_year: Posting year to compute vehicle age against when no
            `posting_date` column is present. CLAUDE.md's "reference date
            trap": this MUST be the data's actual collection year (2021),
            never the real-world current year -- using today's date would
            silently corrupt every downstream age-derived feature and metric.
        age_min: Minimum allowed vehicle age; ages below this (data-entry
            errors where year > posting year, that slipped past cleaning)
            are clipped up to this floor.
    """

    def __init__(self, reference_year: int = REFERENCE_YEAR, age_min: int = AGE_MIN):
        self.reference_year = reference_year
        self.age_min = age_min

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all engineered features to a cleaned DataFrame.

        Args:
            df: Cleaned dataset (post-DataCleaner).

        Returns:
            pd.DataFrame: Copy of `df` with engineered columns added and the
            raw `description` column removed (if present).
        """
        df = df.copy()
        df = self._add_age(df)
        df = self._add_mileage_per_year(df)
        df = self._add_log_price(df)
        df = self._add_log_odometer(df)
        df = self._add_cylinders_numeric(df)
        df = self._add_description_features(df)
        logger.info("Feature engineering done. Shape: %s", df.shape)
        return df

    def _add_description_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Leakage-free trim/equipment keyword features (Phase 7B), then drop
        the raw `description` column -- it never reaches the processed
        parquet or the model in text form."""
        if "description" not in df.columns:
            return df
        from src.features.description import DescriptionFeatureExtractor

        df = DescriptionFeatureExtractor().fit_transform(df)
        return df.drop(columns=["description"])

    def _add_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute vehicle age relative to posting year, NOT the current real-world date.

        Args:
            df: DataFrame with a `year` column and optionally `posting_date`.

        Returns:
            pd.DataFrame: `df` with an `age` column added.
        """
        if "posting_date" in df.columns and df["posting_date"].notna().any():
            posting_year = df["posting_date"].dt.year.fillna(self.reference_year)
        else:
            posting_year = self.reference_year

        df["age"] = posting_year - df["year"].astype(int)
        # Clamp negative / zero ages (data entry errors that passed year filter)
        df["age"] = df["age"].clip(lower=self.age_min)
        return df

    def _add_mileage_per_year(self, df: pd.DataFrame) -> pd.DataFrame:
        df["mileage_per_year"] = df["odometer"] / df["age"]
        return df

    def _add_log_price(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log_price"] = np.log1p(df["price"])
        return df

    def _add_log_odometer(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log_odometer"] = np.log1p(df["odometer"])
        return df

    def _add_cylinders_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract integer cylinder count from strings like '6 cylinders'."""
        if "cylinders" not in df.columns:
            return df
        df["cylinders_num"] = (
            df["cylinders"]
            .str.extract(r"(\d+)")
            .astype(float)
        )
        return df
