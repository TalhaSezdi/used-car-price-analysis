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
    """

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
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
        """Vehicle age relative to posting year, NOT the current real-world date."""
        if "posting_date" in df.columns and df["posting_date"].notna().any():
            posting_year = df["posting_date"].dt.year.fillna(REFERENCE_YEAR)
        else:
            posting_year = REFERENCE_YEAR

        df["age"] = posting_year - df["year"].astype(int)
        # Clamp negative / zero ages (data entry errors that passed year filter)
        df["age"] = df["age"].clip(lower=AGE_MIN)
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
