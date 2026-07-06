"""Feature matrix assembly and train/test split."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

TARGET_COL: str = "log_price"
RAW_PRICE_COL: str = "price"

DROP_BEFORE_MODEL: list[str] = [
    "price", "log_price",
    "VIN", "region", "lat", "long",
    "year",
    "posting_date",
    "size",
]

NUMERIC_FEATURES: list[str] = [
    "age", "odometer", "log_odometer", "mileage_per_year", "cylinders_num",
    # Phase 7B: leakage-free description-derived features (Ablation A4 showed
    # a real improvement: RMSE -5.3%, MAPE -4.6pp -- see docs/phase7_results.md).
    "desc_trim_luxury", "desc_equip_count", "desc_len_log",
]

CATEGORICAL_FEATURES: list[str] = [
    "manufacturer", "model", "condition", "cylinders", "fuel",
    "title_status", "transmission", "drive", "type", "paint_color", "state",
]

HIGH_CARD_FEATURES: list[str] = ["model"]
LOW_CARD_FEATURES: list[str] = [
    f for f in CATEGORICAL_FEATURES if f not in HIGH_CARD_FEATURES
]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    price_test: pd.Series


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Assemble the model feature matrix X, log target y, and raw price.

    Single source of truth for feature selection so the train script and the
    anomaly-scoring script build identical X. Drops leakage / redundant columns
    (VIN, region, lat/long, year collinear with age, raw price, description).
    """
    y = df[TARGET_COL].copy()
    raw_price = df[RAW_PRICE_COL].copy()

    keep = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[keep].copy()
    return X, y, raw_price


def build_split(df: pd.DataFrame, test_size: float = TEST_SIZE) -> SplitData:
    """Assemble X/y and do stratified random split.

    Stratification is by price decile so train and test share the same
    price distribution -- important because price is right-skewed even
    after log transform.
    """
    X, y, raw_price = select_features(df)

    price_decile = pd.qcut(raw_price, q=10, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test, _, price_test_idx = train_test_split(
        X, y, raw_price,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=price_decile,
    )

    logger.info(
        "Split: train=%d, test=%d (%.0f%%)",
        len(X_train), len(X_test), test_size * 100,
    )
    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        price_test=price_test_idx,
    )


def split_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    calib_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carve a calibration set out of TRAIN for conformal interval calibration.

    Never touches TEST -- the calibration set is a second, disjoint hold-out
    used only to measure how wrong the raw quantile models are, so the
    conformal correction stays valid on the real test set.
    """
    return train_test_split(X_train, y_train, test_size=calib_size, random_state=random_state)
