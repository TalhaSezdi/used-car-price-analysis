"""Feature matrix assembly and train/test split."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)

TEST_SIZE: float = 0.20
VAL_SIZE_OF_REMAINDER: float = 0.25  # 0.25 * 0.80 = 0.20 -> 60/20/20 overall

TARGET_COL: str = "log_price"
RAW_PRICE_COL: str = "price"

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
    """Three-way stratified split for a leakage-safe model-selection workflow.

    - `X_train`, `y_train`, `price_train` (60%): fit models during selection.
    - `X_val`, `y_val`, `price_val` (20%): hold-out for model comparison and
       ablation decisions -- never used to fit the final model.
    - `X_test`, `y_test`, `price_test` (20%): final unbiased hold-out. Only the
       CHOSEN model, refit on train+val, is evaluated here.
    - `X_train_full`, `y_train_full`, `price_train_full` (80% = train + val):
       used to refit the final chosen model, and by downstream scripts that
       do not perform model selection (predict_intervals, probe_split_leakage,
       ablations) so their behavior stays identical to the pre-three-way-split
       setup.
    """
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    price_train: pd.Series
    price_val: pd.Series
    price_test: pd.Series
    X_train_full: pd.DataFrame
    y_train_full: pd.Series
    price_train_full: pd.Series


def select_features(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Assemble the model feature matrix X, log target y, and raw price.

    Single source of truth for feature selection so the train script and the
    anomaly-scoring script build identical X. Drops leakage / redundant columns
    (VIN, region, lat/long, year collinear with age, raw price, description)
    by using a whitelist of features to keep, rather than a blacklist of
    columns to drop.

    Args:
        df: Cleaned + feature-engineered dataset.
        numeric_features: Numeric feature columns to keep. Defaults to a copy
            of NUMERIC_FEATURES if None.
        categorical_features: Categorical feature columns to keep. Defaults
            to a copy of CATEGORICAL_FEATURES if None.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.Series]: (X, y, raw_price) where y
        is `log_price` and raw_price is the original dollar `price`.
    """
    numeric_features = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    categorical_features = (
        categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    )

    y = df[TARGET_COL].copy()
    raw_price = df[RAW_PRICE_COL].copy()

    keep = [c for c in numeric_features + categorical_features if c in df.columns]
    X = df[keep].copy()
    return X, y, raw_price


def build_split(df: pd.DataFrame, test_size: float = TEST_SIZE,
                val_size_of_remainder: float = VAL_SIZE_OF_REMAINDER) -> SplitData:
    """Assemble X/y and do a stratified three-way random split.

    Two nested stratified splits so the test set stays byte-identical to the
    prior two-way split (same seed, same test_size, same stratification):
      1. Split all rows into train_full (80%) and test (20%).
      2. Split train_full into train (60%) and val (20%).

    Stratification is by price decile on each level so all three sets share
    the same price distribution -- important because price is right-skewed
    even after log transform.
    """
    X, y, raw_price = select_features(df)
    price_decile = pd.qcut(raw_price, q=10, labels=False, duplicates="drop")

    # Step 1: carve out the final test set. Byte-identical to the old split.
    (X_train_full, X_test,
     y_train_full, y_test,
     price_train_full, price_test,
     strat_train_full, _) = train_test_split(
        X, y, raw_price, price_decile,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=price_decile,
    )

    # Step 2: split the remaining 80% into train (60%) and val (20%).
    (X_train, X_val,
     y_train, y_val,
     price_train, price_val) = train_test_split(
        X_train_full, y_train_full, price_train_full,
        test_size=val_size_of_remainder,
        random_state=RANDOM_STATE,
        stratify=strat_train_full,
    )

    logger.info(
        "Split: train=%d, val=%d, test=%d (%.0f%%/%.0f%%/%.0f%%)",
        len(X_train), len(X_val), len(X_test),
        (1 - test_size) * (1 - val_size_of_remainder) * 100,
        (1 - test_size) * val_size_of_remainder * 100,
        test_size * 100,
    )
    return SplitData(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        price_train=price_train, price_val=price_val, price_test=price_test,
        X_train_full=X_train_full, y_train_full=y_train_full,
        price_train_full=price_train_full,
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
