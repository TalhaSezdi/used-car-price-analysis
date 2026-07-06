"""Tests for src/models/dataset.py -- feature selection and train/test split."""

import numpy as np
import pandas as pd

from src.models.dataset import build_split, select_features, split_calibration


def _synthetic_df(n=500):
    rng = np.random.RandomState(42)
    price = rng.uniform(500, 150_000, n)
    return pd.DataFrame({
        "VIN": [f"vin{i}" for i in range(n)],
        "region": rng.choice(["sf", "la"], n),
        "lat": rng.uniform(30, 40, n),
        "long": rng.uniform(-120, -100, n),
        "year": rng.randint(1990, 2021, n),
        "posting_date": pd.Timestamp("2021-05-01"),
        "size": rng.choice(["compact", "full-size"], n),
        "price": price,
        "log_price": np.log1p(price),
        "age": rng.randint(1, 20, n),
        "odometer": rng.uniform(1_000, 200_000, n),
        "log_odometer": rng.uniform(5, 12, n),
        "mileage_per_year": rng.uniform(1_000, 20_000, n),
        "cylinders_num": rng.choice([4, 6, 8], n).astype(float),
        "desc_trim_luxury": rng.randint(0, 2, n),
        "desc_equip_count": rng.randint(0, 5, n),
        "desc_len_log": rng.uniform(3, 8, n),
        "manufacturer": rng.choice(["ford", "toyota"], n),
        "model": rng.choice([f"model_{i}" for i in range(5)], n),
        "condition": rng.choice(["good", "fair"], n),
        "cylinders": rng.choice(["4 cylinders", "6 cylinders"], n),
        "fuel": rng.choice(["gas", "diesel"], n),
        "title_status": rng.choice(["clean", "rebuilt"], n),
        "transmission": rng.choice(["automatic", "manual"], n),
        "drive": rng.choice(["4wd", "fwd"], n),
        "type": rng.choice(["sedan", "truck"], n),
        "paint_color": rng.choice(["black", "white"], n),
        "state": rng.choice(["ca", "tx"], n),
    })


def test_select_features_excludes_leakage_columns():
    df = _synthetic_df()
    X, y, raw_price = select_features(df)
    for leaky_col in ["VIN", "region", "lat", "long", "year", "posting_date", "size", "price", "log_price"]:
        assert leaky_col not in X.columns


def test_select_features_only_keeps_present_columns():
    df = _synthetic_df().drop(columns=["desc_trim_luxury", "desc_equip_count", "desc_len_log"])
    X, y, raw_price = select_features(df)  # must not raise KeyError
    assert "desc_trim_luxury" not in X.columns


def test_select_features_custom_feature_lists():
    df = _synthetic_df()
    X, y, raw_price = select_features(df, numeric_features=["age"], categorical_features=["manufacturer"])
    assert set(X.columns) == {"age", "manufacturer"}


def test_build_split_sizes():
    df = _synthetic_df(n=1000)
    split = build_split(df)
    n = len(df)
    assert abs(len(split.X_train) / n - 0.60) < 0.02
    assert abs(len(split.X_val) / n - 0.20) < 0.02
    assert abs(len(split.X_test) / n - 0.20) < 0.02
    assert len(split.X_train) + len(split.X_val) + len(split.X_test) == n


def test_build_split_test_set_stable_across_val_size_changes():
    df = _synthetic_df(n=1000)
    split_a = build_split(df, val_size_of_remainder=0.25)
    split_b = build_split(df, val_size_of_remainder=0.30)
    assert list(split_a.X_test.index) == list(split_b.X_test.index)


def test_split_calibration_disjoint_and_sizes_sum():
    df = _synthetic_df(n=200)
    X, y, _ = select_features(df)
    X_tr, X_cal, y_tr, y_cal = split_calibration(X, y, calib_size=0.2)
    assert set(X_tr.index).isdisjoint(set(X_cal.index))
    assert len(X_tr) + len(X_cal) == len(X)
