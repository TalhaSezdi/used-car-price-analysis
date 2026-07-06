"""Tests for src/evaluation/insights.py (extracted from scripts/run_eda.py)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.insights import compute_eda_insights


@pytest.fixture
def eda_df():
    rng = np.random.RandomState(42)
    n = 500
    age = rng.choice([1, 5, 10, 15], n)
    price = 30_000 - age * 1500 + rng.normal(0, 500, n)
    price = np.clip(price, 500, None)
    odometer = rng.uniform(1_000, 200_000, n)
    return pd.DataFrame({
        "manufacturer": rng.choice(["ford", "toyota", "honda", "bmw", "kia", "chevrolet"], n),
        "model": rng.choice([f"m{i}" for i in range(20)], n),
        "price": price,
        "log_price": np.log1p(price),
        "age": age,
        "year": 2021 - age,
        "odometer": odometer,
        "mileage_per_year": odometer / age,
        "cylinders_num": rng.choice([4, 6, 8], n).astype(float),
        "state": rng.choice(["ca", "ak", "tx", "ny"], n),
        "VIN": [f"vin{i}" if i % 3 else None for i in range(n)],
        "condition": rng.choice(["good", "fair", None], n),
        "drive": rng.choice(["4wd", "fwd"], n),
        "posting_date": pd.to_datetime(
            rng.choice(pd.date_range("2021-04-01", "2021-05-01"), n), utc=True
        ),
    })


def test_compute_eda_insights_returns_markdown_with_all_ten_findings(eda_df):
    report = compute_eda_insights(eda_df)
    assert isinstance(report, str)
    for i in range(1, 11):
        assert f"### {i}." in report


def test_compute_eda_insights_depreciation_numbers_match_manual_medians(eda_df):
    report = compute_eda_insights(eda_df)
    age1_median = eda_df[eda_df["age"] == 1]["price"].median()
    assert f"${age1_median:,.0f}" in report


def test_compute_eda_insights_does_not_crash_on_no_ak_rows():
    rng = np.random.RandomState(1)
    n = 50
    age = rng.choice([1, 5, 12], n)
    price = np.clip(20_000 - age * 1000 + rng.normal(0, 100, n), 500, None)
    odometer = rng.uniform(1_000, 100_000, n)
    df = pd.DataFrame({
        "manufacturer": rng.choice(["ford", "toyota"], n),
        "model": rng.choice(["m1", "m2"], n),
        "price": price,
        "log_price": np.log1p(price),
        "age": age,
        "year": 2021 - age,
        "odometer": odometer,
        "mileage_per_year": odometer / age,
        "cylinders_num": rng.choice([4, 6], n).astype(float),
        "state": ["ca"] * n,  # no "ak" rows at all
        "VIN": [f"vin{i}" if i % 2 else None for i in range(n)],
        "condition": ["good" if i % 2 else None for i in range(n)],
        "drive": rng.choice(["4wd", "fwd"], n),
        "posting_date": pd.to_datetime(["2021-04-15"] * n, utc=True),
    })
    report = compute_eda_insights(df)  # must not raise
    assert "0%" in report or "%" in report
