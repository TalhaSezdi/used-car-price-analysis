"""Regression tests for src/features/engineer.py.

test_add_age_uses_reference_year_not_current_date is the guard against
CLAUDE.md's named "reference date trap": vehicle age must be computed
relative to the 2021 posting year, never `datetime.now().year`. It pins an
exact expected age so a regression to real-world-date-based age computation
fails loudly, regardless of what year this test happens to run in.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import AGE_MIN, REFERENCE_YEAR, FeatureEngineer


def _base_df(**overrides):
    df = pd.DataFrame({
        "year": [2015],
        "odometer": [50_000.0],
        "price": [10_000.0],
        "cylinders": ["6 cylinders"],
    })
    for col, val in overrides.items():
        df[col] = val
    return df


def test_add_age_uses_reference_year_not_current_date():
    df = _base_df(year=[2015])  # no posting_date column at all
    out = FeatureEngineer().fit_transform(df)
    assert out["age"].iloc[0] == REFERENCE_YEAR - 2015  # == 6, pinned exactly
    assert REFERENCE_YEAR == 2021  # the documented reference-date-trap constant


def test_add_age_uses_posting_date_year_when_present():
    df = _base_df(year=[2015])
    df["posting_date"] = pd.to_datetime(["2019-03-01"], utc=True)
    out = FeatureEngineer().fit_transform(df)
    assert out["age"].iloc[0] == 2019 - 2015


def test_add_age_clips_negative_to_age_min():
    df = _base_df(year=[2025])  # year > REFERENCE_YEAR -- data-entry error past cleaning
    out = FeatureEngineer().fit_transform(df)
    assert out["age"].iloc[0] == AGE_MIN


def test_mileage_per_year_division():
    df = _base_df(year=[2015], odometer=[60_000.0])
    out = FeatureEngineer().fit_transform(df)
    expected_age = REFERENCE_YEAR - 2015
    assert out["mileage_per_year"].iloc[0] == pytest.approx(60_000.0 / expected_age)


def test_cylinders_extraction_from_string():
    df = _base_df(cylinders=["6 cylinders"])
    out = FeatureEngineer().fit_transform(df)
    assert out["cylinders_num"].iloc[0] == 6.0


def test_cylinders_extraction_handles_other_and_nan():
    df = _base_df(cylinders=["other"])
    out = FeatureEngineer().fit_transform(df)
    assert np.isnan(out["cylinders_num"].iloc[0])


def test_description_features_dropped_after_extraction():
    df = _base_df()
    df["description"] = ["leather seats and a sunroof"]
    out = FeatureEngineer().fit_transform(df)
    assert "description" not in out.columns
    assert "desc_equip_count" in out.columns


def test_fit_transform_missing_description_column_is_noop():
    df = _base_df()
    assert "description" not in df.columns
    out = FeatureEngineer().fit_transform(df)  # must not raise
    assert "desc_equip_count" not in out.columns


def test_reference_year_and_age_min_are_constructor_parameters():
    engineer = FeatureEngineer(reference_year=2019, age_min=2)
    df = _base_df(year=[2015])
    out = engineer.fit_transform(df)
    assert out["age"].iloc[0] == 2019 - 2015


def test_custom_age_min_clips_correctly():
    engineer = FeatureEngineer(reference_year=2021, age_min=2)
    df = _base_df(year=[2021])  # would naturally be age 0
    out = engineer.fit_transform(df)
    assert out["age"].iloc[0] == 2


def test_defaults_match_documented_reference_date_trap_constants():
    engineer = FeatureEngineer()
    assert engineer.reference_year == 2021
    assert engineer.age_min == 1
