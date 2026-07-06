"""Regression tests for src/preprocess/cleaner.py.

These pin the exact current cleaning behavior (price/year/odometer bounds,
dedup rules, title-status handling, core-null drop) so that parametrizing
DROP_COLS/CORE_COLS/cast_strip_lower_cols into constructor arguments (Phase 8)
provably did not change any default outcome.
"""

import pandas as pd
import pytest

from src.preprocess.cleaner import CORE_COLS, DROP_COLS, CleaningReport, DataCleaner


def _base_row(**overrides):
    # "id" is in DROP_COLS by default (dropped before the caller can inspect
    # it), so "row_tag" -- not in DROP_COLS/CORE_COLS/cast_strip_lower_cols --
    # is the column tests use to identify which input rows survived.
    row = {
        "id": 1, "row_tag": overrides.get("id", 1),
        "url": "u", "region_url": "r", "image_url": "i", "county": None,
        "price": 10_000.0, "year": 2015, "odometer": 50_000.0,
        "manufacturer": "Ford", "model": "F-150", "condition": "good",
        "cylinders": "6 cylinders", "fuel": "gas", "title_status": "clean",
        "transmission": "automatic", "drive": "4wd", "size": "full-size",
        "type": "truck", "paint_color": "black", "state": "CA", "region": "sf bay area",
        "VIN": None, "description": "a nice truck",
    }
    row.update(overrides)
    return row


def test_filter_price_bounds():
    df = pd.DataFrame([
        _base_row(id=1, price=100, VIN=f"V1"),      # below min (500) -> dropped
        _base_row(id=2, price=500, VIN=f"V2"),      # at min -> kept
        _base_row(id=3, price=150_000, VIN=f"V3"),  # at max -> kept
        _base_row(id=4, price=150_001, VIN=f"V4"),  # above max -> dropped
        _base_row(id=5, price=80_000, VIN=f"V5"),   # mid -> kept
    ])
    cleaner = DataCleaner()
    out = cleaner.fit_transform(df)
    assert sorted(out["row_tag"].tolist()) == [2, 3, 5]


def test_filter_year_bounds():
    df = pd.DataFrame([
        _base_row(id=1, year=1969, VIN="V1"),  # below min -> dropped
        _base_row(id=2, year=1970, VIN="V2"),  # at min -> kept
        _base_row(id=3, year=2022, VIN="V3"),  # at max -> kept
        _base_row(id=4, year=2023, VIN="V4"),  # above max -> dropped
    ])
    cleaner = DataCleaner()
    out = cleaner.fit_transform(df)
    assert sorted(out["row_tag"].tolist()) == [2, 3]


def test_filter_odometer_bounds():
    df = pd.DataFrame([
        _base_row(id=1, odometer=0, VIN="V1"),        # below min (1) -> dropped
        _base_row(id=2, odometer=1, VIN="V2"),        # at min -> kept
        _base_row(id=3, odometer=500_000, VIN="V3"),  # at max -> kept
        _base_row(id=4, odometer=500_001, VIN="V4"),  # above max -> dropped
    ])
    cleaner = DataCleaner()
    out = cleaner.fit_transform(df)
    assert sorted(out["row_tag"].tolist()) == [2, 3]


def test_dedup_vin_exact():
    df = pd.DataFrame([
        _base_row(id=1, VIN="SAMEVIN"),
        _base_row(id=2, VIN="SAMEVIN", model="Different Model"),
    ])
    cleaner = DataCleaner()
    out = cleaner.fit_transform(df)
    assert len(out) == 1
    assert out["row_tag"].iloc[0] == 1  # keep="first"


def test_dedup_fingerprint_no_vin_only():
    df = pd.DataFrame([
        # Two DIFFERENT VINs sharing a fingerprint -> both kept (distinct cars).
        _base_row(id=1, VIN="VIN_A", manufacturer="ford", model="f-150", price=10_000, odometer=50_000),
        _base_row(id=2, VIN="VIN_B", manufacturer="ford", model="f-150", price=10_000, odometer=50_000),
        # Two no-VIN rows sharing a fingerprint -> collapsed to one.
        _base_row(id=3, VIN=None, manufacturer="toyota", model="camry", price=15_000, odometer=30_000),
        _base_row(id=4, VIN=None, manufacturer="toyota", model="camry", price=15_000, odometer=30_000),
    ])
    cleaner = DataCleaner()
    out = cleaner.fit_transform(df)
    assert sorted(out["row_tag"].tolist()) == [1, 2, 3]


def test_title_status_keeps_null_and_valid_drops_invalid():
    df = pd.DataFrame([
        _base_row(id=1, title_status="clean", VIN="V1"),
        _base_row(id=2, title_status="rebuilt", VIN="V2"),
        _base_row(id=3, title_status="salvage", VIN="V3"),
        _base_row(id=4, title_status=None, VIN="V4"),
    ])
    cleaner = DataCleaner()
    out = cleaner.fit_transform(df)
    assert sorted(out["row_tag"].tolist()) == [1, 2, 4]


def test_core_nulls_dropped():
    df = pd.DataFrame([
        _base_row(id=1, manufacturer="ford", VIN="V1"),
        _base_row(id=2, manufacturer=None, VIN="V2"),
    ])
    cleaner = DataCleaner()
    out = cleaner.fit_transform(df)
    assert out["row_tag"].tolist() == [1]


def test_cleaning_report_retention_pct_zero_rows():
    report = CleaningReport(initial_rows=0)
    assert report.retention_pct() == 0.0


def test_drop_cols_and_core_cols_default_to_module_constants():
    cleaner = DataCleaner()
    assert cleaner.drop_cols == DROP_COLS
    assert cleaner.core_cols == CORE_COLS
    assert cleaner.drop_cols is not DROP_COLS  # copy, not the same list object


def test_drop_cols_overridable():
    df = pd.DataFrame([_base_row(id=1, VIN="V1")])
    cleaner = DataCleaner(drop_cols=["id"])
    out = cleaner.fit_transform(df)
    assert "id" not in out.columns
    assert "url" in out.columns  # not in the overridden drop_cols, so kept


def test_valid_title_status_not_shared_mutable_default():
    c1 = DataCleaner()
    c2 = DataCleaner()
    c1.valid_title_status.add("mutated")
    assert "mutated" not in c2.valid_title_status
