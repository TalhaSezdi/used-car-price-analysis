"""Data cleaning pipeline for the used-car Craigslist dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


PRICE_MIN: float = 500.0
PRICE_MAX: float = 150_000.0
YEAR_MIN: int = 1970
YEAR_MAX: int = 2022
ODOMETER_MIN: float = 1.0
ODOMETER_MAX: float = 500_000.0
VALID_TITLE_STATUS: set[str] = {"clean", "rebuilt"}

DROP_COLS: list[str] = [
    "id", "url", "region_url", "image_url", "county",
]
# NOTE: "description" is intentionally NOT dropped here. It is kept through
# cleaning/dedup (neither step references it) so FeatureEngineer can extract
# leakage-free trim/equipment keyword features (Phase 7B) before the raw text
# itself is dropped -- the processed parquet never contains raw description.
CORE_COLS: list[str] = ["price", "year", "odometer", "manufacturer"]

CAST_STRIP_LOWER_COLS: list[str] = [
    "manufacturer", "model", "condition", "cylinders", "fuel",
    "title_status", "transmission", "drive", "size", "type",
    "paint_color", "state", "region",
]


@dataclass
class CleaningReport:
    """Row counts after each cleaning stage, plus free-text notes.

    Attributes:
        initial_rows: Row count before any cleaning.
        rows_after_drop_cols: Row count after dropping non-predictive columns
            (unaffected by column drops, tracked for pipeline visibility).
        rows_after_price_filter: Row count after the price bound filter.
        rows_after_year_filter: Row count after the year bound filter.
        rows_after_odometer_filter: Row count after the odometer bound filter.
        rows_after_title_filter: Row count after the title-status filter.
        rows_after_dedup: Row count after VIN + fingerprint deduplication.
        rows_after_core_nulls: Row count after dropping null core columns
            (the final row count).
        notes: Human-readable notes recorded per stage.
    """
    initial_rows: int = 0
    rows_after_drop_cols: int = 0
    rows_after_price_filter: int = 0
    rows_after_year_filter: int = 0
    rows_after_odometer_filter: int = 0
    rows_after_title_filter: int = 0
    rows_after_dedup: int = 0
    rows_after_core_nulls: int = 0
    notes: list[str] = field(default_factory=list)

    def retention_pct(self) -> float:
        """Compute the final row count as a percentage of the initial count.

        Returns:
            float: Retention percentage, or 0.0 if initial_rows is 0.
        """
        if self.initial_rows == 0:
            return 0.0
        return round(self.rows_after_core_nulls / self.initial_rows * 100, 2)

    def __str__(self) -> str:
        lines = [
            f"Initial rows          : {self.initial_rows:>10,}",
            f"After drop columns    : {self.rows_after_drop_cols:>10,}",
            f"After price filter    : {self.rows_after_price_filter:>10,}",
            f"After year filter     : {self.rows_after_year_filter:>10,}",
            f"After odometer filter : {self.rows_after_odometer_filter:>10,}",
            f"After title filter    : {self.rows_after_title_filter:>10,}",
            f"After deduplication  : {self.rows_after_dedup:>10,}",
            f"After core null drop  : {self.rows_after_core_nulls:>10,}",
            f"Retention             : {self.retention_pct():>9}%",
        ]
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


class DataCleaner:
    """Applies deterministic cleaning rules to the raw vehicles CSV.

    Every rule (price/year/odometer bounds, which columns get dropped, which
    columns are "core" and required non-null, valid title statuses, and which
    string columns get stripped/lowercased) is a constructor parameter so this
    class works against a structurally different dataset without editing
    source. Defaults reproduce the exact cleaning behavior documented in
    docs/cleaning_pipeline.md and docs/phase1_audit.md.

    Args:
        price_min: Minimum valid price (inclusive).
        price_max: Maximum valid price (inclusive).
        year_min: Minimum valid model year (inclusive).
        year_max: Maximum valid model year (inclusive).
        odometer_min: Minimum valid odometer reading (inclusive).
        odometer_max: Maximum valid odometer reading (inclusive).
        valid_title_status: Title statuses to keep (plus null, always kept).
            Defaults to a copy of VALID_TITLE_STATUS if None.
        drop_cols: Non-predictive columns to drop before cleaning. Defaults
            to a copy of DROP_COLS if None.
        core_cols: Columns that must be non-null after cleaning; rows with a
            null in any of these are dropped. Defaults to a copy of CORE_COLS
            if None.
        cast_strip_lower_cols: String columns to strip and lowercase during
            type casting. Defaults to a copy of CAST_STRIP_LOWER_COLS if None.
    """

    def __init__(
        self,
        price_min: float = PRICE_MIN,
        price_max: float = PRICE_MAX,
        year_min: int = YEAR_MIN,
        year_max: int = YEAR_MAX,
        odometer_min: float = ODOMETER_MIN,
        odometer_max: float = ODOMETER_MAX,
        valid_title_status: set[str] | None = None,
        drop_cols: list[str] | None = None,
        core_cols: list[str] | None = None,
        cast_strip_lower_cols: list[str] | None = None,
    ) -> None:
        self.price_min = price_min
        self.price_max = price_max
        self.year_min = year_min
        self.year_max = year_max
        self.odometer_min = odometer_min
        self.odometer_max = odometer_max
        self.valid_title_status = (
            valid_title_status if valid_title_status is not None else set(VALID_TITLE_STATUS)
        )
        self.drop_cols = drop_cols if drop_cols is not None else list(DROP_COLS)
        self.core_cols = core_cols if core_cols is not None else list(CORE_COLS)
        self.cast_strip_lower_cols = (
            cast_strip_lower_cols if cast_strip_lower_cols is not None else list(CAST_STRIP_LOWER_COLS)
        )
        self.report = CleaningReport()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline and return the cleaned DataFrame.

        Args:
            df: Raw (or partially processed) input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame with a fresh RangeIndex.
        """
        self.report = CleaningReport(initial_rows=len(df))
        df = df.copy()
        df = self._drop_cols(df)
        df = self._cast_types(df)
        df = self._filter_price(df)
        df = self._filter_year(df)
        df = self._filter_odometer(df)
        df = self._filter_title_status(df)
        df = self._deduplicate(df)
        df = self._drop_core_nulls(df)
        df = df.reset_index(drop=True)
        logger.info("\n%s", self.report)
        return df

    def _drop_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop non-predictive columns present in `self.drop_cols`.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: `df` without the dropped columns.
        """
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        self.report.rows_after_drop_cols = len(df)
        self.report.notes.append(f"Dropped columns: {cols_to_drop}")
        return df

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce numeric/date columns and strip+lowercase string columns.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: `df` with types cast; unparseable values become NaN.
        """
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        if "posting_date" in df.columns:
            df["posting_date"] = pd.to_datetime(df["posting_date"], format="ISO8601", errors="coerce", utc=True)
        for col in self.cast_strip_lower_cols:
            if col in df.columns:
                df[col] = df[col].str.strip().str.lower()
        return df

    def _filter_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep rows with `price` in [price_min, price_max] (inclusive).

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        before = len(df)
        mask = df["price"].between(self.price_min, self.price_max)
        df = df[mask]
        self.report.rows_after_price_filter = len(df)
        self.report.notes.append(
            f"Price filter [{self.price_min}, {self.price_max}]: removed {before - len(df):,} rows"
        )
        return df

    def _filter_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep rows with `year` in [year_min, year_max] (inclusive).

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        before = len(df)
        mask = df["year"].between(self.year_min, self.year_max)
        df = df[mask]
        self.report.rows_after_year_filter = len(df)
        self.report.notes.append(
            f"Year filter [{self.year_min}, {self.year_max}]: removed {before - len(df):,} rows"
        )
        return df

    def _filter_odometer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep rows with `odometer` in [odometer_min, odometer_max] (inclusive).

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        before = len(df)
        mask = df["odometer"].between(self.odometer_min, self.odometer_max)
        df = df[mask]
        self.report.rows_after_odometer_filter = len(df)
        self.report.notes.append(
            f"Odometer filter [{self.odometer_min}, {self.odometer_max}]: removed {before - len(df):,} rows"
        )
        return df

    def _filter_title_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep rows with a valid title status, or a null title status.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame (unchanged if no `title_status`
            column is present).
        """
        if "title_status" not in df.columns:
            self.report.rows_after_title_filter = len(df)
            return df
        before = len(df)
        mask = df["title_status"].isin(self.valid_title_status) | df["title_status"].isnull()
        df = df[mask]
        self.report.rows_after_title_filter = len(df)
        self.report.notes.append(
            f"Title status filter (keep {self.valid_title_status} + null): removed {before - len(df):,} rows"
        )
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Two-stage dedup.

        1. VIN is the true unique vehicle id -> drop exact VIN duplicates.
        2. Fingerprint dedup is applied ONLY to rows WITHOUT a VIN. VIN'd rows
           are already correctly deduped in step 1; a shared
           (make, model, price, odometer) across two DIFFERENT VINs means two
           distinct cars, so we must not collapse them.
        """
        before = len(df)
        has_vin = pd.Series(False, index=df.index)
        if "VIN" in df.columns:
            has_vin = df["VIN"].notna() & (df["VIN"].str.strip() != "")
            vin_dupes = df[has_vin].duplicated(subset=["VIN"], keep="first")
            df = df[~(has_vin & vin_dupes)]
            has_vin = has_vin.loc[df.index]
        removed_by_vin = before - len(df)

        fingerprint_cols = [c for c in ["manufacturer", "model", "price", "odometer"] if c in df.columns]
        if fingerprint_cols:
            no_vin = df[~has_vin]
            vin_rows = df[has_vin]
            no_vin = no_vin.drop_duplicates(subset=fingerprint_cols, keep="first")
            df = pd.concat([vin_rows, no_vin]).sort_index()
        removed_by_fp = before - removed_by_vin - len(df)

        self.report.rows_after_dedup = len(df)
        self.report.notes.append(
            f"Deduplication: removed {before - len(df):,} rows "
            f"(VIN exact: {removed_by_vin:,}; no-VIN fingerprint: {removed_by_fp:,})"
        )
        return df

    def _drop_core_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with a null in any of `self.core_cols`.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        before = len(df)
        core = [c for c in self.core_cols if c in df.columns]
        df = df.dropna(subset=core)
        self.report.rows_after_core_nulls = len(df)
        self.report.notes.append(
            f"Dropped null core cols {core}: removed {before - len(df):,} rows"
        )
        return df
