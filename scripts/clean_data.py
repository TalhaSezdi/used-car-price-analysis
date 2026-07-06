"""Entry point: run the full cleaning + feature engineering pipeline.

Usage:
    python scripts/clean_data.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.preprocess.cleaner import DataCleaner
from src.features.engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_PATH = Path(__file__).parents[1] / "data" / "raw" / "vehicles.csv"
OUT_PATH = Path(__file__).parents[1] / "data" / "processed" / "cleaned.parquet"


def main() -> None:
    logger.info("Loading raw data from %s", RAW_PATH)
    df = pd.read_csv(RAW_PATH, low_memory=False)
    logger.info("Raw shape: %s", df.shape)

    cleaner = DataCleaner()
    df = cleaner.fit_transform(df)

    print("\n=== Cleaning Report ===")
    print(cleaner.report)

    engineer = FeatureEngineer()
    df = engineer.fit_transform(df)

    print(f"\n=== Final Dataset ===")
    print(f"Shape      : {df.shape}")
    print(f"Columns    : {list(df.columns)}")
    print(f"\nPrice stats (original):")
    print(df["price"].describe().to_string())
    print(f"\nAge stats:")
    print(df["age"].describe().to_string())
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info("Saved cleaned dataset to %s", OUT_PATH)


if __name__ == "__main__":
    main()
