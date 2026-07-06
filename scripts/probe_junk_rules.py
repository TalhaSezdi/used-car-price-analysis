"""Phase 7A probe: precision-check candidate junk/non-sale listing rules.

Context: the Phase 6C root cause + description recon (probe_description_signal.py)
found that several of the model's worst expensive-listing errors are not missing
trim signal but junk: non-vehicle items, "wanted to buy" ads, and commercial
equipment vehicles the consumer-marketplace price model was never meant to price.

This probe measures match count/share for three candidate rules and prints a
random sample of each for manual precision review -- by project protocol, no
rule enters the cleaning pipeline before its precision is checked here and
approved.

Read-only: does not modify data/processed/cleaned.parquet or any pipeline code.

Usage:
    python scripts/probe_junk_rules.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

import src.preprocess.cleaner as cleaner_mod
from src.features.engineer import FeatureEngineer

ROOT = Path(__file__).parents[1]
RAW_PATH = ROOT / "data" / "raw" / "vehicles.csv"

SAMPLE_N = 15
RANDOM_STATE = 42

# Rule 1: buyer / "wanted" ads, checked near the start of the text so a real
# sale ad mentioning e.g. "no lowball offers wanted" deep in the text is not
# mistakenly caught.
WANTED_PATTERN = re.compile(
    r"\b(?:looking to buy|want(?:ed)? to buy|we buy (?:cars|trucks|vehicles)|"
    r"cash for (?:your|junk) (?:car|cars|vehicle)|top dollar paid for)\b"
)
WANTED_HEAD_CHARS = 200

# Rule 2: non-vehicle listing vocabulary check (used only when model is
# missing -- conservative combination).
VEHICLE_VOCAB = re.compile(
    r"\b(?:miles|mileage|engine|transmission|title|vin|drive|drives|runs|"
    r"odometer|tires|brakes|automatic|manual|4x4|awd|fwd|rwd)\b"
)

# Rule 3: commercial-equipment vehicles -- not consumer-marketplace comparables.
COMMERCIAL_PATTERN = re.compile(
    r"\b(?:bucket truck|boom lift|forestry unit|altec|cutaway van|box truck|"
    r"reefer truck|step van|cherry picker|man lift|utility truck body)\b"
)


def rebuild_cleaned_with_text() -> pd.DataFrame:
    """Reproduce cleaned.parquet's exact rows, but keep id + description.

    Same technique as probe_description_signal.py: temporarily relax DROP_COLS
    so filtering logic (which never references these two columns) is
    unaffected, then restore it.
    """
    original_drop_cols = list(cleaner_mod.DROP_COLS)
    cleaner_mod.DROP_COLS = [c for c in original_drop_cols if c not in ("id", "description")]
    try:
        raw = pd.read_csv(RAW_PATH, low_memory=False)
        cleaner = cleaner_mod.DataCleaner()
        df = cleaner.fit_transform(raw)
    finally:
        cleaner_mod.DROP_COLS = original_drop_cols
    return FeatureEngineer().fit_transform(df)


def show_sample(df: pd.DataFrame, mask: np.ndarray, label: str) -> None:
    n = int(mask.sum())
    print(f"\n=== {label}: {n:,} matches ({n/len(df)*100:.3f}% of {len(df):,}) ===")
    if n == 0:
        return
    idx = np.where(mask)[0]
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        idx, size=min(SAMPLE_N, len(idx)), replace=False
    )
    sample = df.iloc[sample_idx]
    for _, row in sample.iterrows():
        car = f"{row.get('year', '?')} {row.get('manufacturer', '?')} {row.get('model', '?')}"
        price = row.get("price", float("nan"))
        text = str(row.get("description", ""))[:200].encode("ascii", "replace").decode("ascii")
        print(f"  [{car}] price=${price:,.0f} -- {text!r}")


def main() -> None:
    df = rebuild_cleaned_with_text()
    print(f"Base population: {len(df):,} rows (matches cleaned.parquet)")

    text = df["description"].fillna("")
    text_lower = text.str.lower()
    head_lower = text_lower.str.slice(0, WANTED_HEAD_CHARS)

    # --- Rule 1: wanted / buyer ads ---
    mask_wanted = head_lower.str.contains(WANTED_PATTERN).values
    show_sample(df, mask_wanted, "Rule 1: wanted/buyer ads (checked in first 200 chars)")

    # --- Rule 2: non-vehicle items (model missing AND no vehicle vocab) ---
    model_missing = df["model"].isna().values
    has_vehicle_vocab = text_lower.str.contains(VEHICLE_VOCAB).values
    mask_nonvehicle = model_missing & ~has_vehicle_vocab
    show_sample(df, mask_nonvehicle, "Rule 2: non-vehicle items (model missing, no vehicle vocab)")

    # --- Rule 3: commercial equipment vehicles ---
    mask_commercial = text_lower.str.contains(COMMERCIAL_PATTERN).values
    show_sample(df, mask_commercial, "Rule 3: commercial equipment vehicles")

    # --- Overlap check ---
    print("\n=== Overlap between rules ===")
    print(f"Rule1 & Rule2: {int((mask_wanted & mask_nonvehicle).sum())}")
    print(f"Rule1 & Rule3: {int((mask_wanted & mask_commercial).sum())}")
    print(f"Rule2 & Rule3: {int((mask_nonvehicle & mask_commercial).sum())}")
    total_flagged = mask_wanted | mask_nonvehicle | mask_commercial
    print(f"Total flagged (any rule): {int(total_flagged.sum()):,} "
          f"({total_flagged.mean()*100:.3f}% of {len(df):,})")

    # --- Price distribution of flagged vs unflagged, for context ---
    print("\n=== Price stats: flagged vs unflagged ===")
    print("Flagged  :", df.loc[total_flagged, "price"].describe().round(0).to_dict())
    print("Unflagged:", df.loc[~total_flagged, "price"].describe().round(0).to_dict())

    # --- Does this filter even reach the 6C problem segment? ---
    expensive_50_150k = ((df["price"] >= 50_000) & (df["price"] <= 150_000)).values
    n_expensive = int(expensive_50_150k.sum())
    n_expensive_flagged = int((expensive_50_150k & total_flagged).sum())
    print(f"\n=== Reach into the 6C problem segment (price 50-150k) ===")
    print(f"50-150k listings total     : {n_expensive:,}")
    print(f"50-150k listings flagged   : {n_expensive_flagged:,} "
          f"({n_expensive_flagged / max(n_expensive, 1) * 100:.2f}% of that segment)")


if __name__ == "__main__":
    main()
