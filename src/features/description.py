"""Leakage-free features extracted from the free-text `description` field.

Design constraints (Phase 7B, per CLAUDE.md's leakage rule on `description`):
  - NO numeric extraction of any kind (prices, MSRP, stock numbers, phone
    numbers stay out by construction) -- alphabetic keyword matching only.
  - Keyword lists are fixed a priori from domain knowledge, never derived from
    target statistics -- there is no fitting step, so there is no way for
    these features to leak the label.
  - Purely row-wise and stateless: no cross-row statistics, so this runs once
    at cleaning time, before any train/test split exists, with identical
    behavior for every future row (including at inference).

Motivation: Phase 6C found a ~16.5% within-(manufacturer, model, year)-group
price premium for listings mentioning a trim/equipment keyword -- signal the
existing structured features (which only encode make/model/year/odometer)
cannot see. Whether this actually improves the model is decided by Ablation A4
(scripts/ablation_description_features.py), not assumed here.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Curated trim/package names (word-boundary matched, case-insensitive).
TRIM_KEYWORDS: list[str] = [
    "denali", "platinum", "king ranch", "laramie", "lariat", "limited",
    "overland", "rubicon", "trailhawk", "trd", "z71", "big horn", "sahara",
    "touring", "premier", "reserve", "signature", "ultimate", "sr5", "xlt",
    "sport edition", "anniversary edition",
]

# Curated equipment/feature mentions.
EQUIPMENT_KEYWORDS: list[str] = [
    "leather", "sunroof", "moonroof", "navigation", "heated seats",
    "tow package", "backup camera", "bluetooth", "third row",
    "remote start", "panoramic", "premium sound", "keyless entry",
]

_TRIM_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in TRIM_KEYWORDS) + r")\b"
)
_EQUIP_PATTERNS = [
    re.compile(r"\b" + re.escape(k) + r"\b") for k in EQUIPMENT_KEYWORDS
]


class DescriptionFeatureExtractor:
    """Stateless row-wise transform: same output at fit time and inference time."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        text = df["description"].fillna("").str.lower()

        df["desc_trim_luxury"] = text.str.contains(_TRIM_PATTERN).astype(int)
        df["desc_equip_count"] = np.sum(
            [text.str.contains(p).astype(int).values for p in _EQUIP_PATTERNS],
            axis=0,
        )
        df["desc_len_log"] = np.log1p(df["description"].fillna("").str.len())
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)
