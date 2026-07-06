"""Tests for src/features/description.py -- leakage-free description features."""

import numpy as np
import pandas as pd

from src.features.description import DescriptionFeatureExtractor


def test_trim_keyword_word_boundary():
    df = pd.DataFrame({"description": ["nice denali edition", "the denalisystem is great"]})
    out = DescriptionFeatureExtractor().transform(df)
    assert out["desc_trim_luxury"].iloc[0] == 1
    assert out["desc_trim_luxury"].iloc[1] == 0  # "denalisystem" is not a word-boundary match


def test_equipment_count_sums_multiple_keywords():
    df = pd.DataFrame({"description": ["has leather seats and a sunroof"]})
    out = DescriptionFeatureExtractor().transform(df)
    assert out["desc_equip_count"].iloc[0] == 2


def test_case_insensitive_matching():
    df = pd.DataFrame({"description": ["DENALI edition, LEATHER interior"]})
    out = DescriptionFeatureExtractor().transform(df)
    assert out["desc_trim_luxury"].iloc[0] == 1
    assert out["desc_equip_count"].iloc[0] == 1


def test_missing_description_treated_as_empty():
    df = pd.DataFrame({"description": [None]})
    out = DescriptionFeatureExtractor().transform(df)
    assert out["desc_trim_luxury"].iloc[0] == 0
    assert out["desc_equip_count"].iloc[0] == 0
    assert out["desc_len_log"].iloc[0] == np.log1p(0)


def test_no_numeric_leakage_in_features():
    df = pd.DataFrame({"description": ["selling for $15000 firm, denali trim"]})
    out = DescriptionFeatureExtractor().transform(df)
    # desc_trim_luxury/desc_equip_count are keyword-only; desc_len_log is a
    # length count -- none of them are derived from the $15000 figure itself.
    assert out["desc_trim_luxury"].iloc[0] == 1
    assert out["desc_len_log"].iloc[0] == np.log1p(len("selling for $15000 firm, denali trim"))


def test_fit_transform_is_alias_for_transform():
    df = pd.DataFrame({"description": ["denali"]})
    extractor = DescriptionFeatureExtractor()
    a = extractor.transform(df)
    b = extractor.fit_transform(df)
    pd.testing.assert_frame_equal(a, b)
