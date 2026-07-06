"""Tests for src/models/encoders.py -- leakage-safe encoding and imputation."""

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_STATE
from src.models.encoders import FeaturePreprocessor, FrequencyEncoder, SafeTargetEncoder


def test_safe_target_encoder_random_state_is_constructor_param():
    enc = SafeTargetEncoder(cols=["model"])
    assert enc.random_state == RANDOM_STATE
    enc2 = SafeTargetEncoder(cols=["model"], random_state=7)
    assert enc2.random_state == 7


def test_safe_target_encoder_oof_no_self_leakage():
    # Leave-one-out (n_folds == n_rows): every row is alone in its own
    # validation fold, so its OOF-encoded value can ONLY come from the other
    # rows' targets -- this lets us hand-compute the exact expected value and
    # prove the row's own y never contributes to its own encoding.
    y_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    X = pd.DataFrame({"cat": ["a"] * len(y_values)})
    y = pd.Series(y_values)

    enc = SafeTargetEncoder(cols=["cat"], n_folds=len(y_values), smoothing=0)
    oof = enc.fit_transform(X, y)

    total = sum(y_values)
    expected = [(total - v) / (len(y_values) - 1) for v in y_values]
    assert np.allclose(oof["cat"].values, expected)
    # Each row's OOF value must differ from its own y (no self-leakage).
    assert all(abs(o - v) > 1e-9 for o, v in zip(oof["cat"].values, y_values))


def test_safe_target_encoder_smoothing_shrinks_rare_categories():
    X = pd.DataFrame({"cat": ["rare"] + ["common"] * 20})
    y = pd.Series([1000.0] + [100.0] * 20)

    enc = SafeTargetEncoder(cols=["cat"], smoothing=20)
    enc.fit(X, y)
    global_mean = float(y.mean())
    rare_mean = enc.mapping_["cat"]["rare"]
    assert abs(rare_mean - global_mean) < abs(1000.0 - global_mean)


def test_safe_target_encoder_transform_unseen_category_falls_back_to_global_mean():
    X_train = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    y_train = pd.Series([10.0, 20.0, 10.0, 20.0])
    enc = SafeTargetEncoder(cols=["cat"])
    enc.fit(X_train, y_train)

    X_test = pd.DataFrame({"cat": ["c"]})  # unseen at fit time
    out = enc.transform(X_test)
    assert out["cat"].iloc[0] == pytest.approx(enc.global_mean_)


def test_frequency_encoder_matches_value_counts_normalize():
    X = pd.DataFrame({"cat": ["a", "a", "b", "c", "c", "c"]})
    enc = FrequencyEncoder(cols=["cat"])
    enc.fit(X)
    out = enc.transform(X)
    expected = X["cat"].map(X["cat"].value_counts(normalize=True))
    assert np.allclose(out["cat"].values, expected.values)


def test_frequency_encoder_unseen_category_is_zero():
    X_train = pd.DataFrame({"cat": ["a", "a", "b"]})
    enc = FrequencyEncoder(cols=["cat"])
    enc.fit(X_train)
    out = enc.transform(pd.DataFrame({"cat": ["z"]}))
    assert out["cat"].iloc[0] == 0.0


@pytest.fixture
def preprocessor_data():
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "age": rng.uniform(1, 20, n),
        "odometer": rng.uniform(1_000, 200_000, n),
        "model": rng.choice([f"m{i}" for i in range(10)], n),
        "condition": rng.choice(["good", "fair", "excellent"], n),
    })
    X.loc[X.sample(frac=0.1, random_state=1).index, "age"] = np.nan
    y = pd.Series(np.log1p(rng.uniform(1000, 50000, n)))
    return X, y


def test_feature_preprocessor_train_columns_stable_on_transform(preprocessor_data):
    X, y = preprocessor_data
    X_train, X_test = X.iloc[:150].copy(), X.iloc[150:].copy()
    y_train = y.iloc[:150]

    # Force a low-card category present in train but absent from test.
    X_train.loc[X_train.index[0], "condition"] = "salvage"
    X_test = X_test[X_test["condition"] != "salvage"]

    prep = FeaturePreprocessor(
        numeric_cols=["age", "odometer"], low_card_cols=["condition"], high_card_cols=["model"],
    )
    Xt_train = prep.fit_transform(X_train, y_train)
    Xt_test = prep.transform(X_test)
    assert list(Xt_test.columns) == list(Xt_train.columns)
    assert (Xt_test["condition_salvage"] == 0).all()


def test_feature_preprocessor_missing_indicator_created_only_when_nulls_present(preprocessor_data):
    X, y = preprocessor_data
    prep = FeaturePreprocessor(
        numeric_cols=["age", "odometer"], low_card_cols=["condition"], high_card_cols=["model"],
    )
    out = prep.fit_transform(X, y)
    assert out["age_missing"].sum() > 0
    assert (out["odometer_missing"] == 0).all()


def test_feature_preprocessor_frequency_vs_target_method_differ(preprocessor_data):
    X, y = preprocessor_data
    prep_target = FeaturePreprocessor(
        numeric_cols=["age", "odometer"], low_card_cols=["condition"], high_card_cols=["model"],
        high_card_method="target",
    )
    prep_freq = FeaturePreprocessor(
        numeric_cols=["age", "odometer"], low_card_cols=["condition"], high_card_cols=["model"],
        high_card_method="frequency",
    )
    out_target = prep_target.fit_transform(X, y)
    out_freq = prep_freq.fit_transform(X, y)
    assert not np.allclose(out_target["model"].values, out_freq["model"].values)
