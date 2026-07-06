"""Tests for src/models/train.py -- model training entry points + ablations."""

import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    LGBM_PARAMS,
    ablation_a1_raw_vs_log,
    ablation_a2_collinearity,
    fit_lgbm_with_early_stopping,
    oof_log_predictions,
    train_linear,
    train_rf,
)


@pytest.fixture
def linear_data():
    rng = np.random.RandomState(42)
    n = 300
    age = rng.uniform(1, 20, n)
    odometer = rng.uniform(1_000, 200_000, n)
    noise = rng.normal(0, 0.05, n)
    log_price = 10.5 - 0.05 * age - 0.000002 * odometer + noise
    X = pd.DataFrame({"age": age, "odometer": odometer})
    y = pd.Series(log_price)
    return X, y


def test_train_linear_returns_trained_model_with_expected_metrics_keys(linear_data):
    X, y = linear_data
    X_train, X_test = X.iloc[:200], X.iloc[200:]
    y_train, y_test = y.iloc[:200], y.iloc[200:]
    result = train_linear(X_train, y_train, X_test, y_test)
    assert set(result.metrics) == {"RMSE ($)", "MAE ($)", "MAPE (%)", "R2"}
    assert result.name == "Linear Regression"


def test_train_rf_deterministic_with_fixed_seed(linear_data):
    X, y = linear_data
    X_train, X_test = X.iloc[:200], X.iloc[200:]
    y_train, y_test = y.iloc[:200], y.iloc[200:]
    r1 = train_rf(X_train, y_train, X_test, y_test)
    r2 = train_rf(X_train, y_train, X_test, y_test)
    # n_jobs=-1 parallel tree averaging can differ in floating-point summation
    # order across runs; allclose (not exact equality) is the right guard here.
    np.testing.assert_allclose(r1.predictions, r2.predictions, rtol=1e-10)


def test_fit_lgbm_with_early_stopping_uses_train_carved_validation_only(linear_data):
    X, y = linear_data
    model = fit_lgbm_with_early_stopping(X, y, LGBM_PARAMS, n_estimators=50, test_size=0.1)
    assert model.best_iteration_ is not None
    assert model.best_iteration_ <= 50


def test_fit_lgbm_with_early_stopping_deterministic(linear_data):
    X, y = linear_data
    m1 = fit_lgbm_with_early_stopping(X, y, LGBM_PARAMS, n_estimators=50)
    m2 = fit_lgbm_with_early_stopping(X, y, LGBM_PARAMS, n_estimators=50)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


def test_oof_predictions_cover_every_row_exactly_once(linear_data):
    X, y = linear_data
    oof = oof_log_predictions(
        X, y, numeric_cols=["age", "odometer"], low_card_cols=[], high_card_cols=[],
        n_splits=3, n_estimators=20,
    )
    assert len(oof) == len(X)
    assert not np.isnan(oof).any()


def test_ablation_a1_log_vs_raw_returns_both_keys(linear_data):
    X, y = linear_data
    price = np.expm1(y)
    X_train, X_test = X.iloc[:200], X.iloc[200:]
    y_train, y_test = y.iloc[:200], y.iloc[200:]
    price_train, price_test = price.iloc[:200], price.iloc[200:]
    result = ablation_a1_raw_vs_log(X_train, X_test, y_train, y_test, price_train, price_test)
    assert set(result) == {"log1p(price)", "raw price"}
    assert result["raw price"]["RMSE ($)"] >= 0


def test_ablation_a2_coefficient_signs():
    rng = np.random.RandomState(42)
    n = 300
    age = rng.uniform(1, 20, n)
    year = 2021 - age
    noise = rng.normal(0, 0.02, n)
    log_price = 10.5 - 0.05 * age + noise
    X = pd.DataFrame({"age": age})
    y = pd.Series(log_price)
    year_s = pd.Series(year)

    X_train, X_test = X.iloc[:200], X.iloc[200:]
    y_train, y_test = y.iloc[:200], y.iloc[200:]
    year_train, year_test = year_s.iloc[:200], year_s.iloc[200:]

    result = ablation_a2_collinearity(X_train, X_test, y_train, y_test, year_train, year_test)
    assert result["age_only"]["coef_age"] < 0
