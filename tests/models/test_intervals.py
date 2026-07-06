"""Tests for src/models/intervals.py -- conformal prediction intervals."""

import numpy as np
import pandas as pd
import pytest

from src.config import LGBM_QUANTILE_N_ESTIMATORS
from src.models.intervals import ConformalIntervalModel, _fit_lgbm_quantile, coverage


@pytest.fixture
def linear_data():
    rng = np.random.RandomState(42)
    n = 400
    age = rng.uniform(1, 20, n)
    odometer = rng.uniform(1_000, 200_000, n)
    noise = rng.normal(0, 0.1, n)
    log_price = 10.5 - 0.05 * age - 0.000002 * odometer + noise
    X = pd.DataFrame({"age": age, "odometer": odometer})
    y = pd.Series(log_price)
    return X, y


def test_fit_lgbm_quantile_delegates_to_shared_helper_and_respects_n_estimators(linear_data):
    X, y = linear_data
    model = _fit_lgbm_quantile(X, y, alpha=0.5, n_estimators=40)
    assert model.best_iteration_ <= 40


def test_defaults_source_from_config():
    import inspect

    sig = inspect.signature(_fit_lgbm_quantile)
    assert sig.parameters["n_estimators"].default == LGBM_QUANTILE_N_ESTIMATORS


def test_conformal_interval_achieves_roughly_nominal_coverage(linear_data):
    X, y = linear_data
    X_train, X_rest = X.iloc[:250], X.iloc[250:]
    y_train, y_rest = y.iloc[:250], y.iloc[250:]
    X_calib, X_test = X_rest.iloc[:75], X_rest.iloc[75:]
    y_calib, y_test = y_rest.iloc[:75], y_rest.iloc[75:]

    model = ConformalIntervalModel(alpha=0.10, n_estimators=60)
    model.fit(X_train, y_train, X_calib, y_calib)
    lo, hi = model.predict_interval_dollar(X_test)
    price_actual = np.expm1(y_test.values)

    empirical_coverage = coverage(price_actual, lo, hi)
    assert empirical_coverage > 0.6  # loose bound: small synthetic n, not a tight calibration check
