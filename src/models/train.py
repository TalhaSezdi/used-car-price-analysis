"""Model training logic: Linear, Random Forest, LightGBM + ablations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    name: str
    model: object
    predictions: np.ndarray
    metrics: dict[str, float]


def train_linear(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 price_test: np.ndarray | None = None) -> TrainedModel:
    from src.evaluation.metrics import compute_metrics
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    m = compute_metrics(y_test, preds, price_test)
    logger.info("Linear Regression: %s", m)
    return TrainedModel("Linear Regression", lr, preds, m)


def train_rf(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray,
             price_test: np.ndarray | None = None) -> TrainedModel:
    from src.evaluation.metrics import compute_metrics
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    m = compute_metrics(y_test, preds, price_test)
    logger.info("Random Forest: %s", m)
    return TrainedModel("Random Forest", rf, preds, m)


LGBM_PARAMS: dict = dict(
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    verbose=-1,
)


def _fit_lgbm(X_train, y_train, params: dict, n_estimators: int = 3000):
    """Fit LightGBM with early stopping on a validation split carved from TRAIN.

    Critical: the test set is NEVER used for early stopping. Choosing the
    number of boosting rounds by watching the test loss would make the test
    metric optimistically biased (the test set stops being a clean holdout).
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
    )
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def train_lgbm(X_train, y_train, X_test, y_test,
               price_test: np.ndarray | None = None) -> TrainedModel:
    from src.evaluation.metrics import compute_metrics

    model = _fit_lgbm(X_train, y_train, LGBM_PARAMS)
    preds = model.predict(X_test)
    m = compute_metrics(y_test, preds, price_test)
    logger.info("LightGBM (best_iter=%s): %s", model.best_iteration_, m)
    return TrainedModel("LightGBM", model, preds, m)


def oof_log_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    low_card_cols: list[str],
    high_card_cols: list[str],
    n_splits: int = 5,
    n_estimators: int = 600,
) -> np.ndarray:
    """Out-of-fold LightGBM predictions (log scale) for EVERY row.

    Each row is predicted by a model that never saw it in training -- this is
    the leakage-safe basis for residual anomaly scoring. Fitting a model on all
    rows and predicting in-sample would yield artificially small residuals and
    hide real anomalies.

    A fixed n_estimators (no early stopping) is used inside the CV loop: for
    relative residual ranking we need a reasonable model, not a maximally tuned
    one, and a fixed round count keeps the folds comparable.
    """
    from sklearn.model_selection import KFold
    import lightgbm as lgb
    from src.models.encoders import FeaturePreprocessor

    oof = np.full(len(X), np.nan, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]

        prep = FeaturePreprocessor(
            numeric_cols=numeric_cols,
            low_card_cols=low_card_cols,
            high_card_cols=high_card_cols,
            high_card_method="target",
        )
        Xt = prep.fit_transform(X_tr, y_tr)   # OOF target encoding within the fold
        Xv = prep.transform(X_va)

        model = lgb.LGBMRegressor(n_estimators=n_estimators, **LGBM_PARAMS)
        model.fit(Xt, y_tr)
        oof[va_idx] = model.predict(Xv)
        logger.info("OOF fold %d/%d done", fold, n_splits)

    return oof


def ablation_a1_raw_vs_log(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train_log: pd.Series, y_test_log: pd.Series,
    price_train: pd.Series, price_test: pd.Series,
) -> dict[str, dict]:
    """A1: Compare log target vs raw target on the same LightGBM."""
    from src.evaluation.metrics import compute_metrics

    m_log = _fit_lgbm(X_train, y_train_log, LGBM_PARAMS)
    pred_log = m_log.predict(X_test)
    metrics_log = compute_metrics(y_test_log, pred_log, price_test.values)

    m_raw = _fit_lgbm(X_train, price_train, LGBM_PARAMS)
    pred_raw = m_raw.predict(X_test)
    pred_raw_clipped = np.clip(pred_raw, 0, None)
    from sklearn.metrics import r2_score
    metrics_raw = {
        "RMSE ($)": np.sqrt(np.mean((price_test.values - pred_raw_clipped) ** 2)),
        "MAE ($)": np.mean(np.abs(price_test.values - pred_raw_clipped)),
        "MAPE (%)": np.mean(np.abs(
            (price_test.values - pred_raw_clipped) / np.maximum(price_test.values, 1)
        )) * 100,
        "R2": r2_score(price_test.values, pred_raw_clipped),
    }

    return {"log1p(price)": metrics_log, "raw price": metrics_raw}


def ablation_a2_collinearity(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    year_train: pd.Series, year_test: pd.Series,
) -> dict:
    """A2: Linear model with age only vs age+year -- show VIF explosion."""
    from sklearn.linear_model import LinearRegression
    from src.evaluation.metrics import compute_metrics

    lr_age = LinearRegression()
    lr_age.fit(X_train, y_train)
    pred_age = lr_age.predict(X_test)
    m_age = compute_metrics(y_test, pred_age)

    X_tr2 = X_train.copy()
    X_te2 = X_test.copy()
    X_tr2["year"] = year_train.values
    X_te2["year"] = year_test.values

    lr_both = LinearRegression()
    lr_both.fit(X_tr2, y_train)
    pred_both = lr_both.predict(X_te2)
    m_both = compute_metrics(y_test, pred_both)

    age_idx = list(X_train.columns).index("age") if "age" in X_train.columns else None
    coef_age_only = lr_age.coef_[age_idx] if age_idx is not None else None

    age_idx2 = list(X_tr2.columns).index("age")
    year_idx2 = list(X_tr2.columns).index("year")
    coef_age_both = lr_both.coef_[age_idx2]
    coef_year_both = lr_both.coef_[year_idx2]

    return {
        "age_only": {"metrics": m_age, "coef_age": coef_age_only},
        "age_and_year": {
            "metrics": m_both,
            "coef_age": coef_age_both,
            "coef_year": coef_year_both,
        },
    }
