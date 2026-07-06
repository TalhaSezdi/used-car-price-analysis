"""Smoke test for scripts/train.py::write_results after the Phase 8.12 refactor.

Exercises the exact wiring (gain_importance_table, a1_verdict, a3_verdict,
model_comparison_verdict, age_segment_observation, price_segment_observation)
with synthetic data shaped like the real pipeline's outputs, without running
the actual 426k-row training pipeline.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

import scripts.train as train_script


@dataclass
class _FakeTrainedModel:
    name: str
    metrics: dict


def test_write_results_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(train_script, "RESULTS", tmp_path / "phase3_results.md")

    val_comparison = pd.DataFrame({
        "RMSE ($)": [8000.0, 6800.0, 6000.0], "MAE ($)": [4000.0, 3600.0, 3200.0],
        "MAPE (%)": [60.0, 45.0, 34.0], "R2": [0.6, 0.73, 0.78],
    }, index=["Linear Regression", "Random Forest", "LightGBM"])

    top_features = pd.Series({"age": 45.0, "model": 18.0, "odometer": 9.0})

    err_age = pd.DataFrame(
        {"MAE": [3000.0, 1200.0], "MAPE": [80.0, 15.0], "count": [100, 200]},
        index=["0-3yr", "16+yr"],
    )
    err_price = pd.DataFrame(
        {"MAE": [1500.0, 20000.0], "MAPE": [89.0, 28.0], "count": [300, 50]},
        index=["<5k", "50-150k"],
    )
    err_brand = pd.DataFrame(
        {"MAE": [3500.0], "MAPE": [30.0], "count": [400]}, index=["ford"],
    )

    a1_table = pd.DataFrame({
        "RMSE ($)": [6000.0, 5800.0], "MAE ($)": [3000.0, 3100.0],
        "MAPE (%)": [34.0, 49.0], "R2": [0.78, 0.75],
    }, index=["log1p(price)", "raw price"])

    a2 = {
        "age_only": {"metrics": {"RMSE ($)": 8200.0, "MAE ($)": 4400.0, "MAPE (%)": 61.0, "R2": 0.62}, "coef_age": -0.05},
        "age_and_year": {
            "metrics": {"RMSE ($)": 8200.0, "MAE ($)": 4400.0, "MAPE (%)": 61.0, "R2": 0.62},
            "coef_age": -1000.0, "coef_year": 999.0,
        },
    }

    a3_table = pd.DataFrame({
        "RMSE ($)": [6000.0, 6100.0, 6300.0], "MAE ($)": [3000.0, 3050.0, 3200.0],
        "MAPE (%)": [34.0, 33.5, 36.0], "R2": [0.78, 0.77, 0.75],
    }, index=["target_encoding", "frequency_encoding", "drop_model_column"])

    lr_model = _FakeTrainedModel("Linear Regression", val_comparison.loc["Linear Regression"].to_dict())
    rf_model = _FakeTrainedModel("Random Forest", val_comparison.loc["Random Forest"].to_dict())
    lgbm_val_model = _FakeTrainedModel("LightGBM", val_comparison.loc["LightGBM"].to_dict())
    lgbm_final_model = _FakeTrainedModel("LightGBM", {"RMSE ($)": 6253.0, "MAE ($)": 3143.0, "MAPE (%)": 32.4, "R2": 0.78})

    n = 1000
    split = type("Split", (), {
        "X_train": pd.DataFrame(index=range(int(n * 0.6))),
        "X_val": pd.DataFrame(index=range(int(n * 0.2))),
        "X_test": pd.DataFrame(index=range(int(n * 0.2))),
        "X_train_full": pd.DataFrame(index=range(int(n * 0.8))),
    })()

    train_script.write_results(
        val_comparison, top_features,
        err_age, err_price, err_brand,
        a1_table, a2, a3_table,
        lr_model, rf_model, lgbm_val_model, lgbm_final_model,
        split,
    )

    text = train_script.RESULTS.read_text(encoding="utf-8")
    assert "Phase 3 Results" in text
    assert "LightGBM" in text
    assert "log target" in text
