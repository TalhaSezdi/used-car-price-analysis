"""Smoke test for scripts/ablation_description_features.py::write_results (Phase 8.12)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

import scripts.ablation_description_features as ablation_script


def test_write_results_smoke(tmp_path, monkeypatch):
    results_path = tmp_path / "phase7_results.md"
    results_path.write_text("# Existing doc\n\n7A section stays\n", encoding="utf-8")
    monkeypatch.setattr(ablation_script, "RESULTS", results_path)

    comparison = pd.DataFrame({
        "RMSE ($)": [6591.16, 6252.5], "MAE ($)": [3349.24, 3142.81],
        "MAPE (%)": [36.93, 32.4], "R2": [0.76, 0.78],
    }, index=["baseline (no desc_*)", "with desc_* features"])
    gain_desc = pd.Series({"desc_len_log": 5.764, "desc_equip_count": 1.501, "desc_trim_luxury": 1.076})
    err_price = pd.DataFrame({"MAE": [1534.85], "MAPE": [89.26], "count": [7738]}, index=["<5k"])

    ablation_script.write_results(
        comparison, gain_desc, err_price,
        rmse_base=6591.16, rmse_ext=6252.5, mape_base=36.93, mape_ext=32.4, pct_rmse=-5.14,
    )

    text = results_path.read_text(encoding="utf-8")
    assert "7A section stays" in text
    assert "7B. Description trim/equipment features" in text
    assert "real improvement" in text
