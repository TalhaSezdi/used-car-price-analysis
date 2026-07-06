"""Smoke test for scripts/predict_intervals.py::write_results after Phase 8.12."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

import scripts.predict_intervals as pi_script


def test_write_results_smoke(tmp_path, monkeypatch):
    results_path = tmp_path / "phase6_results.md"
    results_path.write_text("# Existing doc\n\nsome earlier phase content\n", encoding="utf-8")
    monkeypatch.setattr(pi_script, "RESULTS", results_path)

    cov = {"90_raw": 0.87, "90_std": 0.902, "90_mon": 0.898, "99_std": 0.988, "99_mon": 0.991}
    seg_compare = pd.DataFrame({
        "standard_coverage": [0.70, 0.92], "mondrian_coverage": [0.89, 0.90],
        "mondrian_median_width": [5000.0, 8000.0], "count": [929, 9191],
    }, index=["50-150k", "20-50k"])
    mon_by_age = pd.DataFrame({
        "coverage": [0.90, 0.89], "median_width": [4000.0, 6000.0], "count": [500, 400],
    }, index=["0-3yr", "16+yr"])
    tie_in = {
        "n_flag90": 3900, "n_flag99": 400, "n_resid": 1400, "n_strong": 300,
        "overlap_90_resid": 1200, "overlap_99_strong": 280,
    }

    pi_script.write_results(
        cov=cov, seg_compare=seg_compare, mon_by_age=mon_by_age,
        n_test=39563, tie_in=tie_in,
        correction_90=0.35, correction_99=0.62,
        mondrian_corrections_90=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
    )

    text = results_path.read_text(encoding="utf-8")
    assert "# Existing doc" in text  # earlier content preserved
    assert "some earlier phase content" in text
    assert "6B/6C. Prediction intervals" in text
    assert "Verdict" in text
