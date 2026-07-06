"""Smoke test for scripts/detect_anomalies.py::write_results after Phase 8.12.

Exercises the wiring to anomaly_listing_note and fat_tail_comparison with
synthetic data, without running the actual 5-fold OOF pipeline.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

import scripts.detect_anomalies as detect_script


def test_write_results_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(detect_script, "RESULTS", tmp_path / "phase4_results.md")

    listing = pd.Series({
        "year": 2015, "manufacturer": "ford", "model": "f-150", "odometer": 50000,
        "condition": "good", "price": 500.0, "predicted_price": 15000.0,
        "residual_pct": -96.7, "if_score": 0.1, "if_flag": True,
    })
    top_under = pd.DataFrame([listing])
    top_over = pd.DataFrame([listing])
    top_struct = pd.DataFrame([listing])

    detect_script.write_results(
        n_total=197814,
        n_resid=7000, n_strong=2197, n_moderate=5362,
        n_if=1979, n_both=220, n_high=53,
        oof_std=0.42, insample_std=0.38,
        d_std_low=1200.0, d_std_high=4300.0,
        l_std_low=0.35, l_std_high=0.30,
        if_runtime=12.3,
        fat_tail=[(3.5, 7000, 3.5, 87.0, 0.044), (5, 2197, 1.1, 0.5, 0.0003)],
        top_under=top_under, top_over=top_over, top_struct=top_struct,
    )

    text = detect_script.RESULTS.read_text(encoding="utf-8")
    assert "Phase 4 Results" in text
    assert "2015 ford f-150" in text
