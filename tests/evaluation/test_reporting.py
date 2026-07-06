"""Tests for src/evaluation/reporting.py (extracted from scripts/*.py)."""

import pandas as pd
import pytest

from src.evaluation.reporting import (
    a1_verdict,
    a3_verdict,
    age_segment_observation,
    anomaly_listing_note,
    desc_ablation_verdict,
    model_comparison_verdict,
    mondrian_segment_verdict,
    price_segment_observation,
    replace_doc_section,
)


def test_replace_doc_section_replaces_existing_section(tmp_path):
    path = tmp_path / "results.md"
    path.write_text("# Title\n\n## Section A\nold content\n", encoding="utf-8")
    replace_doc_section(path, "## Section A", "## Section A\nnew content\n")
    text = path.read_text(encoding="utf-8")
    assert "old content" not in text
    assert "new content" in text
    assert "# Title" in text


def test_replace_doc_section_tries_multiple_headers_in_order(tmp_path):
    path = tmp_path / "results.md"
    path.write_text("# Title\n\n## Old Header\nstale\n", encoding="utf-8")
    replace_doc_section(path, ["## New Header", "## Old Header"], "## New Header\nfresh\n")
    text = path.read_text(encoding="utf-8")
    assert "stale" not in text
    assert "fresh" in text


def test_replace_doc_section_appends_when_header_absent(tmp_path):
    path = tmp_path / "results.md"
    path.write_text("# Title\nexisting body\n", encoding="utf-8")
    replace_doc_section(path, "## New Section", "## New Section\ncontent\n")
    text = path.read_text(encoding="utf-8")
    assert "existing body" in text
    assert "## New Section" in text


def test_replace_doc_section_creates_new_file(tmp_path):
    path = tmp_path / "new.md"
    replace_doc_section(path, "## Section", "## Section\nhello\n")
    assert path.read_text(encoding="utf-8") == "## Section\nhello\n"


def test_model_comparison_verdict_names_lightgbm_winner():
    lr = {"RMSE ($)": 8000.0, "MAE ($)": 4000.0, "MAPE (%)": 60.0, "R2": 0.6}
    lgbm = {"RMSE ($)": 6000.0, "MAE ($)": 3000.0, "MAPE (%)": 35.0, "R2": 0.78}
    text = model_comparison_verdict(lr, lgbm)
    assert "LightGBM" in text
    assert "25%" in text  # (8000-6000)/8000*100


def test_a1_verdict_reports_log_target_chosen():
    table = pd.DataFrame({
        "RMSE ($)": [6000.0, 5800.0], "MAE ($)": [3000.0, 3100.0],
        "MAPE (%)": [34.0, 49.0], "R2": [0.78, 0.75],
    }, index=["log1p(price)", "raw price"])
    text = a1_verdict(table)
    assert "log target" in text


def test_a3_verdict_names_rmse_winner():
    table = pd.DataFrame({
        "RMSE ($)": [6000.0, 6100.0, 6300.0], "MAE ($)": [3000.0, 3050.0, 3200.0],
        "MAPE (%)": [34.0, 33.5, 36.0], "R2": [0.78, 0.77, 0.75],
    }, index=["target_encoding", "frequency_encoding", "drop_model_column"])
    text = a3_verdict(table)
    assert "target_encoding" in text


def test_age_segment_observation_same_bucket_wording():
    err_age = pd.DataFrame({"MAE": [5000.0, 1000.0], "MAPE": [50.0, 10.0]}, index=["0-3yr", "16+yr"])
    text = age_segment_observation(err_age)
    assert "0-3yr" in text
    assert "both the highest MAE" in text


def test_age_segment_observation_different_bucket_wording():
    err_age = pd.DataFrame({"MAE": [5000.0, 1000.0], "MAPE": [10.0, 90.0]}, index=["0-3yr", "16+yr"])
    text = age_segment_observation(err_age)
    assert "highest-MAE age bucket is 0-3yr" in text
    assert "highest-MAPE bucket is 16+yr" in text


def test_price_segment_observation_uses_priciest_label_when_present():
    err_price = pd.DataFrame(
        {"MAE": [1000.0, 20000.0], "MAPE": [90.0, 20.0]}, index=["<5k", "50-150k"]
    )
    text = price_segment_observation(err_price)
    assert "<5k" in text
    assert "50-150k" in text


def test_price_segment_observation_falls_back_to_first_row():
    err_price = pd.DataFrame({"MAE": [1000.0], "MAPE": [50.0]}, index=["<5k"])
    text = price_segment_observation(err_price)
    assert "<5k" in text


def test_mondrian_segment_verdict_gap_closed():
    seg = pd.DataFrame({
        "standard_coverage": [0.70, 0.75], "mondrian_coverage": [0.89, 0.90],
    }, index=["50-150k", "20-50k"])
    assert "materially closed" in mondrian_segment_verdict(seg)


def test_mondrian_segment_verdict_not_closed():
    seg = pd.DataFrame({
        "standard_coverage": [0.70, 0.75], "mondrian_coverage": [0.70, 0.76],
    }, index=["50-150k", "20-50k"])
    assert "did NOT materially improve" in mondrian_segment_verdict(seg)


def test_desc_ablation_verdict_real_improvement():
    text = desc_ablation_verdict(rmse_base=6591.0, rmse_ext=6252.0, mape_base=36.93,
                                  mape_ext=32.40, pct_rmse=-5.14, total_gain=8.34)
    assert "real improvement" in text


def test_desc_ablation_verdict_negative_result():
    text = desc_ablation_verdict(rmse_base=6591.0, rmse_ext=6600.0, mape_base=36.93,
                                  mape_ext=37.0, pct_rmse=0.1, total_gain=0.2)
    assert "negative result" in text


def test_anomaly_listing_note_underpriced():
    row = pd.Series({
        "year": 2015, "manufacturer": "ford", "model": "f-150", "odometer": 50000,
        "condition": "good", "price": 500.0, "predicted_price": 15000.0,
        "residual_pct": -96.7, "if_score": 0.1, "if_flag": True,
    })
    note = anomaly_listing_note(row, "underpriced")
    assert "2015 ford f-150" in note
    assert "scam" in note
    assert "Also flagged by Isolation Forest" in note


def test_anomaly_listing_note_missing_condition():
    row = pd.Series({
        "year": 2015, "manufacturer": "ford", "model": "f-150", "odometer": 50000,
        "condition": None, "price": 123456.0, "predicted_price": 15000.0,
        "residual_pct": 723.0, "if_score": 0.1, "if_flag": False,
    })
    note = anomaly_listing_note(row, "overpriced")
    assert "condition missing" in note
    assert "Also flagged" not in note
