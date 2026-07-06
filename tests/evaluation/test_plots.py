"""Smoke tests for src/evaluation/plots.py.

Plots are checked structurally (figure returned, right number of axes, no
exception on edge-case input) rather than by pixel comparison. Every test
uses a small synthetic DataFrame -- no dependency on the real 426k-row
dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.evaluation import plots


@pytest.fixture
def cleaned_df():
    n = 200
    rng = np.random.RandomState(42)
    manufacturers = rng.choice(["ford", "toyota", "honda", "chevrolet", "bmw", "nissan", "kia"], n)
    age = rng.randint(1, 20, n)
    odometer = rng.uniform(1_000, 200_000, n)
    price = rng.uniform(1_000, 60_000, n)
    return pd.DataFrame({
        "manufacturer": manufacturers,
        "model": rng.choice([f"model_{i}" for i in range(10)], n),
        "age": age,
        "odometer": odometer,
        "price": price,
        "log_price": np.log1p(price),
        "log_odometer": np.log1p(odometer),
        "year": 2021 - age,
        "mileage_per_year": odometer / age,
        "cylinders_num": rng.choice([4, 6, 8], n).astype(float),
        "condition": rng.choice(["new", "good", "fair", None], n),
        "type": rng.choice(["sedan", "truck", "suv", None], n),
        "drive": rng.choice(["4wd", "fwd", None], n),
        "paint_color": rng.choice(["black", "white", None], n),
        "cylinders": rng.choice(["4 cylinders", "6 cylinders", None], n),
        "size": rng.choice(["compact", "full-size", None], n),
        "VIN": [f"VIN{i}" if i % 3 else None for i in range(n)],
        "state": rng.choice(["ca", "tx", "ny"], n),
    })


def test_save_creates_parent_dirs(tmp_path):
    fig, _ = plt.subplots()
    out = tmp_path / "nested" / "out.png"
    plots._save(fig, out)
    assert out.exists()


def test_save_noop_when_path_none():
    fig, _ = plt.subplots()
    plots._save(fig, None)  # must not raise


def test_plot_price_distribution_returns_figure_with_two_axes(cleaned_df):
    fig = plots.plot_price_distribution(cleaned_df)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2


def test_plot_depreciation_respects_top_n_param(cleaned_df):
    fig = plots.plot_depreciation(cleaned_df, top_n_manufacturers=3)
    ax = fig.axes[0]
    assert len(ax.get_legend().get_texts()) == 3


def test_plot_depreciation_handles_missing_ages_without_erroring(cleaned_df):
    df = cleaned_df.copy()
    df = df[~((df["manufacturer"] == df["manufacturer"].iloc[0]) & (df["age"] == df["age"].iloc[0]))]
    fig = plots.plot_depreciation(df)
    assert isinstance(fig, plt.Figure)


def test_plot_odometer_vs_price_sampling_deterministic(cleaned_df):
    fig1 = plots.plot_odometer_vs_price(cleaned_df, sample_size=50, random_state=7)
    fig2 = plots.plot_odometer_vs_price(cleaned_df, sample_size=50, random_state=7)
    x1 = fig1.axes[0].collections[0].get_offsets()
    x2 = fig2.axes[0].collections[0].get_offsets()
    assert np.array_equal(x1, x2)


def test_plot_value_heaping_custom_highlight_endings(cleaned_df):
    fig = plots.plot_value_heaping(cleaned_df, highlight_endings=(1,))
    assert isinstance(fig, plt.Figure)


def test_plot_confound_check_uses_config_defaults(cleaned_df):
    fig = plots.plot_confound_check(cleaned_df)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2


def test_plot_age_odometer_interaction_custom_bins(cleaned_df):
    fig = plots.plot_age_odometer_interaction(cleaned_df, age_bins=(0, 10, 60), odo_bins=(0, 100_000, 500_000))
    assert isinstance(fig, plt.Figure)


def test_plot_missingness_and_cardinality_custom_cols(cleaned_df):
    fig = plots.plot_missingness_and_cardinality(cleaned_df, miss_cols=("condition", "type"))
    assert isinstance(fig, plt.Figure)


def test_plot_anomaly_overview_z_threshold_matches_flag_line(cleaned_df):
    n = len(cleaned_df)
    residual_z = np.random.RandomState(0).normal(size=n)
    flag = np.abs(residual_z) > 2.0
    fig = plots.plot_anomaly_overview(
        residual_z, cleaned_df["price"].values, cleaned_df["price"].values * 0.9, flag,
        z_threshold=2.0,
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2


def test_plot_functions_accept_save_path_as_str_or_path(cleaned_df, tmp_path):
    fig1 = plots.plot_correlation_heatmap(cleaned_df, save_path=str(tmp_path / "a.png"))
    fig2 = plots.plot_correlation_heatmap(cleaned_df, save_path=tmp_path / "b.png")
    assert (tmp_path / "a.png").exists()
    assert (tmp_path / "b.png").exists()


def test_plot_correlation_heatmap_custom_columns(cleaned_df):
    fig = plots.plot_correlation_heatmap(cleaned_df, numeric_cols=("price", "age"))
    assert isinstance(fig, plt.Figure)


def test_plot_interval_width_nominal_default_matches_config():
    import inspect

    from src.config import INTERVAL_ALPHA

    default = inspect.signature(plots.plot_interval_width).parameters["nominal"].default
    assert abs(default - (1 - INTERVAL_ALPHA)) < 1e-9
