"""Reusable EDA plotting utilities.

Each function builds one figure, optionally saves it to disk, and returns the
Matplotlib Figure so a notebook can display it inline. No business logic or
data cleaning happens here - input is always the cleaned dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import AGE_BUCKET_BINS, AGE_BUCKET_LABELS, ANOMALY_Z_THRESHOLD, INTERVAL_ALPHA, RANDOM_STATE

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

TOP_N_MANUFACTURERS = 6


def _save(fig: plt.Figure, save_path: Path | str | None) -> None:
    """Save a figure to disk, creating parent directories as needed.

    Args:
        fig: Figure to save.
        save_path: Destination path. If None, this is a no-op.
    """
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)


def plot_price_distribution(df: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Plot raw price vs log1p(price) to justify the log target transform.

    Args:
        df: Cleaned dataset with `price` and `log_price` columns.
        save_path: Optional path to save the figure to.

    Returns:
        plt.Figure: The two-panel histogram figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df["price"], bins=60, ax=axes[0], color="#c0392b")
    axes[0].set_title(f"Raw price (skew = {df['price'].skew():.2f})")
    axes[0].set_xlabel("price ($)")

    sns.histplot(df["log_price"], bins=60, ax=axes[1], color="#27ae60")
    axes[1].set_title(f"log1p(price) (skew = {df['log_price'].skew():.2f})")
    axes[1].set_xlabel("log1p(price)")

    fig.suptitle("Price is right-skewed; log transform makes it near-symmetric", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_depreciation(
    df: pd.DataFrame,
    save_path: Path | str | None = None,
    max_age: int = 30,
    top_n_manufacturers: int = TOP_N_MANUFACTURERS,
) -> plt.Figure:
    """Plot median price vs vehicle age for the top-N manufacturers.

    Args:
        df: Cleaned dataset with `manufacturer`, `age`, `price` columns.
        save_path: Optional path to save the figure to.
        max_age: Maximum vehicle age (years) to include.
        top_n_manufacturers: Number of manufacturers (by listing count) to plot.

    Returns:
        plt.Figure: The depreciation-curve figure.
    """
    top = df["manufacturer"].value_counts().head(top_n_manufacturers).index
    sub = df[(df["manufacturer"].isin(top)) & (df["age"] <= max_age)]
    curve = (
        sub.groupby(["manufacturer", "age"])["price"]
        .median()
        .reset_index()
    )

    full_age_range = pd.RangeIndex(int(sub["age"].min()), int(sub["age"].max()) + 1, name="age")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for man in top:
        d = (
            curve[curve["manufacturer"] == man]
            .set_index("age")
            .reindex(full_age_range)
        )
        ax.plot(d.index, d["price"], marker="o", markersize=3, label=man)
    ax.set_title(f"Depreciation curves: median price vs age (top {top_n_manufacturers} manufacturers)")
    ax.set_xlabel("Vehicle age (years)")
    ax.set_ylabel("Median price ($)")
    ax.legend(title="manufacturer")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_odometer_vs_price(
    df: pd.DataFrame,
    save_path: Path | str | None = None,
    sample_size: int = 40_000,
    random_state: int = RANDOM_STATE,
) -> plt.Figure:
    """Plot density of odometer vs price with a binned median trend line.

    Args:
        df: Cleaned dataset with `odometer` and `price` columns.
        save_path: Optional path to save the figure to.
        sample_size: Max number of rows to scatter-plot (for render speed).
        random_state: Seed for the row sample.

    Returns:
        plt.Figure: The odometer-vs-price scatter + trend figure.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sample = df.sample(min(sample_size, len(df)), random_state=random_state)
    ax.scatter(sample["odometer"], sample["price"], s=4, alpha=0.15, color="#2c3e50")

    bins = np.linspace(0, df["odometer"].quantile(0.99), 40)
    df_b = df[df["odometer"] <= bins[-1]].copy()
    df_b["bin"] = pd.cut(df_b["odometer"], bins)
    trend = df_b.groupby("bin", observed=True)["price"].median()
    centers = [iv.mid for iv in trend.index]
    ax.plot(centers, trend.values, color="#e67e22", linewidth=2.5, label="median price")

    ax.set_title("Price vs odometer: steep drop early, flattening after ~150k miles")
    ax.set_xlabel("odometer (miles)")
    ax.set_ylabel("price ($)")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_price_by_category(df: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Plot boxplots of price by vehicle type and by condition.

    Args:
        df: Cleaned dataset with `type`, `condition`, `price` columns.
        save_path: Optional path to save the figure to.

    Returns:
        plt.Figure: The two-panel boxplot figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    type_order = (
        df[df["type"].notna()].groupby("type")["price"].median().sort_values(ascending=False).index
    )
    sns.boxplot(data=df[df["type"].notna()], x="price", y="type", order=type_order,
                ax=axes[0], showfliers=False, color="#3498db")
    axes[0].set_title("Price by vehicle type")

    cond_order = ["new", "like new", "excellent", "good", "fair", "salvage"]
    cond_present = [c for c in cond_order if c in df["condition"].unique()]
    sns.boxplot(data=df[df["condition"].isin(cond_present)], x="price", y="condition",
                order=cond_present, ax=axes[1], showfliers=False, color="#9b59b6")
    axes[1].set_title("Price by condition")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_manufacturer_median_price(df: pd.DataFrame, save_path: Path | str | None = None,
                                   min_count: int = 500) -> plt.Figure:
    """Plot median price by manufacturer (only brands with enough listings).

    Args:
        df: Cleaned dataset with `manufacturer` and `price` columns.
        save_path: Optional path to save the figure to.
        min_count: Minimum listing count for a manufacturer to be included.

    Returns:
        plt.Figure: The horizontal bar chart figure.
    """
    counts = df["manufacturer"].value_counts()
    keep = counts[counts >= min_count].index
    med = (
        df[df["manufacturer"].isin(keep)]
        .groupby("manufacturer")["price"]
        .median()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.barplot(x=med.values, y=med.index, ax=ax, color="#16a085")
    ax.set_title(f"Median listing price by manufacturer (>= {min_count} listings)")
    ax.set_xlabel("Median price ($)")
    ax.set_ylabel("")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_state_median_price(df: pd.DataFrame, save_path: Path | str | None = None,
                            top_bottom: int = 12) -> plt.Figure:
    """Plot top and bottom states by median price to show regional arbitrage.

    Args:
        df: Cleaned dataset with `state` and `price` columns.
        save_path: Optional path to save the figure to.
        top_bottom: Number of states to show at each end of the ranking.

    Returns:
        plt.Figure: The horizontal bar chart figure.
    """
    med = df.groupby("state")["price"].median().sort_values(ascending=False)
    sel = pd.concat([med.head(top_bottom), med.tail(top_bottom)])
    colors = ["#27ae60"] * top_bottom + ["#c0392b"] * top_bottom
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y=range(len(sel)), width=sel.values, color=colors)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels(sel.index)
    ax.invert_yaxis()
    ax.set_title(f"Highest vs lowest {top_bottom} states by median price")
    ax.set_xlabel("Median price ($)")
    ax.set_ylabel("state")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_value_heaping(
    df: pd.DataFrame,
    save_path: Path | str | None = None,
    highlight_endings: tuple[int, ...] = (995, 999, 0, 500),
) -> plt.Figure:
    """Plot psychological price endings and odometer rounding.

    Behavioral pricing pattern (e.g. $X,995) plus a data-quality signal
    (rounded odometer readings).

    Args:
        df: Cleaned dataset with `price` and `odometer` columns.
        save_path: Optional path to save the figure to.
        highlight_endings: Price-mod-1000 values to highlight as "interesting"
            (psychologically anchored endings) in the left panel.

    Returns:
        plt.Figure: The two-panel figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    price_end = (df["price"].astype(int) % 1000)
    top_end = price_end.value_counts().head(12).sort_values(ascending=False)
    colors = ["#c0392b" if e in highlight_endings else "#95a5a6" for e in top_end.index]
    axes[0].bar([str(e) for e in top_end.index], top_end.values, color=colors)
    axes[0].set_title("Most common price endings (last 3 digits)")
    axes[0].set_xlabel("price mod 1000")
    axes[0].set_ylabel("count")
    axes[0].tick_params(axis="x", rotation=45)

    odo_round = (df["odometer"].astype(int) % 1000 == 0)
    share = pd.Series({
        "ends in 000\n(rounded)": odo_round.mean(),
        "any other\nvalue": 1 - odo_round.mean(),
    })
    axes[1].bar(share.index, share.values, color=["#e67e22", "#95a5a6"])
    axes[1].set_title("Odometer rounding: ~30% report a round-thousand mileage")
    axes[1].set_ylabel("share of listings")
    for i, v in enumerate(share.values):
        axes[1].text(i, v + 0.01, f"{v:.0%}", ha="center")

    fig.suptitle("Value heaping: prices anchor at .995/.999, mileage is rounded", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_confound_check(
    df: pd.DataFrame,
    save_path: Path | str | None = None,
    age_bins: list[float] = AGE_BUCKET_BINS,
    age_labels: list[str] = AGE_BUCKET_LABELS,
) -> plt.Figure:
    """Show that raw 'premiums' for VIN and missing-condition largely reflect age.

    Args:
        df: Cleaned dataset with `VIN`, `condition`, `age`, `price` columns.
        save_path: Optional path to save the figure to.
        age_bins: Bin edges for the age bucketing.
        age_labels: Labels matching `age_bins` (one fewer than the number of edges).

    Returns:
        plt.Figure: The two-panel grouped bar chart figure.
    """
    d = df.copy()
    d["has_vin"] = d["VIN"].notna() & (d["VIN"].str.strip() != "")
    d["cond_missing"] = d["condition"].isna()
    d["age_bucket"] = pd.cut(d["age"], age_bins, labels=age_labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, flag, labels, title in [
        (axes[0], "has_vin", ("no VIN", "has VIN"), "VIN 'premium' by age bucket"),
        (axes[1], "cond_missing", ("condition present", "condition missing"),
         "Missing-condition 'premium' by age bucket"),
    ]:
        piv = d.pivot_table("price", "age_bucket", flag, "median", observed=True)
        x = np.arange(len(piv.index))
        w = 0.38
        ax.bar(x - w / 2, piv[False].values, w, label=labels[0], color="#95a5a6")
        ax.bar(x + w / 2, piv[True].values, w, label=labels[1], color="#2980b9")
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index)
        ax.set_xlabel("age bucket (years)")
        ax.set_ylabel("median price ($)")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Within each age bucket the gaps shrink: the raw premiums are mostly age",
                 fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_age_odometer_interaction(
    df: pd.DataFrame,
    save_path: Path | str | None = None,
    age_bins: list[float] = (0, 2, 4, 6, 8, 10, 13, 16, 20, 60),
    odo_bins: list[float] = (0, 20_000, 40_000, 60_000, 80_000, 100_000, 130_000, 160_000, 200_000, 500_000),
) -> plt.Figure:
    """Plot a 2D median-price grid over age x odometer to expose their interaction.

    Args:
        df: Cleaned dataset with `age`, `odometer`, `price` columns.
        save_path: Optional path to save the figure to.
        age_bins: Bin edges for the age axis.
        odo_bins: Bin edges for the odometer axis.

    Returns:
        plt.Figure: The heatmap figure.
    """
    d = df.copy()
    d["age_b"] = pd.cut(d["age"], list(age_bins))
    d["odo_b"] = pd.cut(d["odometer"], list(odo_bins))
    grid = d.pivot_table("price", "odo_b", "age_b", "median", observed=False)

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(grid / 1000, annot=True, fmt=".0f", cmap="viridis", ax=ax,
                cbar_kws={"label": "median price ($k)"})
    ax.set_title("Median price ($k) over age x odometer: the two interact, not add")
    ax.set_xlabel("age bucket (years)")
    ax.set_ylabel("odometer bucket (miles)")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_missingness_and_cardinality(
    df: pd.DataFrame,
    save_path: Path | str | None = None,
    miss_cols: list[str] = ("condition", "drive", "type", "paint_color", "cylinders", "size"),
    cardinality_bins: list[float] = (0, 1, 4, 19, 99, np.inf),
    cardinality_labels: list[str] = ("1", "2-4", "5-19", "20-99", "100+"),
) -> plt.Figure:
    """Plot structured missingness co-occurrence and model long-tail cardinality.

    Args:
        df: Cleaned dataset.
        save_path: Optional path to save the figure to.
        miss_cols: Columns to include in the missingness co-occurrence panel.
        cardinality_bins: Bin edges for "listings per model" bucketing.
        cardinality_labels: Labels matching `cardinality_bins`.

    Returns:
        plt.Figure: The two-panel figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    cols = [c for c in miss_cols if c in df.columns]
    miss_corr = df[cols].isna().corr()
    sns.heatmap(miss_corr, annot=True, fmt=".2f", cmap="Oranges", vmin=0, vmax=1,
                ax=axes[0], square=True, cbar_kws={"shrink": 0.7})
    axes[0].set_title("Missingness co-occurrence (phi)\nfields go missing together")

    vc = df["model"].value_counts()
    cardinality_labels = list(cardinality_labels)
    buckets = pd.cut(vc, list(cardinality_bins), labels=cardinality_labels)
    n_models = buckets.value_counts().reindex(cardinality_labels)
    rows_covered = vc.groupby(buckets, observed=False).sum().reindex(n_models.index)

    x = np.arange(len(n_models))
    ax2 = axes[1]
    ax2.bar(x - 0.2, n_models.values, 0.4, label="# of models", color="#2980b9")
    ax2.set_ylabel("# of distinct models", color="#2980b9")
    ax2.set_yscale("log")
    ax3 = ax2.twinx()
    ax3.bar(x + 0.2, rows_covered.values, 0.4, label="# of listings", color="#e67e22")
    ax3.set_ylabel("# of listings covered", color="#e67e22")
    ax2.set_xticks(x)
    ax2.set_xticklabels(n_models.index)
    ax2.set_xlabel("listings per model")
    ax2.set_title(f"model cardinality: {df['model'].nunique():,} unique;\n"
                  f"{(vc < 5).mean():.0%} have < 5 listings")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_anomaly_overview(
    residual_z: np.ndarray,
    price: np.ndarray,
    predicted: np.ndarray,
    residual_flag: np.ndarray,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
    save_path: Path | str | None = None,
    sample_size: int = 30_000,
    random_state: int = RANDOM_STATE,
) -> plt.Figure:
    """Plot the robust-z residual distribution and predicted-vs-actual scatter.

    Left panel: robust-z residual distribution with flagged tails. Right
    panel: predicted vs actual price, flagged listings highlighted.

    Args:
        residual_z: Robust z-score per row.
        price: Actual listed price per row.
        predicted: Model-predicted price per row.
        residual_flag: Boolean flag array, True where a row is anomalous.
        z_threshold: Reference line drawn at +/- this z value. Should match
            the threshold actually used to compute `residual_flag`.
        save_path: Optional path to save the figure to.
        sample_size: Max number of rows to scatter-plot in the right panel.
        random_state: Seed for the row sample in the right panel.

    Returns:
        plt.Figure: The two-panel figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    clipped = np.clip(residual_z, -10, 10)
    axes[0].hist(clipped, bins=80, color="#95a5a6")
    axes[0].axvline(z_threshold, color="#c0392b", ls="--", label=f"+/-{z_threshold}")
    axes[0].axvline(-z_threshold, color="#c0392b", ls="--")
    axes[0].set_title("Robust residual z-score (clipped to +/-10)")
    axes[0].set_xlabel("residual z (median/MAD standardized)")
    axes[0].set_ylabel("count")
    axes[0].set_yscale("log")
    axes[0].legend()

    n = len(price)
    idx = np.random.RandomState(random_state).choice(n, size=min(sample_size, n), replace=False)
    normal = idx[~residual_flag[idx]]
    flagged = idx[residual_flag[idx]]
    axes[1].scatter(predicted[normal], price[normal], s=4, alpha=0.12, color="#2c3e50",
                    label="normal")
    axes[1].scatter(predicted[flagged], price[flagged], s=10, alpha=0.5, color="#c0392b",
                    label="flagged")
    lim = np.percentile(price, 99.5)
    axes[1].plot([0, lim], [0, lim], color="#e67e22", lw=1.5, label="perfect")
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].set_title("Predicted vs actual price (flagged = mispriced)")
    axes[1].set_xlabel("predicted price ($)")
    axes[1].set_ylabel("actual listed price ($)")
    axes[1].legend()

    fig.suptitle("Anomaly detection: residual tails and mispriced listings", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_interval_width(
    price: np.ndarray,
    width: np.ndarray,
    coverage_by_segment: pd.Series | None = None,
    coverage_comparison: pd.Series | None = None,
    comparison_labels: tuple[str, str] = ("standard CQR", "Mondrian"),
    nominal: float = 1 - INTERVAL_ALPHA,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot conformal interval width vs price and empirical coverage by segment.

    Left panel: conformal interval width vs price (binned median) -- shows
    heteroscedasticity honestly instead of hiding it behind a global z-score.
    Right panel: empirical coverage per segment vs the nominal target; if
    `coverage_comparison` is given, grouped bars compare the two variants.

    Args:
        price: Actual listed price per row.
        width: Interval width (hi - lo) per row.
        coverage_by_segment: Optional per-segment empirical coverage.
        coverage_comparison: Optional second per-segment coverage series to
            compare against `coverage_by_segment`.
        comparison_labels: Legend labels for the two coverage series.
        nominal: Nominal target coverage (e.g. 0.90 for a 90% interval).
        save_path: Optional path to save the figure to.

    Returns:
        plt.Figure: The two-panel figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cap = np.percentile(price, 99)
    d = pd.DataFrame({"price": price, "width": width})
    d = d[d["price"] <= cap]
    bins = np.linspace(0, cap, 30)
    d["bin"] = pd.cut(d["price"], bins)
    trend = d.groupby("bin", observed=True)["width"].median()
    centers = [iv.mid for iv in trend.index]
    axes[0].plot(centers, trend.values, color="#8e44ad", marker="o", markersize=3)
    axes[0].set_title(f"{nominal:.0%} conformal interval width vs price")
    axes[0].set_xlabel("price ($)")
    axes[0].set_ylabel("interval width ($)")

    if coverage_by_segment is not None:
        labels = coverage_by_segment.index.astype(str)
        x = np.arange(len(labels))
        if coverage_comparison is not None:
            w = 0.38
            axes[1].bar(x - w / 2, coverage_by_segment.values, w,
                        label=comparison_labels[0], color="#95a5a6")
            axes[1].bar(x + w / 2, coverage_comparison.values, w,
                        label=comparison_labels[1], color="#2980b9")
        else:
            axes[1].bar(x, coverage_by_segment.values, color="#2980b9")
        axes[1].axhline(nominal, color="#c0392b", ls="--", label=f"nominal {nominal:.0%}")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45)
        axes[1].set_title("Empirical coverage by price segment")
        axes[1].set_ylabel("coverage")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()

    title = ("Prediction intervals: width scales with price; Mondrian calibration "
             "vs standard CQR coverage"
             if coverage_comparison is not None else
             "Prediction intervals: width scales with price, but coverage is uneven "
             "across price tiers")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Path | str | None = None,
    numeric_cols: list[str] = (
        "price", "log_price", "age", "year", "odometer", "log_odometer",
        "mileage_per_year", "cylinders_num",
    ),
) -> plt.Figure:
    """Plot correlation of numeric features with price.

    Args:
        df: Cleaned dataset.
        save_path: Optional path to save the figure to.
        numeric_cols: Numeric columns to include in the correlation matrix.

    Returns:
        plt.Figure: The correlation heatmap figure.
    """
    cols = [c for c in numeric_cols if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Numeric feature correlations")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
