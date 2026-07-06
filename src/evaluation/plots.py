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

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

TOP_N_MANUFACTURERS = 6


def _save(fig: plt.Figure, save_path: Path | str | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)


def plot_price_distribution(df: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Raw price vs log1p(price) to justify the log target transform."""
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


def plot_depreciation(df: pd.DataFrame, save_path: Path | str | None = None,
                      max_age: int = 30) -> plt.Figure:
    """Median price vs vehicle age for the top-N manufacturers."""
    top = df["manufacturer"].value_counts().head(TOP_N_MANUFACTURERS).index
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
    ax.set_title("Depreciation curves: median price vs age (top 6 manufacturers)")
    ax.set_xlabel("Vehicle age (years)")
    ax.set_ylabel("Median price ($)")
    ax.legend(title="manufacturer")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_odometer_vs_price(df: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Density of odometer vs price with a binned median trend line."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sample = df.sample(min(40000, len(df)), random_state=42)
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
    """Boxplots of price by vehicle type and by condition."""
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
    """Median price by manufacturer (only brands with enough listings)."""
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
    """Top and bottom states by median price to show regional arbitrage."""
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


def plot_value_heaping(df: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Psychological price endings and odometer rounding (behavioral + data quality)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    price_end = (df["price"].astype(int) % 1000)
    top_end = price_end.value_counts().head(12).sort_values(ascending=False)
    colors = ["#c0392b" if e in (995, 999, 0, 500) else "#95a5a6" for e in top_end.index]
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


def plot_confound_check(df: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Show that raw 'premiums' for VIN and missing-condition largely reflect age."""
    d = df.copy()
    d["has_vin"] = d["VIN"].notna() & (d["VIN"].str.strip() != "")
    d["cond_missing"] = d["condition"].isna()
    d["age_bucket"] = pd.cut(d["age"], [0, 3, 6, 10, 15, 60],
                             labels=["0-3", "4-6", "7-10", "11-15", "16+"])

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


def plot_age_odometer_interaction(df: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """2D median-price grid over age x odometer to expose their interaction."""
    d = df.copy()
    age_bins = [0, 2, 4, 6, 8, 10, 13, 16, 20, 60]
    odo_bins = [0, 20_000, 40_000, 60_000, 80_000, 100_000, 130_000, 160_000, 200_000, 500_000]
    d["age_b"] = pd.cut(d["age"], age_bins)
    d["odo_b"] = pd.cut(d["odometer"], odo_bins)
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


def plot_missingness_and_cardinality(df: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Left: structured missingness (co-occurrence). Right: model long-tail cardinality."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    miss_cols = ["condition", "drive", "type", "paint_color", "cylinders", "size"]
    miss_cols = [c for c in miss_cols if c in df.columns]
    miss_corr = df[miss_cols].isna().corr()
    sns.heatmap(miss_corr, annot=True, fmt=".2f", cmap="Oranges", vmin=0, vmax=1,
                ax=axes[0], square=True, cbar_kws={"shrink": 0.7})
    axes[0].set_title("Missingness co-occurrence (phi)\nfields go missing together")

    vc = df["model"].value_counts()
    buckets = pd.cut(vc, [0, 1, 4, 19, 99, np.inf],
                     labels=["1", "2-4", "5-19", "20-99", "100+"])
    n_models = buckets.value_counts().reindex(["1", "2-4", "5-19", "20-99", "100+"])
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
    z_threshold: float = 3.5,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Left: robust-z residual distribution with flagged tails.
    Right: predicted vs actual price, flagged listings highlighted."""
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
    idx = np.random.RandomState(42).choice(n, size=min(30000, n), replace=False)
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
    nominal: float = 0.90,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Left: conformal interval width vs price (binned median) -- shows
    heteroscedasticity honestly instead of hiding it behind a global z-score.
    Right: empirical coverage per segment vs the nominal target; if
    coverage_comparison is given, grouped bars compare the two variants."""
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


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path | str | None = None) -> plt.Figure:
    """Correlation of numeric features with price."""
    cols = ["price", "log_price", "age", "year", "odometer", "log_odometer",
            "mileage_per_year", "cylinders_num"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Numeric feature correlations")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
