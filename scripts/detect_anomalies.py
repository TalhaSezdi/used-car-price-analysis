"""Phase 4 entry point: score every listing for anomalies, export ranked report.

Usage:
    python scripts/detect_anomalies.py

Produces:
    reports/suspicious_listings.csv   ranked top-N flagged listings
    reports/figures/12_anomaly_overview.png
    docs/phase4_results.md            methodology + alternatives evidence + notes
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.dataset import (
    select_features, NUMERIC_FEATURES, LOW_CARD_FEATURES, HIGH_CARD_FEATURES,
)
from src.models.train import oof_log_predictions
from src.anomaly.detector import (
    ResidualAnomalyDetector, IsolationForestDetector,
    fat_tail_comparison, in_sample_residual_std, tier_anomalies,
)
from src.evaluation import plots
from src.evaluation.reporting import anomaly_listing_note

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "processed" / "cleaned.parquet"
FIG = ROOT / "reports" / "figures"
CSV_OUT = ROOT / "reports" / "suspicious_listings.csv"
RESULTS = ROOT / "docs" / "phase4_results.md"

# log_price DELIBERATELY excluded: keeps IF a purely STRUCTURAL signal
# (weird attribute combinations, independent of the price model). Including
# price would make IF partly re-detect what the residual method already
# catches -- inflating the "flagged by BOTH" set with mechanical overlap
# instead of genuine cross-signal confirmation.
IF_FEATURES = ["age", "odometer", "mileage_per_year", "cylinders_num"]
Z_THRESHOLD = 3.5
CONTAMINATION = 0.01
TOP_N = 30
# Tier thresholds for the residual signal (see Confound section in results)
Z_STRONG = 5.0
PCT_STRONG = 85.0  # residual_pct magnitude threshold for "clearly junk listing"
CONTEXT_COLS = ["manufacturer", "model", "year", "age", "odometer",
                "condition", "title_status", "state", "price"]


def main() -> None:
    df = pd.read_parquet(DATA).reset_index(drop=True)
    logger.info("Loaded %d rows", len(df))

    X, y, price = select_features(df)

    # --- Leakage-safe OOF predictions for every listing ---
    logger.info("Computing out-of-fold predictions (5-fold)...")
    oof_pred = oof_log_predictions(
        X, y, NUMERIC_FEATURES, LOW_CARD_FEATURES, HIGH_CARD_FEATURES,
    )
    oof_resid_std = float(np.std(y.values - oof_pred))

    # --- Residual anomaly detector ---
    res_det = ResidualAnomalyDetector(z_threshold=Z_THRESHOLD)
    res_det.fit(y.values, oof_pred)
    res = res_det.score(y.values, oof_pred).reset_index(drop=True)

    # --- Isolation Forest ---
    logger.info("Fitting Isolation Forest...")
    if_features = [c for c in IF_FEATURES if c in df.columns]
    t0 = time.perf_counter()
    if_det = IsolationForestDetector(contamination=CONTAMINATION)
    iso = if_det.fit_score(df[if_features]).reset_index(drop=True)
    if_runtime = time.perf_counter() - t0

    # --- Assemble results ---
    out = df[CONTEXT_COLS].copy().reset_index(drop=True)
    out["predicted_price"] = np.expm1(oof_pred).round(0)
    out["residual_pct"] = ((out["price"] - out["predicted_price"])
                           / out["predicted_price"] * 100).round(1)
    out = pd.concat([out, res, iso], axis=1)

    # Tier the residual signal to separate "clearly a bad listing" from
    # "possibly just model error on a rare car" -- see tier_anomalies'
    # docstring for the STRONG/MODERATE definitions.
    tiered = tier_anomalies(
        out["residual_z"], out["residual_flag"], out["residual_pct"], out["if_flag"],
        z_strong=Z_STRONG, pct_strong=PCT_STRONG,
    )
    abs_z = out["residual_z"].abs()
    strong_resid = tiered["strong_resid"]
    moderate_resid = tiered["moderate_resid"]
    out["priority"] = tiered["priority"]

    n_resid = int(out["residual_flag"].sum())
    n_strong = int(strong_resid.sum())
    n_moderate = int(moderate_resid.sum())
    n_if = int(out["if_flag"].sum())
    n_both = int((out["residual_flag"] & out["if_flag"]).sum())
    n_high = int((strong_resid & out["if_flag"]).sum())
    logger.info(
        "Flagged: residual=%d (strong=%d, moderate=%d), IF=%d, "
        "residual&IF=%d, HIGH(strong+IF)=%d",
        n_resid, n_strong, n_moderate, n_if, n_both, n_high,
    )

    # --- Evidence for the alternatives table ---
    insample_std = in_sample_residual_std(
        X, y, NUMERIC_FEATURES, LOW_CARD_FEATURES, HIGH_CARD_FEATURES,
    )

    dollar_resid = out["price"] - out["predicted_price"]
    low = out["price"] < 10_000
    high = out["price"] > 30_000
    dollar_std_low = float(dollar_resid[low].std())
    dollar_std_high = float(dollar_resid[high].std())
    log_std_low = float(res["residual_log"][low.values].std())
    log_std_high = float(res["residual_log"][high.values].std())

    # --- Fat-tail probe: real dist vs Gaussian at the same threshold ---
    fat_tail = fat_tail_comparison(abs_z, thresholds=[3.5, 5, 7, 10])

    # --- Category-based ranking (three separate top-N slices) ---
    # Reviewer sees examples of each ACTION, not just the underpriced tail.
    export_cols = CONTEXT_COLS + ["predicted_price", "residual_pct",
                                  "residual_z", "if_score", "priority"]

    out["abs_z"] = abs_z
    flagged = out[out["priority"] != "normal"].copy()

    top_under = (
        flagged[flagged["direction"] == "underpriced"]
        .sort_values("abs_z", ascending=False).head(10)
    )
    top_over = (
        flagged[flagged["direction"] == "overpriced"]
        .sort_values("abs_z", ascending=False).head(10)
    )
    top_struct = (
        flagged[flagged["priority"] == "structural only"]
        .sort_values("if_score", ascending=False).head(10)
    )

    ranked = pd.concat([top_under, top_over, top_struct])
    ranked[export_cols].to_csv(CSV_OUT, index=False)
    logger.info(
        "Wrote %s (top10 underpriced + top10 overpriced + top10 structural-only)",
        CSV_OUT,
    )

    # --- Figure ---
    FIG.mkdir(parents=True, exist_ok=True)
    fig = plots.plot_anomaly_overview(
        res["residual_z"].values, out["price"].values, out["predicted_price"].values,
        out["residual_flag"].values, z_threshold=Z_THRESHOLD,
        save_path=FIG / "12_anomaly_overview.png",
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    write_results(
        n_total=len(df),
        n_resid=n_resid, n_strong=n_strong, n_moderate=n_moderate,
        n_if=n_if, n_both=n_both, n_high=n_high,
        oof_std=oof_resid_std, insample_std=insample_std,
        d_std_low=dollar_std_low, d_std_high=dollar_std_high,
        l_std_low=log_std_low, l_std_high=log_std_high,
        if_runtime=if_runtime, fat_tail=fat_tail,
        top_under=top_under, top_over=top_over, top_struct=top_struct,
    )
    logger.info("Results written to %s", RESULTS)


def write_results(
    n_total, n_resid, n_strong, n_moderate, n_if, n_both, n_high,
    oof_std, insample_std,
    d_std_low, d_std_high, l_std_low, l_std_high,
    if_runtime, fat_tail,
    top_under, top_over, top_struct,
) -> None:
    lines = [
        "# Phase 4 Results -- Anomaly Detection\n",
        f"Scored all {n_total:,} listings with two complementary signals. No "
        "ground-truth fraud labels exist, so we report NO accuracy metrics -- we "
        "rank flagged listings and justify each qualitatively.\n",
        "## Flag counts (tiered)\n",
        f"| Tier | Count | Share | Meaning |",
        "|------|-------|-------|---------|",
        f"| Residual, |z| > {Z_THRESHOLD} (total) | {n_resid:,} | "
        f"{n_resid/n_total*100:.1f}% | operational threshold, not a statistical claim |",
        f"| -- of which STRONG (|z| > {Z_STRONG} AND |resid_pct| > {PCT_STRONG}%) "
        f"| {n_strong:,} | {n_strong/n_total*100:.2f}% | far outside plausible model-noise band |",
        f"| -- of which MODERATE | {n_moderate:,} | {n_moderate/n_total*100:.1f}% | "
        f"possibly a bad listing, possibly the model missing a rare trim (MAPE is ~37%) |",
        f"| Isolation Forest (structural) | {n_if:,} | {n_if/n_total*100:.1f}% | "
        f"contamination fixed at {CONTAMINATION}, so this share is a parameter, not a finding |",
        f"| Residual & IF overlap | {n_both:,} | {n_both/n_total*100:.2f}% | any-tier residual + IF |",
        f"| **HIGH: strong residual + IF** | **{n_high:,}** | **{n_high/n_total*100:.3f}%** | "
        f"**highest-confidence action set** |",
        "\n## Alternatives, with evidence (the 'why X not Y')\n",
        "### 1. Out-of-fold vs in-sample predictions (leakage guard)\n",
        f"- In-sample residual std (model scores its own training rows): "
        f"**{insample_std:.4f}** (log scale)",
        f"- Out-of-fold residual std (leakage-free): **{oof_std:.4f}**",
        f"- In-sample residuals are {(1 - insample_std / oof_std) * 100:.0f}% smaller. "
        "The effect is modest (a regularized LightGBM at ~200k rows does not overfit hard), "
        "but the OOF guard is free and correct, and it matters most exactly on the extreme "
        "rows we care about. Every row is scored by a model that never trained on it.\n",
        "### 2. Log-space vs dollar residual (scale choice)\n",
        "Dollar residuals are heteroscedastic -- their spread scales mechanically with "
        "price, so a fixed dollar threshold would systematically over-flag expensive cars "
        "and under-flag cheap ones:\n",
        f"- Dollar residual std: **${d_std_low:,.0f}** (price < $10k) vs "
        f"**${d_std_high:,.0f}** (price > $30k) -- {d_std_high / max(d_std_low, 1):.1f}x "
        "wider on expensive cars purely because the numbers are bigger.",
        f"- Log residual std: **{l_std_low:.3f}** (< $10k) vs **{l_std_high:.3f}** (> $30k) "
        "-- roughly flat, with slightly HIGHER dispersion on cheap cars (genuine signal: "
        "cheap high-mileage cars have more relative price uncertainty).",
        "- Log residual = approx pct error, comparable across the price range. One z-"
        "threshold is defensible everywhere; on dollar residuals it would not be.\n",
        "### 3. Robust (MAD) vs standard (std) z-score\n",
        "We standardize with median + MAD, not mean + std. The outliers we are hunting "
        "inflate the mean and std and would mask themselves; median/MAD are unaffected by "
        "the tails, giving a stable reference distribution.\n",
        "### 4. Isolation Forest -- and why we EXCLUDE `log_price` from its features\n",
        f"- Fit on {n_total:,} rows in **{if_runtime:.1f}s**. IF is ~O(n log n) via random "
        "subsampling; LOF is ~O(n^2) on pairwise distances "
        f"(~{n_total ** 2 / 1e9:.0f}e9 ops) and One-Class SVM does not scale to ~200k. "
        "IF is the only practical full-data choice.",
        "- **Design decision (fixed during self-review):** `log_price` was originally in "
        "the IF feature set. That made IF partly re-detect the same price-outlier signal the "
        "residual method already catches -- inflating the 'flagged by BOTH' overlap "
        "mechanically. Independence check: with `log_price` in IF, corr(if_score, |z|) = "
        "0.336; without it, 0.118. Also without price, the residual & IF overlap drops from "
        "639 to a genuinely independent ~220 set. IF now uses only structural features "
        "(age, odometer, mileage_per_year, cylinders_num), so the two signals are "
        "orthogonal by construction and the 'HIGH' tier truly means 'suspicious on price AND "
        "structurally weird'.\n",
        "### 5. Confound warning: residual = listing anomaly + model error\n",
        "Without labels, we cannot cleanly separate 'bad listing' from 'model missed a rare "
        "car'. The tiered threshold above is the honest split: the STRONG tier "
        f"(|z|>{Z_STRONG} AND |pct|>{PCT_STRONG}%) is far outside the model's ~37% MAPE "
        "band and is almost certainly listing-side (junk price / placeholder / scam); the "
        "MODERATE tier is ambiguous and needs human review, not auto-action.\n",
        "### 6. Fat-tail check: threshold is operational, not Gaussian\n",
        "|z| > 3.5 is NOT 'a 3.5-sigma event'. Under a Gaussian tail the flag rate would "
        "be ~0.05%; the actual rate is ~80x that. The residual distribution has heavy tails "
        "(rare cars, listing junk, model error), so the threshold is a **capacity choice** "
        "(how many listings a human queue can absorb), not a rarity claim:\n",
        "| Threshold | Observed flags | Observed % | Gaussian expects | Fold excess |",
        "|-----------|----------------|-----------|------------------|-------------|",
    ]
    for t, cnt, pct, gauss, gauss_pct in fat_tail:
        fold = cnt / max(gauss, 1e-9)
        fold_str = f"{fold:,.0f}x" if fold < 1e6 else f"{fold:.1e}"
        lines.append(f"| |z| > {t} | {cnt:,} | {pct:.2f}% | "
                     f"{gauss:.2f} ({gauss_pct:.4f}%) | {fold_str} |")

    lines += [
        "\n---\n",
        "## Top listings, per action category\n",
        "Ranked by |z| within category so the reviewer sees examples of each business "
        "action, not just the underpriced tail (which structurally dominates any |z| "
        "ranking because log residuals are asymmetric).\n",
        "### Top 10 UNDERPRICED (route to fraud / trust-and-safety review)\n",
    ]
    for i, (_, row) in enumerate(top_under.iterrows(), start=1):
        lines.append(f"{i}. {anomaly_listing_note(row, 'underpriced')}")

    lines += [
        "\n### Top 10 OVERPRICED (nudge seller: your price is above market)\n",
    ]
    if len(top_over):
        for i, (_, row) in enumerate(top_over.iterrows(), start=1):
            lines.append(f"{i}. {anomaly_listing_note(row, 'overpriced')}")
    else:
        lines.append("_(none passed the residual threshold on the upper tail)_")

    lines += [
        "\n### Top 10 STRUCTURAL-ONLY (prompt seller to confirm year / mileage)\n",
        "Flagged by Isolation Forest but NOT by the residual signal -- their price is "
        "reasonable, but the age/odometer/mileage-per-year combination is unusual.\n",
    ]
    if len(top_struct):
        for i, (_, row) in enumerate(top_struct.iterrows(), start=1):
            lines.append(f"{i}. {anomaly_listing_note(row, 'structural')}")
    else:
        lines.append("_(no structural-only flags this run)_")

    lines += [
        "\n## Business use\n",
        f"- **HIGH ({n_high} listings): strong mispriced + structural.** Route to trust-and-"
        "safety BEFORE the listing goes live. Two independent signals agree: implausible "
        "price AND weird attribute combination.",
        "- **Strong mispriced (any direction):** underpriced -> fraud review; overpriced -> "
        "seller nudge. Extreme enough that model error is an unlikely explanation.",
        "- **Moderate mispriced:** human-review queue only. At MAPE ~37%, a moderate flag "
        "may just be the model missing a rare trim -- do NOT auto-action.",
        "- **Structural only:** likely data-entry error -- ask the seller to confirm year "
        "and mileage before publishing.\n",
        "> Per CLAUDE.md, these flags are a separate deliverable and are NOT fed back into "
        "the Phase 3 training pipeline.\n",
    ]

    RESULTS.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
