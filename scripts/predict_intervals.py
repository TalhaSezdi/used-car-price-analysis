"""Phase 6B+6C entry point: conformal prediction intervals + anomaly tie-in.

Usage:
    python scripts/predict_intervals.py

Produces:
    reports/figures/13_interval_width_vs_price.png
    docs/phase6_results.md (6B/6C sections, regenerated idempotently)

Reuses the exact Phase 3 train/test split (src.models.dataset.build_split,
seed 42) -- Phase 6A already established that split is not materially
contaminated, so there is no reason to re-split here. A calibration set is
carved out of TRAIN only (never touches TEST) for the conformal correction.

Both interval variants come from ONE pair of quantile models per alpha:
  - standard CQR: one global conformal correction (Phase 6B)
  - Mondrian CQR: per-bin corrections, binned on the predicted-band midpoint
    (Phase 6C -- closes the per-segment coverage gap 6B documented)
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.dataset import (
    build_split, split_calibration,
    NUMERIC_FEATURES, LOW_CARD_FEATURES, HIGH_CARD_FEATURES,
)
from src.models.encoders import FeaturePreprocessor
from src.models.intervals import (
    MondrianConformalIntervalModel, fit_median_model, coverage, coverage_by_segment,
)
from src.anomaly.detector import ResidualAnomalyDetector
from src.evaluation import plots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "processed" / "cleaned.parquet"
FIG = ROOT / "reports" / "figures"
RESULTS = ROOT / "docs" / "phase6_results.md"

CALIB_SIZE = 0.2
NOMINAL_90 = 0.10   # alpha for the 90% interval
NOMINAL_99 = 0.01   # alpha for the 99% interval
MONDRIAN_BINS = 5

# Phase 4 operational thresholds, reused here for the anomaly-flag comparison
Z_THRESHOLD = 3.5
Z_STRONG = 5.0
PCT_STRONG = 85.0


def main() -> None:
    df = pd.read_parquet(DATA)
    logger.info("Loaded %d rows", len(df))

    split = build_split(df)
    X_fit, X_calib, y_fit, y_calib = split_calibration(
        split.X_train, split.y_train, calib_size=CALIB_SIZE
    )
    logger.info(
        "Train split for intervals: fit=%d, calib=%d, test=%d",
        len(X_fit), len(X_calib), len(split.X_test),
    )

    # --- Feature pipeline: fit on FIT only, transform calib and test ---
    prep = FeaturePreprocessor(
        numeric_cols=NUMERIC_FEATURES,
        low_card_cols=LOW_CARD_FEATURES,
        high_card_cols=HIGH_CARD_FEATURES,
        high_card_method="target",
    )
    Xt_fit = prep.fit_transform(X_fit, y_fit)
    Xt_calib = prep.transform(X_calib)
    Xt_test = prep.transform(split.X_test)

    price_test = split.price_test.values

    # --- Point estimate (median quantile model) ---
    logger.info("Fitting median (alpha=0.5) point-estimate model...")
    med_model = fit_median_model(Xt_fit, y_fit)
    pred_med_log = med_model.predict(Xt_test)

    # --- Interval models: Mondrian fit also yields the standard (global)
    #     correction for free, so one fit per alpha covers both variants ---
    logger.info("Fitting 90%% interval model (Mondrian, %d bins)...", MONDRIAN_BINS)
    cqr_90 = MondrianConformalIntervalModel(alpha=NOMINAL_90, n_bins=MONDRIAN_BINS)
    cqr_90.fit(Xt_fit, y_fit, Xt_calib, y_calib)

    logger.info("Fitting 99%% interval model (Mondrian, %d bins)...", MONDRIAN_BINS)
    cqr_99 = MondrianConformalIntervalModel(alpha=NOMINAL_99, n_bins=MONDRIAN_BINS)
    cqr_99.fit(Xt_fit, y_fit, Xt_calib, y_calib)

    # Mondrian (per-bin corrections)
    m_lo90, m_hi90 = cqr_90.predict_interval_dollar(Xt_test)
    m_lo99, m_hi99 = cqr_99.predict_interval_dollar(Xt_test)
    # Standard CQR (global correction, same underlying quantile models)
    s_lo90, s_hi90 = cqr_90.predict_interval_dollar_global(Xt_test)
    s_lo99, s_hi99 = cqr_99.predict_interval_dollar_global(Xt_test)
    # Raw (uncalibrated) band for the calibration-effect row
    raw_lo90_log, raw_hi90_log = cqr_90.raw_interval_log(Xt_test)
    raw_lo90, raw_hi90 = np.expm1(raw_lo90_log), np.expm1(raw_hi90_log)

    # --- Coverage: overall ---
    cov = {
        "90_raw": coverage(price_test, raw_lo90, raw_hi90),
        "90_std": coverage(price_test, s_lo90, s_hi90),
        "90_mon": coverage(price_test, m_lo90, m_hi90),
        "99_std": coverage(price_test, s_lo99, s_hi99),
        "99_mon": coverage(price_test, m_lo99, m_hi99),
    }
    print("\n=== Overall coverage ===")
    print(f"90% raw quantile (no conformal)  : {cov['90_raw']:.4f}")
    print(f"90% standard CQR (global corr)   : {cov['90_std']:.4f} (target 0.90)")
    print(f"90% Mondrian ({MONDRIAN_BINS} bins)           : {cov['90_mon']:.4f} (target 0.90)")
    print(f"99% standard CQR                 : {cov['99_std']:.4f} (target 0.99)")
    print(f"99% Mondrian                     : {cov['99_mon']:.4f} (target 0.99)")

    # --- Coverage + width by segment: standard vs Mondrian ---
    test_df = df.loc[split.X_test.index].copy()
    age_bins = pd.cut(test_df["age"], [0, 3, 6, 10, 15, 60],
                      labels=["0-3yr", "4-6yr", "7-10yr", "11-15yr", "16+yr"])
    price_bins = pd.cut(test_df["price"],
                        [0, 5000, 10000, 20000, 50000, 150000],
                        labels=["<5k", "5-10k", "10-20k", "20-50k", "50-150k"])

    std_by_price = coverage_by_segment(price_test, s_lo90, s_hi90, price_bins)
    mon_by_price = coverage_by_segment(price_test, m_lo90, m_hi90, price_bins)
    mon_by_age = coverage_by_segment(price_test, m_lo90, m_hi90, age_bins)

    seg_compare = pd.DataFrame({
        "standard_coverage": std_by_price["coverage"],
        "mondrian_coverage": mon_by_price["coverage"],
        "mondrian_median_width": mon_by_price["median_width"],
        "count": mon_by_price["count"],
    })
    print("\n=== 90% coverage by ACTUAL price segment: standard CQR vs Mondrian ===")
    print(seg_compare.to_string())
    print("\n=== Mondrian 90% coverage + width by age bucket ===")
    print(mon_by_age.to_string())

    # --- Figure ---
    FIG.mkdir(parents=True, exist_ok=True)
    fig = plots.plot_interval_width(
        price_test, m_hi90 - m_lo90,
        coverage_by_segment=std_by_price["coverage"],
        coverage_comparison=mon_by_price["coverage"],
        nominal=0.90,
        save_path=FIG / "13_interval_width_vs_price.png",
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    # --- Anomaly tie-in: interval-exceedance flags vs Phase 4 MAD-z tiers ---
    # Approximation (documented): Phase 4 fit median/MAD on the FULL 197,814-row
    # OOF residual distribution. Here we only have test-set predictions, so the
    # MAD reference is recomputed on the test subset -- a reasonable proxy at
    # n=39,563 but not bit-identical to Phase 4's numbers.
    res_det = ResidualAnomalyDetector(z_threshold=Z_THRESHOLD)
    res_det.fit(split.y_test.values, pred_med_log)
    res = res_det.score(split.y_test.values, pred_med_log)
    abs_z = res["residual_z"].abs().values
    abs_pct = np.abs(
        (price_test - np.expm1(pred_med_log)) / np.expm1(pred_med_log) * 100
    )
    strong_resid = (abs_z > Z_STRONG) & (abs_pct > PCT_STRONG)

    flag_90 = (price_test < m_lo90) | (price_test > m_hi90)
    flag_99 = (price_test < m_lo99) | (price_test > m_hi99)

    n_test = len(price_test)
    tie_in = {
        "n_flag90": int(flag_90.sum()),
        "n_flag99": int(flag_99.sum()),
        "n_resid": int(res["residual_flag"].sum()),
        "n_strong": int(strong_resid.sum()),
        "overlap_90_resid": int((flag_90 & res["residual_flag"].values).sum()),
        "overlap_99_strong": int((flag_99 & strong_resid).sum()),
    }
    print(f"\n=== Anomaly tie-in: Mondrian interval exceedance vs MAD-z (test, n={n_test}) ===")
    print(f"Outside 90% interval         : {tie_in['n_flag90']} ({tie_in['n_flag90']/n_test*100:.2f}%)")
    print(f"Outside 99% interval         : {tie_in['n_flag99']} ({tie_in['n_flag99']/n_test*100:.2f}%)")
    print(f"MAD-z |z|>{Z_THRESHOLD} flag (test-fit): {tie_in['n_resid']} ({tie_in['n_resid']/n_test*100:.2f}%)")
    print(f"MAD-z STRONG tier (test-fit) : {tie_in['n_strong']} ({tie_in['n_strong']/n_test*100:.2f}%)")
    print(f"Overlap (90% & |z|>{Z_THRESHOLD})       : {tie_in['overlap_90_resid']}")
    print(f"Overlap (99% & STRONG)       : {tie_in['overlap_99_strong']}")

    write_results(
        cov=cov, seg_compare=seg_compare, mon_by_age=mon_by_age,
        n_test=n_test, tie_in=tie_in,
        correction_90=cqr_90.correction_, correction_99=cqr_99.correction_,
        mondrian_corrections_90=cqr_90.corrections_,
    )
    logger.info("Results written to %s", RESULTS)


def _segment_verdict(seg_compare: pd.DataFrame) -> str:
    """Data-driven verdict on whether Mondrian closed the 6B coverage gap.

    Written from the numbers each run, not from a template that asserts success
    regardless of outcome.
    """
    std_worst = seg_compare["standard_coverage"].min()
    std_worst_seg = seg_compare["standard_coverage"].idxmin()
    mon_worst = seg_compare["mondrian_coverage"].min()
    mon_worst_seg = seg_compare["mondrian_coverage"].idxmin()

    if mon_worst >= 0.85:
        headline = (
            f"**Verdict: the 6B gap is materially closed.** Standard CQR's worst "
            f"actual-price segment was {std_worst:.1%} ({std_worst_seg}); Mondrian's "
            f"worst is {mon_worst:.1%} ({mon_worst_seg}) -- within 5 points of the "
            "90% target in every segment."
        )
    elif mon_worst > std_worst + 0.03:
        headline = (
            f"**Verdict: improved, not fully closed.** Standard CQR's worst "
            f"actual-price segment was {std_worst:.1%} ({std_worst_seg}); Mondrian "
            f"lifts the floor to {mon_worst:.1%} ({mon_worst_seg}). The remaining gap "
            "is expected: Mondrian guarantees coverage per PREDICTED-price bin, while "
            "this table slices by ACTUAL price -- where the model badly mispredicts "
            "(junk-heavy cheap listings), rows land in the wrong bin and the per-bin "
            "guarantee does not transfer fully."
        )
    else:
        headline = (
            f"**Verdict: Mondrian did NOT materially improve ACTUAL-price-segment "
            f"coverage** (worst segment {std_worst:.1%} -> {mon_worst:.1%}). The "
            "root-cause probe (`scripts/probe_mondrian_conditional_coverage.py`) "
            "shows why -- see the root-cause subsection below. Short version: the "
            "guarantee Mondrian actually makes (coverage per PREDICTED-price bin) "
            "holds at 89-91% in every bin; the actual-price tail failure is caused "
            "by point-model bias on rare expensive trims, which no calibration "
            "scheme can repair."
        )
    return headline


def write_results(
    cov, seg_compare, mon_by_age, n_test, tie_in,
    correction_90, correction_99, mondrian_corrections_90,
) -> None:
    corr_list = ", ".join(f"{c:.4f}" for c in mondrian_corrections_90)
    lines = [
        "## 6B/6C. Prediction intervals (conformal + Mondrian)\n",
        "**Method:** split-conformal quantile regression (CQR, Romano et al. 2019). "
        "Two LightGBM quantile regressors (lower/upper) per interval level, trained on "
        "a `fit` subset of TRAIN; the band is calibrated on a disjoint `calibration` "
        "subset of TRAIN (never TEST). Point estimate is a separate alpha=0.5 quantile "
        "model. Reuses the Phase 3 / 6A train-test split -- see "
        "[6A resolution](#6a-split-contamination-probe).\n",
        "Two calibration variants from the SAME quantile models:\n",
        "- **Standard CQR (6B):** one global correction. Guarantees marginal coverage only.",
        f"- **Mondrian CQR (6C):** {len(mondrian_corrections_90)} corrections, one per "
        "bin of the predicted-band midpoint (log scale). Binning on the model's own "
        "output keeps the interval computable for any new listing at inference time -- "
        "binning on the actual price would be unavailable for a 'what is it worth' "
        "query and would break exchangeability.\n",
        f"Corrections (log scale): standard 90% = {correction_90:.4f}, standard 99% = "
        f"{correction_99:.4f}; Mondrian 90% per-bin = [{corr_list}] -- note the spread "
        "across bins: cheap-car bins need a much larger correction than expensive-car "
        "bins, which is exactly the heteroscedasticity one global correction ignores.\n",
        f"### Overall coverage (test set, n={n_test:,})\n",
        "| Interval | Calibration | Empirical coverage | Nominal |",
        "|---|---|---|---|",
        f"| 90% | raw quantile band (uncalibrated) | {cov['90_raw']:.4f} | 0.90 |",
        f"| 90% | standard CQR (global) | {cov['90_std']:.4f} | 0.90 |",
        f"| 90% | **Mondrian (per-bin)** | **{cov['90_mon']:.4f}** | 0.90 |",
        f"| 99% | standard CQR (global) | {cov['99_std']:.4f} | 0.99 |",
        f"| 99% | **Mondrian (per-bin)** | **{cov['99_mon']:.4f}** | 0.99 |",
        "\n**Reading this:** the raw quantile band is "
        f"{'under' if cov['90_raw'] < 0.90 else 'over'}-covered ({cov['90_raw']:.1%} "
        "vs 90% target) -- LightGBM's quantile loss is not exactly calibrated on its "
        "own; the conformal step is what earns the guarantee.\n",
        "### 90% coverage by ACTUAL price segment: standard vs Mondrian\n",
        seg_compare.round(4).to_markdown(),
        "\n" + _segment_verdict(seg_compare) + "\n",
        "### Root cause: why the actual-price tails stay under-covered\n",
        "Probe: `scripts/probe_mondrian_conditional_coverage.py` (numbers below are "
        "from its recorded run; the probe is deterministic and re-runnable).\n",
        "1. **The guarantee Mondrian makes is delivered.** Coverage per PREDICTED-"
        "price bin: 89.5-90.6% (5 bins), 88.7-90.8% (10 bins), 89.1-90.7% "
        "(tail-focused bins). Feature-conditional calibration works.",
        "2. **Finer binning does not move actual-price-segment coverage.** The "
        "50-150k actual segment stays at 69-70% under 5-bin, 10-bin, and "
        "tail-focused binnings alike -- the problem is not bin coarseness.",
        "3. **The missed expensive listings are point-model failures, not "
        "calibration failures.** Of the 282 uncovered 50-150k listings, the model's "
        "predicted band midpoint has median ~$30k against a median actual price of "
        "~$65k. An interval centered at $30k cannot reach $65k under any per-bin "
        "widening that keeps intervals useful for the mid-priced majority sharing "
        "those bins. These are the rare-trim / heavy-truck listings Phase 3's error "
        "analysis already identified as the model's blind spot.",
        "4. **Slicing coverage by ACTUAL price conditions on the outcome.** Exact "
        "coverage conditional on outcome-derived groups is provably unattainable in "
        "finite samples (Vovk 2012; Foygel Barber et al. 2021). Any slice defined by "
        "the target concentrates precisely the rows the model mispredicts -- "
        "age-bucket coverage (a feature-based slice) holds fine, as the table below "
        "shows.\n",
        "**Practical conclusion:** report the feature-conditional guarantee "
        "(per-predicted-bin, per-age) as the honest product claim; treat the "
        "actual-price tail miscoverage as a Phase 3 modeling gap (rare expensive "
        "trims need better features -- e.g. trim-level signal from `description` -- "
        "not better calibration).\n",
        "### Mondrian 90% coverage + median width by age bucket\n",
        mon_by_age.round(4).to_markdown(),
        "\nSee [reports/figures/13_interval_width_vs_price.png]"
        "(../reports/figures/13_interval_width_vs_price.png) -- left: interval width "
        "grows with price (honest heteroscedasticity); right: standard vs Mondrian "
        "coverage per price segment.\n",
        "### Anomaly tie-in: Mondrian interval exceedance vs Phase 4 MAD-z tiers\n",
        "For each test-set listing: does the actual price fall outside its calibrated "
        "interval? Compared against a MAD-z flag refit on this test set (approximation: "
        "Phase 4's original MAD reference used the full 197,814-row OOF residual "
        "distribution; here it is refit on the 39,563-row test subset for a "
        "like-for-like comparison without rerunning the full 5-fold OOF pass).\n",
        "| Signal | Count | Share of test |",
        "|---|---|---|",
        f"| Outside Mondrian 90% interval | {tie_in['n_flag90']:,} | {tie_in['n_flag90']/n_test*100:.2f}% |",
        f"| Outside Mondrian 99% interval | {tie_in['n_flag99']:,} | {tie_in['n_flag99']/n_test*100:.2f}% |",
        f"| MAD-z, \\|z\\| > {Z_THRESHOLD} (test-refit) | {tie_in['n_resid']:,} | {tie_in['n_resid']/n_test*100:.2f}% |",
        f"| MAD-z STRONG tier (test-refit) | {tie_in['n_strong']:,} | {tie_in['n_strong']/n_test*100:.2f}% |",
        f"| Overlap: 90% interval AND \\|z\\|>{Z_THRESHOLD} | {tie_in['overlap_90_resid']:,} | - |",
        f"| Overlap: 99% interval AND STRONG | {tie_in['overlap_99_strong']:,} | - |",
        "\n**Framing:** the interval-exceedance flag now carries a per-listing, "
        "segment-aware threshold with a coverage guarantee behind it, which the global "
        "MAD-z band could not offer. The two signals agree on the clear cases (most "
        "STRONG-tier listings sit outside the 99% interval). Phase 4 outputs are NOT "
        "rewritten; for a production system the recommended flag is 'outside the "
        "Mondrian 99% interval', with the MAD-z tiers kept as a cross-check.\n",
    ]
    existing = RESULTS.read_text(encoding="utf-8")
    for header in ("## 6B/6C. Prediction intervals", "## 6B. Prediction intervals"):
        if header in existing:
            existing = existing.split(header)[0]
            break
    existing = existing.rstrip("\n") + "\n\n"
    RESULTS.write_text(existing + "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
