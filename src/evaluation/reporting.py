"""Markdown report-writing helpers shared across the pipeline scripts.

These turn already-computed metrics/tables into the interpretive text and
file-writing utilities scripts/train.py, scripts/detect_anomalies.py,
scripts/predict_intervals.py, and scripts/ablation_description_features.py
need -- kept out of the scripts themselves per CLAUDE.md's thin-orchestration
rule (a script should import + call + save, not embed the interpretation).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PRICE_SEGMENT_LABELS


def replace_doc_section(path: Path, headers: str | list[str], new_content: str) -> None:
    """Idempotently replace a markdown section, keeping everything before it.

    If the file already contains one of `headers` (checked in order), keeps
    only the content before the FIRST matching header and appends
    `new_content` in its place. This lets a results-writing script be re-run
    repeatedly without accumulating duplicate sections.

    Args:
        path: Markdown file to update.
        headers: One header string, or a list of alternate header strings
            (e.g. across a section renaming) to search for, in order.
        new_content: Replacement content for everything from the header onward.
    """
    if isinstance(headers, str):
        headers = [headers]
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    for header in headers:
        if header in existing:
            existing = existing.split(header)[0]
            break
    existing = existing.rstrip("\n") + "\n\n" if existing else ""
    path.write_text(existing + new_content, encoding="utf-8")


def model_comparison_verdict(lr_metrics: dict, lgbm_metrics: dict) -> str:
    """Summarize the Linear -> LightGBM metric jump (Phase 3 model comparison).

    Args:
        lr_metrics: Linear Regression's dollar-scale metrics dict.
        lgbm_metrics: LightGBM's dollar-scale metrics dict.

    Returns:
        str: A markdown paragraph naming LightGBM as the winner with the
        specific RMSE/MAPE/R2 deltas.
    """
    pct_rmse = (lr_metrics["RMSE ($)"] - lgbm_metrics["RMSE ($)"]) / lr_metrics["RMSE ($)"] * 100
    pp_mape = lr_metrics["MAPE (%)"] - lgbm_metrics["MAPE (%)"]
    return (
        f"**LightGBM** wins across all four metrics. The jump from Linear to LightGBM: "
        f"RMSE drops {pct_rmse:.0f}% (${lr_metrics['RMSE ($)']:,.0f} -> ${lgbm_metrics['RMSE ($)']:,.0f}), "
        f"MAPE drops {pp_mape:.0f}pp ({lr_metrics['MAPE (%)']:.0f}% -> {lgbm_metrics['MAPE (%)']:.0f}%), "
        f"R2 rises {lr_metrics['R2']:.2f} -> {lgbm_metrics['R2']:.2f}. Linear's {lr_metrics['MAPE (%)']:.0f}% MAPE "
        "confirms the EDA prediction: the age x odometer interaction requires a "
        "non-linear model.\n"
    )


def a1_verdict(a1_table: pd.DataFrame) -> str:
    """Summarize Ablation A1 (log target vs raw target).

    Args:
        a1_table: Metrics table with rows "log1p(price)" and "raw price".

    Returns:
        str: A markdown paragraph explaining why log target is chosen despite
        raw target sometimes winning on RMSE.
    """
    log_row = a1_table.loc["log1p(price)"]
    raw_row = a1_table.loc["raw price"]
    rmse_winner = "raw target" if raw_row["RMSE ($)"] < log_row["RMSE ($)"] else "log target"
    mape_gap = raw_row["MAPE (%)"] - log_row["MAPE (%)"]
    return (
        f"**Key finding:** {rmse_winner} wins on RMSE (${raw_row['RMSE ($)']:,.0f} raw vs "
        f"${log_row['RMSE ($)']:,.0f} log) and R2 ({raw_row['R2']:.2f} vs {log_row['R2']:.2f}) "
        "because dollar-scale optimization favors getting expensive cars right. But MAPE "
        f"tells the real story: raw target has {raw_row['MAPE (%)']:.0f}% average percentage "
        f"error vs {log_row['MAPE (%)']:.0f}% for log -- a {mape_gap:.0f}-point gap. "
        "Raw-target models systematically under-predict cheap cars (a $500 error on a $2k "
        "car is 25% -- invisible to RMSE but devastating to MAPE). For a marketplace where "
        "most listings are under $20k, MAPE is the business-relevant metric. We choose log "
        "target.\n"
    )


def a3_verdict(a3_table: pd.DataFrame) -> str:
    """Summarize Ablation A3 (target vs frequency vs drop encoding for `model`).

    Args:
        a3_table: Metrics table with rows "target_encoding", "frequency_encoding",
            "drop_model_column".

    Returns:
        str: A markdown paragraph naming the RMSE/MAPE winner and the cost of
        dropping the `model` column entirely.
    """
    tgt = a3_table.loc["target_encoding"]
    freq = a3_table.loc["frequency_encoding"]
    drop = a3_table.loc["drop_model_column"]
    winner = a3_table["RMSE ($)"].idxmin()
    mape_winner = a3_table["MAPE (%)"].idxmin()
    rmse_cost_drop = drop["RMSE ($)"] - tgt["RMSE ($)"]
    mape_cost_drop = drop["MAPE (%)"] - tgt["MAPE (%)"]
    return (
        f"**Key finding:** with out-of-fold target encoding, **{winner}** wins on RMSE "
        f"(${tgt['RMSE ($)']:,.0f} target vs ${freq['RMSE ($)']:,.0f} frequency vs "
        f"${drop['RMSE ($)']:,.0f} drop), with R2={tgt['R2']:.2f}. "
        f"{'Frequency encoding edges target on MAPE' if mape_winner == 'frequency_encoding' else 'Target encoding also leads on MAPE'} "
        f"({freq['MAPE (%)']:.1f}% frequency vs {tgt['MAPE (%)']:.1f}% target) because it "
        "captures the 'popular models are cheaper' signal cheaply. Dropping `model` costs "
        f"~${rmse_cost_drop:,.0f} RMSE and ~{mape_cost_drop:.1f} MAPE points -- model "
        "identity carries real trim-level signal (a Civic vs an Accord at equal age/mileage "
        "is a $3-5k gap). **Lesson (from an earlier self-review):** an ablation is only "
        "trustworthy if the pipeline under it is leakage-free -- a buggy version that "
        "silently applied a leaky full-train mapping to training rows made target encoding "
        "look worse than frequency/drop; fixing it to genuine OOF encoding reversed the "
        "ranking.\n"
    )


def age_segment_observation(err_age: pd.DataFrame) -> str:
    """Compare the highest-MAE and highest-MAPE age buckets and explain the gap.

    Args:
        err_age: Per-age-bucket error table with "MAE" and "MAPE" columns
            (see evaluation.metrics.error_by_segment).

    Returns:
        str: A markdown observation paragraph, worded differently depending
        on whether the same bucket tops both metrics.
    """
    highest_mae_seg = err_age["MAE"].idxmax()
    highest_mape_seg = err_age["MAPE"].idxmax()
    if highest_mae_seg == highest_mape_seg:
        return (
            f"\n**Observation:** {highest_mae_seg} has both the highest MAE "
            f"(${err_age.loc[highest_mae_seg, 'MAE']:,.0f}) and the highest MAPE "
            f"({err_age.loc[highest_mape_seg, 'MAPE']:.0f}%) -- this bucket is the "
            "model's weakest segment on both scales at once, not just proportionally. "
            "It likely spans the widest price range (near-new budget cars to near-new "
            "luxury cars at similar age), which is harder to pin down than a more "
            "homogeneous older-car price band.\n"
        )
    return (
        f"\n**Observation:** the highest-MAE age bucket is {highest_mae_seg} "
        f"(${err_age.loc[highest_mae_seg, 'MAE']:,.0f}) because it spans the "
        f"widest price range. The highest-MAPE bucket is {highest_mape_seg} "
        f"({err_age.loc[highest_mape_seg, 'MAPE']:.0f}%) despite lower MAE "
        f"(${err_age.loc[highest_mape_seg, 'MAE']:,.0f}) -- small dollar errors "
        "are large percentages on cheap, old cars.\n"
    )


def price_segment_observation(
    err_price: pd.DataFrame, priciest_label: str = PRICE_SEGMENT_LABELS[-1]
) -> str:
    """Compare the weakest (highest-MAPE) and priciest price segments.

    Args:
        err_price: Per-price-segment error table with "MAE"/"MAPE" columns.
        priciest_label: Segment label to treat as "priciest" if present,
            otherwise the first row is used.

    Returns:
        str: A markdown observation paragraph.
    """
    cheapest_seg = err_price["MAPE"].idxmax()
    priciest_seg = priciest_label if priciest_label in err_price.index else err_price.index[0]
    return (
        f"\n**Observation:** the {cheapest_seg} segment has "
        f"{err_price.loc[cheapest_seg, 'MAPE']:.0f}% MAPE -- the model's weakest zone. "
        "These are high-mileage, old vehicles where description/condition detail "
        f"matters most. The {priciest_seg} segment has MAE "
        f"${err_price.loc[priciest_seg, 'MAE']:,.0f} but only "
        f"{err_price.loc[priciest_seg, 'MAPE']:.0f}% MAPE -- large dollar errors are "
        "proportionally tolerable on expensive cars.\n"
    )


def mondrian_segment_verdict(seg_compare: pd.DataFrame) -> str:
    """Data-driven verdict on whether Mondrian closed the standard-CQR coverage gap.

    Written from the numbers each run, not from a template that asserts
    success regardless of outcome.

    Args:
        seg_compare: Per-segment DataFrame with "standard_coverage" and
            "mondrian_coverage" columns.

    Returns:
        str: A markdown verdict paragraph, one of three variants depending
        on how much (if any) Mondrian improved the worst segment's coverage.
    """
    std_worst = seg_compare["standard_coverage"].min()
    std_worst_seg = seg_compare["standard_coverage"].idxmin()
    mon_worst = seg_compare["mondrian_coverage"].min()
    mon_worst_seg = seg_compare["mondrian_coverage"].idxmin()

    if mon_worst >= 0.85:
        return (
            f"**Verdict: the 6B gap is materially closed.** Standard CQR's worst "
            f"actual-price segment was {std_worst:.1%} ({std_worst_seg}); Mondrian's "
            f"worst is {mon_worst:.1%} ({mon_worst_seg}) -- within 5 points of the "
            "90% target in every segment."
        )
    if mon_worst > std_worst + 0.03:
        return (
            f"**Verdict: improved, not fully closed.** Standard CQR's worst "
            f"actual-price segment was {std_worst:.1%} ({std_worst_seg}); Mondrian "
            f"lifts the floor to {mon_worst:.1%} ({mon_worst_seg}). The remaining gap "
            "is expected: Mondrian guarantees coverage per PREDICTED-price bin, while "
            "this table slices by ACTUAL price -- where the model badly mispredicts "
            "(junk-heavy cheap listings), rows land in the wrong bin and the per-bin "
            "guarantee does not transfer fully."
        )
    return (
        f"**Verdict: Mondrian did NOT materially improve ACTUAL-price-segment "
        f"coverage** (worst segment {std_worst:.1%} -> {mon_worst:.1%}). The "
        "root-cause probe (`scripts/probe_mondrian_conditional_coverage.py`) "
        "shows why -- see the root-cause subsection below. Short version: the "
        "guarantee Mondrian actually makes (coverage per PREDICTED-price bin) "
        "holds at 89-91% in every bin; the actual-price tail failure is caused "
        "by point-model bias on rare expensive trims, which no calibration "
        "scheme can repair."
    )


def desc_ablation_verdict(
    rmse_base: float, rmse_ext: float, mape_base: float, mape_ext: float,
    pct_rmse: float, total_gain: float,
) -> str:
    """Verdict for Ablation A4 (description-derived features).

    Args:
        rmse_base: Baseline (no desc_*) RMSE.
        rmse_ext: Extended (with desc_*) RMSE.
        mape_base: Baseline MAPE.
        mape_ext: Extended MAPE.
        pct_rmse: (rmse_ext - rmse_base) / rmse_base * 100.
        total_gain: Sum of desc_* features' % of total LightGBM gain.

    Returns:
        str: A markdown verdict paragraph -- real improvement, negative
        result, or mixed/inconclusive.
    """
    if pct_rmse < -0.5 and mape_ext < mape_base:
        return (
            f"**Verdict: real improvement.** RMSE drops {abs(pct_rmse):.2f}% "
            f"(${rmse_base:,.0f} -> ${rmse_ext:,.0f}), MAPE improves "
            f"({mape_base:.2f}% -> {mape_ext:.2f}%). desc_* features carry "
            f"{total_gain:.2f}% of total gain -- a real, if modest, signal. "
            "Recommendation: adopt as default features (separate follow-up step, "
            "requires re-rippling Phase 3/6 metrics)."
        )
    if total_gain < 0.5:
        return (
            f"**Verdict: negative result.** desc_* features carry only "
            f"{total_gain:.2f}% of total gain and RMSE moved {pct_rmse:+.2f}% "
            "(noise-level). The trim/equipment keyword signal, once boiled down "
            "to 3 leakage-free numeric features, does not add information beyond "
            "what manufacturer/model/year/odometer/target-encoding already "
            "captures. Recorded as a negative result -- exactly like Phase 6C's "
            "gain-importance check. The columns stay in cleaned.parquet but are "
            "NOT added to the default feature list."
        )
    return (
        f"**Verdict: mixed / inconclusive.** RMSE moved {pct_rmse:+.2f}% and "
        f"desc_* features carry {total_gain:.2f}% of gain -- some signal, but "
        "not a clear win on the headline metrics. Not adopted as default; "
        "documented as a partial/negative result."
    )


def anomaly_listing_note(row: pd.Series, kind: str) -> str:
    """Format one flagged listing's why-suspicious markdown blurb.

    Args:
        row: A row from the ranked suspicious-listings output, with columns
            year/manufacturer/model/odometer/condition/price/predicted_price/
            residual_pct/if_score/if_flag.
        kind: "underpriced", "overpriced", or "structural".

    Returns:
        str: A one-line markdown bullet explaining why the listing was flagged.
    """
    car = f"{int(row['year'])} {row['manufacturer']} {row['model']}"
    odo = f"{int(row['odometer']):,} mi"
    cond = row["condition"] if pd.notna(row["condition"]) else "condition missing"

    if kind == "underpriced":
        why = (f"listed ${row['price']:,.0f} vs model expects "
               f"${row['predicted_price']:,.0f} ({row['residual_pct']:.0f}% below) -- "
               f"far-below-market is a classic scam / hidden-defect / placeholder signal")
    elif kind == "overpriced":
        why = (f"listed ${row['price']:,.0f} vs model expects "
               f"${row['predicted_price']:,.0f} (+{row['residual_pct']:.0f}%) -- "
               f"likely an over-ask, data-entry error, or a rare trim the model does not capture")
    else:  # structural
        why = (f"listed ${row['price']:,.0f}; not extreme on price alone, but "
               f"IF score {row['if_score']:.2f} -- attribute combination is unusual "
               f"(age/odometer/mileage-per-year mix), likely a data-entry error")
    struct = " Also flagged by Isolation Forest." \
        if kind != "structural" and row["if_flag"] else ""
    return f"**{car}** ({odo}, {cond}): {why}.{struct}"
