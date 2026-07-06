"""Phase 3 entry point: train models, run ablations, write results.

Workflow (three-way split, leakage-safe model selection):
    1. Fit Linear / RF / LightGBM on 60% train.
    2. Compare them on 20% validation. Ablations (A1, A2, A3) also on val.
       Every design choice is made on data the final test set has never seen.
    3. Refit the winning LightGBM on 80% (train + val).
    4. Report the final unbiased headline metric on 20% test.

Usage:
    python scripts/train.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.dataset import (
    build_split, NUMERIC_FEATURES, LOW_CARD_FEATURES, HIGH_CARD_FEATURES,
)
from src.models.encoders import FeaturePreprocessor
from src.models.train import (
    train_linear, train_rf, train_lgbm,
    ablation_a1_raw_vs_log, ablation_a2_collinearity,
)
from src.evaluation.metrics import metrics_table, error_by_segment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "processed" / "cleaned.parquet"
RESULTS = ROOT / "docs" / "phase3_results.md"


def main() -> None:
    df = pd.read_parquet(DATA)
    logger.info("Loaded %d rows", len(df))

    split = build_split(df)

    year_train = df.loc[split.X_train.index, "year"]
    year_val = df.loc[split.X_val.index, "year"]

    # ================================================================
    # Stage 1: MODEL SELECTION on 60% train / 20% validation.
    # No model in this stage is allowed to touch the test set.
    # ================================================================
    prep_sel = FeaturePreprocessor(
        numeric_cols=NUMERIC_FEATURES,
        low_card_cols=LOW_CARD_FEATURES,
        high_card_cols=HIGH_CARD_FEATURES,
        high_card_method="target",
    )
    Xt = prep_sel.fit_transform(split.X_train, split.y_train)  # OOF target enc
    Xv = prep_sel.transform(split.X_val)                        # full-train mapping
    logger.info("Selection feature matrix: %d columns", Xt.shape[1])

    Xt_arr, Xv_arr = Xt.values, Xv.values
    yt = split.y_train.values
    yv = split.y_val.values
    pv = split.price_val.values

    val_results = {}

    lr_val = train_linear(Xt_arr, yt, Xv_arr, yv, pv)
    val_results[lr_val.name] = lr_val.metrics

    rf_val = train_rf(Xt_arr, yt, Xv_arr, yv, pv)
    val_results[rf_val.name] = rf_val.metrics

    lgbm_val = train_lgbm(Xt, yt, Xv, yv, pv)
    val_results[lgbm_val.name] = lgbm_val.metrics

    val_comparison = metrics_table(val_results)
    print("\n=== Model Comparison (VALIDATION set, models fit on 60% train) ===")
    print(val_comparison.to_string())

    # --- Ablation A1: log vs raw target (on val) ---
    logger.info("Running Ablation A1: log vs raw target...")
    a1 = ablation_a1_raw_vs_log(
        Xt, Xv, split.y_train, split.y_val,
        df.loc[split.X_train.index, "price"],
        df.loc[split.X_val.index, "price"],
    )
    a1_table = metrics_table(a1)
    print("\n=== Ablation A1: log vs raw target (val) ===")
    print(a1_table.to_string())

    # --- Ablation A2: collinearity (on val) ---
    logger.info("Running Ablation A2: age only vs age+year...")
    a2 = ablation_a2_collinearity(
        Xt, Xv, split.y_train, split.y_val, year_train, year_val,
    )

    # --- Ablation A3: target vs frequency vs drop encoding (on val) ---
    logger.info("Running Ablation A3: encoding comparison...")
    a3_results = {"target_encoding": lgbm_val.metrics}

    prep_freq = FeaturePreprocessor(
        numeric_cols=NUMERIC_FEATURES,
        low_card_cols=LOW_CARD_FEATURES,
        high_card_cols=HIGH_CARD_FEATURES,
        high_card_method="frequency",
    )
    Xt_freq = prep_freq.fit_transform(split.X_train, split.y_train)
    Xv_freq = prep_freq.transform(split.X_val)
    lgbm_freq = train_lgbm(Xt_freq, yt, Xv_freq, yv, pv)
    a3_results["frequency_encoding"] = lgbm_freq.metrics

    prep_drop = FeaturePreprocessor(
        numeric_cols=NUMERIC_FEATURES,
        low_card_cols=LOW_CARD_FEATURES,
        high_card_cols=[],
        high_card_method="target",
    )
    X_train_no_model = split.X_train.drop(columns=HIGH_CARD_FEATURES)
    X_val_no_model = split.X_val.drop(columns=HIGH_CARD_FEATURES)
    Xt_drop = prep_drop.fit_transform(X_train_no_model, split.y_train)
    Xv_drop = prep_drop.transform(X_val_no_model)
    lgbm_drop = train_lgbm(Xt_drop, yt, Xv_drop, yv, pv)
    a3_results["drop_model_column"] = lgbm_drop.metrics

    a3_table = metrics_table(a3_results)
    print("\n=== Ablation A3: model encoding strategy (val) ===")
    print(a3_table.to_string())

    # ================================================================
    # Stage 2: FINAL EVALUATION on 20% test.
    # Winner is LightGBM (validated in Stage 1). Refit on train + val
    # (the full 80%), then measure once on the untouched test set.
    # ================================================================
    logger.info("Refitting final LightGBM on train+val (80%%) for test-set evaluation...")
    prep_final = FeaturePreprocessor(
        numeric_cols=NUMERIC_FEATURES,
        low_card_cols=LOW_CARD_FEATURES,
        high_card_cols=HIGH_CARD_FEATURES,
        high_card_method="target",
    )
    Xtf = prep_final.fit_transform(split.X_train_full, split.y_train_full)
    Xtest = prep_final.transform(split.X_test)
    lgbm_final = train_lgbm(
        Xtf, split.y_train_full.values,
        Xtest, split.y_test.values, split.price_test.values,
    )
    print("\n=== FINAL LightGBM on TEST (refit on train + val = 80%) ===")
    print(pd.Series(lgbm_final.metrics).round(2).to_string())

    # --- Feature importance from the FINAL model (deployed) ---
    gain = lgbm_final.model.booster_.feature_importance(importance_type="gain")
    imp = pd.Series(gain, index=lgbm_final.model.booster_.feature_name())
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    top_features = imp.head(15).round(2)
    print("\n=== Top 15 Features (final LightGBM, % of total gain) ===")
    print(top_features.to_string())

    # --- Error analysis on TEST with the final model ---
    test_df = df.loc[split.X_test.index].copy()
    test_df["pred_dollar"] = np.expm1(lgbm_final.predictions)

    age_bins = pd.cut(test_df["age"], [0, 3, 6, 10, 15, 60],
                      labels=["0-3yr", "4-6yr", "7-10yr", "11-15yr", "16+yr"])
    err_age = error_by_segment(
        split.y_test.values, lgbm_final.predictions, age_bins, split.price_test.values
    )

    price_bins = pd.cut(test_df["price"],
                        [0, 5000, 10000, 20000, 50000, 150000],
                        labels=["<5k", "5-10k", "10-20k", "20-50k", "50-150k"])
    err_price = error_by_segment(
        split.y_test.values, lgbm_final.predictions, price_bins, split.price_test.values
    )

    top_brands = test_df["manufacturer"].value_counts().head(8).index
    brand_mask = test_df["manufacturer"].isin(top_brands)
    err_brand = error_by_segment(
        split.y_test.values[brand_mask],
        lgbm_final.predictions[brand_mask],
        test_df.loc[brand_mask, "manufacturer"],
        split.price_test.values[brand_mask],
    )

    # --- Write results ---
    write_results(
        val_comparison, top_features, imp,
        err_age, err_price, err_brand,
        a1_table, a2, a3_table,
        lr_val, rf_val, lgbm_val, lgbm_final,
        split,
    )
    logger.info("Results written to %s", RESULTS)


def _a1_note(a1_table) -> str:
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


def _a3_note(a3_table) -> str:
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


def _model_comparison_note(lr_model, lgbm_model) -> str:
    lr, lgb_m = lr_model.metrics, lgbm_model.metrics
    pct_rmse = (lr["RMSE ($)"] - lgb_m["RMSE ($)"]) / lr["RMSE ($)"] * 100
    pp_mape = lr["MAPE (%)"] - lgb_m["MAPE (%)"]
    return (
        f"**LightGBM** wins across all four metrics. The jump from Linear to LightGBM: "
        f"RMSE drops {pct_rmse:.0f}% (${lr['RMSE ($)']:,.0f} -> ${lgb_m['RMSE ($)']:,.0f}), "
        f"MAPE drops {pp_mape:.0f}pp ({lr['MAPE (%)']:.0f}% -> {lgb_m['MAPE (%)']:.0f}%), "
        f"R2 rises {lr['R2']:.2f} -> {lgb_m['R2']:.2f}. Linear's {lr['MAPE (%)']:.0f}% MAPE "
        "confirms the EDA prediction: the age x odometer interaction requires a "
        "non-linear model.\n"
    )


def write_results(
    val_comparison, top_features, all_imp,
    err_age, err_price, err_brand,
    a1_table, a2, a3_table,
    lr_model, rf_model, lgbm_val_model, lgbm_final_model,
    split,
) -> None:
    final_m = lgbm_final_model.metrics
    lines = [
        "# Phase 3 Results -- Price Prediction Model\n",
        "## Split protocol (leakage-safe model selection)\n",
        f"Three-way stratified split (price decile, seed 42): "
        f"**train {len(split.X_train):,} ({len(split.X_train)/len(split.X_train_full)*0.8*100:.0f}%) "
        f"/ val {len(split.X_val):,} (20%) / test {len(split.X_test):,} (20%)**. "
        "All model selection and every ablation below are evaluated on the VAL set. "
        "The chosen model (LightGBM) is then refit on train + val (80%) and measured "
        "once on TEST -- the number below is that single unbiased estimate.\n",
        "## Model comparison (validation set)\n",
        val_comparison.to_markdown(),
        "\n",
        _model_comparison_note(lr_model, lgbm_val_model),
        "> Methodology: high-card `model` uses **out-of-fold** KFold target encoding "
        "(a row is never encoded with its own label); LightGBM early-stops on a "
        "sub-validation split carved from TRAIN (not on the val or test set); "
        "feature importance below is **gain-based**, not split-count.\n",
        "## Final headline metric (test set)\n",
        f"LightGBM refit on train + val ({len(split.X_train_full):,} rows), evaluated on "
        f"the untouched test set ({len(split.X_test):,} rows):\n",
        f"| Model | RMSE ($) | MAE ($) | MAPE (%) | R2 |",
        f"|---|---|---|---|---|",
        f"| LightGBM | {final_m['RMSE ($)']:.2f} | {final_m['MAE ($)']:.2f} "
        f"| {final_m['MAPE (%)']:.2f} | {final_m['R2']:.2f} |",
        "\n> Split integrity (Phase 6A): near-duplicate listings (re-posts) straddling "
        "train/test were measured directly -- 4.6% of test rows have a near-duplicate in "
        "train, but the effect on this headline metric is +0.6% RMSE, not material. "
        "Full probe methodology: [phase6_results.md](phase6_results.md#6a-split-contamination-probe).\n",
        "> Features (Phase 7B): includes 3 leakage-free description-derived features "
        "(`desc_trim_luxury`, `desc_equip_count`, `desc_len_log`) adopted as default after "
        "Ablation A4 showed a real improvement (RMSE -5.3%, MAPE -4.6pp vs the pre-7B "
        "feature set). Details: [phase7_results.md](phase7_results.md).\n",
        "## Feature importance (final LightGBM, % of total gain, top 15)\n",
        "Gain-based (loss reduction), not split-count. Split-count would inflate "
        "high-cardinality features (`model`) and continuous ones (`odometer`) "
        "regardless of real predictive value.\n",
        "| Feature | % of gain |",
        "|---------|-----------|",
    ]
    for feat, val in top_features.items():
        lines.append(f"| {feat} | {val} |")

    highest_mae_age_seg = err_age["MAE"].idxmax()
    highest_mape_age_seg = err_age["MAPE"].idxmax()
    cheapest_seg = err_price["MAPE"].idxmax()
    priciest_seg = "50-150k" if "50-150k" in err_price.index else err_price.index[0]

    if highest_mae_age_seg == highest_mape_age_seg:
        age_note = (
            f"\n**Observation:** {highest_mae_age_seg} has both the highest MAE "
            f"(${err_age.loc[highest_mae_age_seg, 'MAE']:,.0f}) and the highest MAPE "
            f"({err_age.loc[highest_mape_age_seg, 'MAPE']:.0f}%) -- this bucket is the "
            "model's weakest segment on both scales at once, not just proportionally. "
            "It likely spans the widest price range (near-new budget cars to near-new "
            "luxury cars at similar age), which is harder to pin down than a more "
            "homogeneous older-car price band.\n"
        )
    else:
        age_note = (
            f"\n**Observation:** the highest-MAE age bucket is {highest_mae_age_seg} "
            f"(${err_age.loc[highest_mae_age_seg, 'MAE']:,.0f}) because it spans the "
            f"widest price range. The highest-MAPE bucket is {highest_mape_age_seg} "
            f"({err_age.loc[highest_mape_age_seg, 'MAPE']:.0f}%) despite lower MAE "
            f"(${err_age.loc[highest_mape_age_seg, 'MAE']:,.0f}) -- small dollar errors "
            "are large percentages on cheap, old cars.\n"
        )

    lines += [
        "\n## Error analysis (final LightGBM on test)\n",
        "### By age bucket\n",
        err_age.to_markdown(),
        age_note,
        "### By price segment\n",
        err_price.to_markdown(),
        f"\n**Observation:** the {cheapest_seg} segment has "
        f"{err_price.loc[cheapest_seg, 'MAPE']:.0f}% MAPE -- the model's weakest zone. "
        "These are high-mileage, old vehicles where description/condition detail "
        f"matters most. The {priciest_seg} segment has MAE "
        f"${err_price.loc[priciest_seg, 'MAE']:,.0f} but only "
        f"{err_price.loc[priciest_seg, 'MAPE']:.0f}% MAPE -- large dollar errors are "
        "proportionally tolerable on expensive cars.\n",
        "### By manufacturer (top 8)\n",
        err_brand.to_markdown(),
        "\n**Observation:** trucks/SUVs (Ram, GMC, Ford) have the highest errors. "
        "Truck pricing is more variable due to trim/package diversity (a base F-150 vs "
        "a Platinum can differ $30k at the same age). Honda/Nissan have the lowest errors "
        "-- sedan pricing is more predictable.\n",
        "\n---\n",
        "## Ablation studies (all evaluated on val)\n",
        "### A1: Log target vs raw target (LightGBM)\n",
        "**Question:** why train on log1p(price) instead of raw price?\n",
        a1_table.to_markdown(),
        "\n" + _a1_note(a1_table),
        "### A2: Collinearity -- age only vs age + year\n",
        "**Question:** why drop `year` when we have `age`?\n",
        f"- With `age` only: age coefficient = {a2['age_only']['coef_age']:.6f}",
        f"- With both: age coefficient = {a2['age_and_year']['coef_age']:.6f}, "
        f"year coefficient = {a2['age_and_year']['coef_year']:.6f}",
        "- Because corr(age, year) = -1.00 by construction (age = posting_year - year), "
        "the two carry identical information. Adding both makes the linear coefficients "
        "unstable (they can trade magnitude freely without changing predictions). "
        "Tree models are unaffected but it wastes a split dimension.",
        "",
        "| Variant | RMSE ($) | MAE ($) | MAPE (%) | R2 |",
        "|---------|----------|---------|----------|-----|",
        f"| age only | {a2['age_only']['metrics']['RMSE ($)']:.2f} "
        f"| {a2['age_only']['metrics']['MAE ($)']:.2f} "
        f"| {a2['age_only']['metrics']['MAPE (%)']:.2f} "
        f"| {a2['age_only']['metrics']['R2']:.4f} |",
        f"| age + year | {a2['age_and_year']['metrics']['RMSE ($)']:.2f} "
        f"| {a2['age_and_year']['metrics']['MAE ($)']:.2f} "
        f"| {a2['age_and_year']['metrics']['MAPE (%)']:.2f} "
        f"| {a2['age_and_year']['metrics']['R2']:.4f} |",
        "\n**Conclusion:** dropping `year` is correct -- no information is lost and "
        "coefficient interpretation is clean.\n",
        "### A3: High-cardinality encoding for `model` (LightGBM)\n",
        "**Question:** target encoding vs frequency encoding vs dropping `model`?\n",
        a3_table.to_markdown(),
        "\n" + _a3_note(a3_table),
    ]

    RESULTS.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
