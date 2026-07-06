# Phase 3 Results -- Price Prediction Model

## Model Comparison

| Model             |   RMSE ($) |   MAE ($) |   MAPE (%) |   R2 |
|:------------------|-----------:|----------:|-----------:|-----:|
| Linear Regression |    8045.98 |   4345.6  |      55.56 | 0.64 |
| Random Forest     |    6955.36 |   3620.62 |      42.94 | 0.73 |
| LightGBM          |    6261.07 |   3149.15 |      32.34 | 0.78 |


**LightGBM** wins across all four metrics. The jump from Linear to LightGBM: RMSE drops 22% ($8,046 -> $6,261), MAPE drops 23pp (56% -> 32%), R2 rises 0.64 -> 0.78. Linear's 56% MAPE confirms the EDA prediction: the age x odometer interaction requires a non-linear model.

> Methodology (fixed during self-review): high-card `model` uses **out-of-fold** KFold target encoding (a row is never encoded with its own label); LightGBM early-stops on a validation split carved from TRAIN, **never** on the test set; feature importance below is **gain-based**, not split-count.

> Split integrity (Phase 6A): near-duplicate listings (re-posts) straddling train/test were measured directly -- 4.6% of test rows have a near-duplicate in train, but the effect on these headline metrics is +0.6% RMSE, not material. Full probe methodology: [phase6_results.md](phase6_results.md#6a-split-contamination-probe).

> Features (Phase 7B): includes 3 leakage-free description-derived features (`desc_trim_luxury`, `desc_equip_count`, `desc_len_log`) adopted as default after Ablation A4 showed a real improvement (RMSE -5.3%, MAPE -4.6pp vs the pre-7B feature set). Details: [phase7_results.md](phase7_results.md).

## Feature Importance (LightGBM, % of total gain, top 15)

Gain-based (loss reduction), not split-count. Split-count would inflate high-cardinality features (`model`) and continuous ones (`odometer`) regardless of real predictive value.

| Feature | % of gain |
|---------|-----------|
| age | 39.16 |
| model | 16.03 |
| odometer | 10.03 |
| desc_len_log | 5.79 |
| cylinders_num | 3.43 |
| mileage_per_year | 2.37 |
| condition_fair | 2.35 |
| fuel_gas | 1.93 |
| drive_fwd | 1.58 |
| desc_equip_count | 1.53 |
| log_odometer | 1.5 |
| desc_trim_luxury | 1.08 |
| paint_color_missing | 1.01 |
| state_or | 0.87 |
| drive_missing | 0.78 |

## Error Analysis

### By age bucket

| segment   |     MAE |   MAPE |   count |
|:----------|--------:|-------:|--------:|
| 0-3yr     | 6111.84 |  42.71 |    5398 |
| 4-6yr     | 3933.58 |  30.12 |    7426 |
| 7-10yr    | 2631.26 |  27.09 |    9880 |
| 16+yr     | 2439.97 |  39.31 |    8217 |
| 11-15yr   | 1890.91 |  27.13 |    8642 |

**Observation:** 0-3yr has both the highest MAE ($6,112) and the highest MAPE (43%) -- this bucket is the model's weakest segment on both scales at once, not just proportionally. It likely spans the widest price range (near-new budget cars to near-new luxury cars at similar age), which is harder to pin down than a more homogeneous older-car price band.

### By price segment

| segment   |      MAE |   MAPE |   count |
|:----------|---------:|-------:|--------:|
| 50-150k   | 20626.3  |  28.79 |     929 |
| 20-50k    |  5241.81 |  16.97 |    9191 |
| 10-20k    |  2601.72 |  17.5  |   11327 |
| 5-10k     |  1536.61 |  20.39 |   10378 |
| <5k       |  1529.31 |  88.77 |    7738 |

**Observation:** the <5k segment has 89% MAPE -- the model's weakest zone. These are high-mileage, old vehicles where description/condition detail matters most. The 50-150k segment has MAE $20,626 but only 29% MAPE -- large dollar errors are proportionally tolerable on expensive cars.

### By manufacturer (top 8)

| segment   |     MAE |   MAPE |   count |
|:----------|--------:|-------:|--------:|
| ram       | 5242.8  |  38.77 |    1388 |
| gmc       | 4061.4  |  35.08 |    1501 |
| ford      | 3782.73 |  34.11 |    6585 |
| jeep      | 3527.06 |  34.4  |    1868 |
| chevrolet | 3452.55 |  35.68 |    5208 |
| toyota    | 2600.12 |  29.56 |    3636 |
| nissan    | 2173.69 |  27.75 |    2075 |
| honda     | 1702.01 |  27    |    2535 |

**Observation:** trucks/SUVs (Ram, GMC, Ford) have the highest errors. Truck pricing is more variable due to trim/package diversity (a base F-150 vs a Platinum can differ $30k at the same age). Honda/Nissan have the lowest errors -- sedan pricing is more predictable.


---

## Ablation Studies

### A1: Log target vs raw target (LightGBM)

**Question:** why train on log1p(price) instead of raw price?

| Model        |   RMSE ($) |   MAE ($) |   MAPE (%) |   R2 |
|:-------------|-----------:|----------:|-----------:|-----:|
| log1p(price) |    6261.07 |   3149.15 |      32.34 | 0.78 |
| raw price    |    5765.26 |   3019.48 |      44.86 | 0.82 |

**Key finding:** raw target wins on RMSE ($5,765 raw vs $6,261 log) and R2 (0.82 vs 0.78) because dollar-scale optimization favors getting expensive cars right. But MAPE tells the real story: raw target has 45% average percentage error vs 32% for log -- a 13-point gap. Raw-target models systematically under-predict cheap cars (a $500 error on a $2k car is 25% -- invisible to RMSE but devastating to MAPE). For a marketplace where most listings are under $20k, MAPE is the business-relevant metric. We choose log target.

### A2: Collinearity -- age only vs age + year

**Question:** why drop `year` when we have `age`?

- With `age` only: age coefficient = -0.022912
- With both: age coefficient = 0.219774, year coefficient = 0.242528
- Because corr(age, year) = -1.00 by construction (age = posting_year - year), the two carry identical information. Adding both makes the linear coefficients unstable (they can trade magnitude freely without changing predictions). Tree models are unaffected but it wastes a split dimension.

| Variant | RMSE ($) | MAE ($) | MAPE (%) | R2 |
|---------|----------|---------|----------|-----|
| age only | 8045.98 | 4345.60 | 55.56 | 0.6399 |
| age + year | 8022.88 | 4338.83 | 55.63 | 0.6419 |

**Conclusion:** dropping `year` is correct -- no information is lost and coefficient interpretation is clean.

### A3: High-cardinality encoding for `model` (LightGBM)

**Question:** target encoding vs frequency encoding vs dropping `model`?

| Model              |   RMSE ($) |   MAE ($) |   MAPE (%) |   R2 |
|:-------------------|-----------:|----------:|-----------:|-----:|
| target_encoding    |    6261.07 |   3149.15 |      32.34 | 0.78 |
| frequency_encoding |    6458.61 |   3207.11 |      32.23 | 0.77 |
| drop_model_column  |    6449.16 |   3319.23 |      33.69 | 0.77 |

**Key finding:** with out-of-fold target encoding, **target_encoding** wins on RMSE ($6,261 target vs $6,459 frequency vs $6,449 drop), with R2=0.78. Frequency encoding edges target on MAPE (32.2% frequency vs 32.3% target) because it captures the 'popular models are cheaper' signal cheaply. Dropping `model` costs ~$188 RMSE and ~1.3 MAPE points -- model identity carries real trim-level signal (a Civic vs an Accord at equal age/mileage is a $3-5k gap). **Lesson (from an earlier self-review):** an ablation is only trustworthy if the pipeline under it is leakage-free -- a buggy version that silently applied a leaky full-train mapping to training rows made target encoding look worse than frequency/drop; fixing it to genuine OOF encoding reversed the ranking.
