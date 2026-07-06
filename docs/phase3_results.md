# Phase 3 Results -- Price Prediction Model

## Model Comparison

| Model             |   RMSE ($) |   MAE ($) |   MAPE (%) |   R2 |
|:------------------|-----------:|----------:|-----------:|-----:|
| Linear Regression |    8060.17 |   4349.24 |      55.6  | 0.64 |
| Random Forest     |    6974.05 |   3625.98 |      42.78 | 0.73 |
| LightGBM          |    6252.5  |   3142.81 |      32.4  | 0.78 |


**LightGBM** wins across all four metrics. The jump from Linear to LightGBM: RMSE drops 22% ($8,060 -> $6,253), MAPE drops 23pp (56% -> 32%), R2 rises 0.64 -> 0.78. Linear's 56% MAPE confirms the EDA prediction: the age x odometer interaction requires a non-linear model.

> Methodology (fixed during self-review): high-card `model` uses **out-of-fold** KFold target encoding (a row is never encoded with its own label); LightGBM early-stops on a validation split carved from TRAIN, **never** on the test set; feature importance below is **gain-based**, not split-count.

> Split integrity (Phase 6A): near-duplicate listings (re-posts) straddling train/test were measured directly -- 4.6% of test rows have a near-duplicate in train, but the effect on these headline metrics is +0.6% RMSE, not material. Full probe methodology: [phase6_results.md](phase6_results.md#6a-split-contamination-probe).

> Features (Phase 7B): includes 3 leakage-free description-derived features (`desc_trim_luxury`, `desc_equip_count`, `desc_len_log`) adopted as default after Ablation A4 showed a real improvement (RMSE -5.3%, MAPE -4.6pp vs the pre-7B feature set). Details: [phase7_results.md](phase7_results.md).

## Feature Importance (LightGBM, % of total gain, top 15)

Gain-based (loss reduction), not split-count. Split-count would inflate high-cardinality features (`model`) and continuous ones (`odometer`) regardless of real predictive value.

| Feature | % of gain |
|---------|-----------|
| age | 39.21 |
| model | 16.02 |
| odometer | 10.08 |
| desc_len_log | 5.76 |
| cylinders_num | 3.41 |
| mileage_per_year | 2.38 |
| condition_fair | 2.36 |
| fuel_gas | 1.94 |
| drive_fwd | 1.63 |
| desc_equip_count | 1.5 |
| log_odometer | 1.45 |
| desc_trim_luxury | 1.08 |
| paint_color_missing | 0.96 |
| state_or | 0.81 |
| drive_missing | 0.74 |

## Error Analysis

### By age bucket

| segment   |     MAE |   MAPE |   count |
|:----------|--------:|-------:|--------:|
| 0-3yr     | 6089.05 |  43.14 |    5398 |
| 4-6yr     | 3919.12 |  30.59 |    7426 |
| 7-10yr    | 2638.75 |  26.87 |    9880 |
| 16+yr     | 2431.28 |  39.16 |    8217 |
| 11-15yr   | 1888.24 |  27.15 |    8642 |

**Observation:** 0-3yr has both the highest MAE ($6,089) and the highest MAPE (43%) -- this bucket is the model's weakest segment on both scales at once, not just proportionally. It likely spans the widest price range (near-new budget cars to near-new luxury cars at similar age), which is harder to pin down than a more homogeneous older-car price band.

### By price segment

| segment   |      MAE |   MAPE |   count |
|:----------|---------:|-------:|--------:|
| 50-150k   | 20592.7  |  28.7  |     929 |
| 20-50k    |  5213.55 |  16.89 |    9191 |
| 10-20k    |  2603.24 |  17.5  |   11327 |
| <5k       |  1534.85 |  89.26 |    7738 |
| 5-10k     |  1534.69 |  20.33 |   10378 |

**Observation:** the <5k segment has 89% MAPE -- the model's weakest zone. These are high-mileage, old vehicles where description/condition detail matters most. The 50-150k segment has MAE $20,593 but only 29% MAPE -- large dollar errors are proportionally tolerable on expensive cars.

### By manufacturer (top 8)

| segment   |     MAE |   MAPE |   count |
|:----------|--------:|-------:|--------:|
| ram       | 5189.6  |  38.33 |    1388 |
| gmc       | 4147.5  |  36.29 |    1501 |
| ford      | 3776.73 |  34.26 |    6585 |
| jeep      | 3497.06 |  35.2  |    1868 |
| chevrolet | 3439.25 |  35.39 |    5208 |
| toyota    | 2587.3  |  29.49 |    3636 |
| nissan    | 2165.59 |  27.18 |    2075 |
| honda     | 1693.01 |  26.96 |    2535 |

**Observation:** trucks/SUVs (Ram, GMC, Ford) have the highest errors. Truck pricing is more variable due to trim/package diversity (a base F-150 vs a Platinum can differ $30k at the same age). Honda/Nissan have the lowest errors -- sedan pricing is more predictable.


---

## Ablation Studies

### A1: Log target vs raw target (LightGBM)

**Question:** why train on log1p(price) instead of raw price?

| Model        |   RMSE ($) |   MAE ($) |   MAPE (%) |   R2 |
|:-------------|-----------:|----------:|-----------:|-----:|
| log1p(price) |    6252.5  |   3142.81 |      32.4  | 0.78 |
| raw price    |    5773.58 |   3032.99 |      44.92 | 0.81 |

**Key finding:** raw target wins on RMSE ($5,774 raw vs $6,252 log) and R2 (0.81 vs 0.78) because dollar-scale optimization favors getting expensive cars right. But MAPE tells the real story: raw target has 45% average percentage error vs 32% for log -- a 13-point gap. Raw-target models systematically under-predict cheap cars (a $500 error on a $2k car is 25% -- invisible to RMSE but devastating to MAPE). For a marketplace where most listings are under $20k, MAPE is the business-relevant metric. We choose log target.

### A2: Collinearity -- age only vs age + year

**Question:** why drop `year` when we have `age`?

- With `age` only: age coefficient = -0.022610
- With both: age coefficient = 0.219845, year coefficient = 0.242297
- Because corr(age, year) = -1.00 by construction (age = posting_year - year), the two carry identical information. Adding both makes the linear coefficients unstable (they can trade magnitude freely without changing predictions). Tree models are unaffected but it wastes a split dimension.

| Variant | RMSE ($) | MAE ($) | MAPE (%) | R2 |
|---------|----------|---------|----------|-----|
| age only | 8060.17 | 4349.24 | 55.60 | 0.6386 |
| age + year | 8036.90 | 4342.44 | 55.67 | 0.6407 |

**Conclusion:** dropping `year` is correct -- no information is lost and coefficient interpretation is clean.

### A3: High-cardinality encoding for `model` (LightGBM)

**Question:** target encoding vs frequency encoding vs dropping `model`?

| Model              |   RMSE ($) |   MAE ($) |   MAPE (%) |   R2 |
|:-------------------|-----------:|----------:|-----------:|-----:|
| target_encoding    |    6252.5  |   3142.81 |      32.4  | 0.78 |
| frequency_encoding |    6176.81 |   3094.16 |      32.12 | 0.79 |
| drop_model_column  |    6449.16 |   3319.23 |      33.69 | 0.77 |

**Key finding:** with out-of-fold target encoding, **frequency_encoding** wins on RMSE ($6,252 target vs $6,177 frequency vs $6,449 drop), with R2=0.78. Frequency encoding edges target on MAPE (32.1% frequency vs 32.4% target) because it captures the 'popular models are cheaper' signal cheaply. Dropping `model` costs ~$197 RMSE and ~1.3 MAPE points -- model identity carries real trim-level signal (a Civic vs an Accord at equal age/mileage is a $3-5k gap). **Lesson (from an earlier self-review):** an ablation is only trustworthy if the pipeline under it is leakage-free -- a buggy version that silently applied a leaky full-train mapping to training rows made target encoding look worse than frequency/drop; fixing it to genuine OOF encoding reversed the ranking.
