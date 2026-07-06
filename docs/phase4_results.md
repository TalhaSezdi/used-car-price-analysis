# Phase 4 Results -- Anomaly Detection

Scored all 197,814 listings with two complementary signals. No ground-truth fraud labels exist, so we report NO accuracy metrics -- we rank flagged listings and justify each qualitatively.

## Flag counts (tiered)

| Tier | Count | Share | Meaning |
|------|-------|-------|---------|
| Residual, |z| > 3.5 (total) | 7,479 | 3.8% | operational threshold, not a statistical claim |
| -- of which STRONG (|z| > 5.0 AND |resid_pct| > 85.0%) | 1,955 | 0.99% | far outside plausible model-noise band |
| -- of which MODERATE | 5,524 | 2.8% | possibly a bad listing, possibly the model missing a rare trim (MAPE is ~37%) |
| Isolation Forest (structural) | 1,979 | 1.0% | contamination fixed at 0.01, so this share is a parameter, not a finding |
| Residual & IF overlap | 224 | 0.11% | any-tier residual + IF |
| **HIGH: strong residual + IF** | **49** | **0.025%** | **highest-confidence action set** |

## Alternatives, with evidence (the 'why X not Y')

### 1. Out-of-fold vs in-sample predictions (leakage guard)

- In-sample residual std (model scores its own training rows): **0.3804** (log scale)
- Out-of-fold residual std (leakage-free): **0.4082**
- In-sample residuals are 7% smaller. The effect is modest (a regularized LightGBM at ~200k rows does not overfit hard), but the OOF guard is free and correct, and it matters most exactly on the extreme rows we care about. Every row is scored by a model that never trained on it.

### 2. Log-space vs dollar residual (scale choice)

Dollar residuals are heteroscedastic -- their spread scales mechanically with price, so a fixed dollar threshold would systematically over-flag expensive cars and under-flag cheap ones:

- Dollar residual std: **$2,736** (price < $10k) vs **$13,068** (price > $30k) -- 4.8x wider on expensive cars purely because the numbers are bigger.
- Log residual std: **0.489** (< $10k) vs **0.347** (> $30k) -- roughly flat, with slightly HIGHER dispersion on cheap cars (genuine signal: cheap high-mileage cars have more relative price uncertainty).
- Log residual = approx pct error, comparable across the price range. One z-threshold is defensible everywhere; on dollar residuals it would not be.

### 3. Robust (MAD) vs standard (std) z-score

We standardize with median + MAD, not mean + std. The outliers we are hunting inflate the mean and std and would mask themselves; median/MAD are unaffected by the tails, giving a stable reference distribution.

### 4. Isolation Forest -- and why we EXCLUDE `log_price` from its features

- Fit on 197,814 rows in **13.1s**. IF is ~O(n log n) via random subsampling; LOF is ~O(n^2) on pairwise distances (~39e9 ops) and One-Class SVM does not scale to ~200k. IF is the only practical full-data choice.
- **Design decision (fixed during self-review):** `log_price` was originally in the IF feature set. That made IF partly re-detect the same price-outlier signal the residual method already catches -- inflating the 'flagged by BOTH' overlap mechanically. Independence check: with `log_price` in IF, corr(if_score, |z|) = 0.336; without it, 0.118. Also without price, the residual & IF overlap drops from 639 to a genuinely independent ~220 set. IF now uses only structural features (age, odometer, mileage_per_year, cylinders_num), so the two signals are orthogonal by construction and the 'HIGH' tier truly means 'suspicious on price AND structurally weird'.

### 5. Confound warning: residual = listing anomaly + model error

Without labels, we cannot cleanly separate 'bad listing' from 'model missed a rare car'. The tiered threshold above is the honest split: the STRONG tier (|z|>5.0 AND |pct|>85.0%) is far outside the model's ~37% MAPE band and is almost certainly listing-side (junk price / placeholder / scam); the MODERATE tier is ambiguous and needs human review, not auto-action.

### 6. Fat-tail check: threshold is operational, not Gaussian

|z| > 3.5 is NOT 'a 3.5-sigma event'. Under a Gaussian tail the flag rate would be ~0.05%; the actual rate is ~80x that. The residual distribution has heavy tails (rare cars, listing junk, model error), so the threshold is a **capacity choice** (how many listings a human queue can absorb), not a rarity claim:

| Threshold | Observed flags | Observed % | Gaussian expects | Fold excess |
|-----------|----------------|-----------|------------------|-------------|
| |z| > 3.5 | 7,479 | 3.78% | 92.03 (0.0465%) | 81x |
| |z| > 5 | 3,688 | 1.86% | 0.11 (0.0001%) | 32,520x |
| |z| > 7 | 1,828 | 0.92% | 0.00 (0.0000%) | 3.6e+09 |
| |z| > 10 | 749 | 0.38% | 0.00 (0.0000%) | 7.5e+11 |

---

## Top listings, per action category

Ranked by |z| within category so the reviewer sees examples of each business action, not just the underpriced tail (which structurally dominates any |z| ranking because log residuals are asymmetric).

### Top 10 UNDERPRICED (route to fraud / trust-and-safety review)

1. **2014 ram 3500 laramie** (65,862 mi, condition missing): listed $549 vs model expects $52,275 (-99% below) -- far-below-market is a classic scam / hidden-defect / placeholder signal.
2. **2018 toyota tacoma trd sport** (22,190 mi, good): listed $512 vs model expects $47,396 (-99% below) -- far-below-market is a classic scam / hidden-defect / placeholder signal.
3. **2018 toyota tacoma trd sport** (36,890 mi, good): listed $539 vs model expects $49,485 (-99% below) -- far-below-market is a classic scam / hidden-defect / placeholder signal.
4. **2019 ram 1500 big horn/lone star** (25,561 mi, good): listed $515 vs model expects $39,090 (-99% below) -- far-below-market is a classic scam / hidden-defect / placeholder signal.
5. **2020 toyota tacoma** (11,000 mi, like new): listed $555 vs model expects $39,869 (-99% below) -- far-below-market is a classic scam / hidden-defect / placeholder signal.
6. **2015 rover sport su** (55,187 mi, condition missing): listed $507 vs model expects $36,403 (-99% below) -- far-below-market is a classic scam / hidden-defect / placeholder signal.
7. **2019 lexus nx** (16,544 mi, excellent): listed $521 vs model expects $37,284 (-99% below) -- far-below-market is a classic scam / hidden-defect / placeholder signal.
8. **2017 toyota tundra limited** (44,875 mi, good): listed $619 vs model expects $42,465 (-98% below) -- far-below-market is a classic scam / hidden-defect / placeholder signal.
9. **2020 ram laramie** (15,084 mi, like new): listed $722 vs model expects $49,463 (-98% below) -- far-below-market is a classic scam / hidden-defect / placeholder signal.
10. **2020 ram 1500 classic slt** (56,908 mi, good): listed $510 vs model expects $34,887 (-98% below) -- far-below-market is a classic scam / hidden-defect / placeholder signal.

### Top 10 OVERPRICED (nudge seller: your price is above market)

1. **2020 chevrolet silverado 3500hd** (10,426 mi, condition missing): listed $79,710 vs model expects $597 (+13252%) -- likely an over-ask, data-entry error, or a rare trim the model does not capture.
2. **1997 chevrolet tahoe** (270,000 mi, fair): listed $123,456 vs model expects $2,454 (+4931%) -- likely an over-ask, data-entry error, or a rare trim the model does not capture.
3. **1983 dodge rampage** (100,000 mi, fair): listed $75,000 vs model expects $1,693 (+4330%) -- likely an over-ask, data-entry error, or a rare trim the model does not capture.
4. **2006 dodge caravan** (200,000 mi, excellent): listed $111,111 vs model expects $2,594 (+4183%) -- likely an over-ask, data-entry error, or a rare trim the model does not capture.
5. **2020 toyota tacoma** (23,524 mi, condition missing): listed $43,988 vs model expects $1,233 (+3468%) -- likely an over-ask, data-entry error, or a rare trim the model does not capture.
6. **1997 chevrolet express 2500 4x4** (195,469 mi, fair): listed $100,000 vs model expects $3,046 (+3183%) -- likely an over-ask, data-entry error, or a rare trim the model does not capture.
7. **2017 ford f-150** (24,955 mi, condition missing): listed $41,747 vs model expects $1,464 (+2752%) -- likely an over-ask, data-entry error, or a rare trim the model does not capture.
8. **2009 ford escape** (182,415 mi, good): listed $150,000 vs model expects $5,332 (+2713%) -- likely an over-ask, data-entry error, or a rare trim the model does not capture.
9. **2009 nissan rogue** (198,000 mi, good): listed $123,456 vs model expects $4,754 (+2497%) -- likely an over-ask, data-entry error, or a rare trim the model does not capture.
10. **2018 jeep wrangler jk unlimited rubi** (21,868 mi, condition missing): listed $46,638 vs model expects $1,833 (+2444%) -- likely an over-ask, data-entry error, or a rare trim the model does not capture.

### Top 10 STRUCTURAL-ONLY (prompt seller to confirm year / mileage)

Flagged by Isolation Forest but NOT by the residual signal -- their price is reasonable, but the age/odometer/mileage-per-year combination is unusual.

1. **2016 chevrolet silverado** (473,100 mi, excellent): listed $32,500; not extreme on price alone, but IF score 0.75 -- attribute combination is unusual (age/odometer/mileage-per-year mix), likely a data-entry error.
2. **2016 ford f550 super duty bus** (291,000 mi, like new): listed $19,200; not extreme on price alone, but IF score 0.75 -- attribute combination is unusual (age/odometer/mileage-per-year mix), likely a data-entry error.
3. **2015 ford super duty f-250 xl** (401,474 mi, good): listed $6,995; not extreme on price alone, but IF score 0.75 -- attribute combination is unusual (age/odometer/mileage-per-year mix), likely a data-entry error.
4. **2021 ford f450 super duty** (300,000 mi, good): listed $18,500; not extreme on price alone, but IF score 0.75 -- attribute combination is unusual (age/odometer/mileage-per-year mix), likely a data-entry error.
5. **2013 gmc yukon** (470,000 mi, excellent): listed $11,000; not extreme on price alone, but IF score 0.74 -- attribute combination is unusual (age/odometer/mileage-per-year mix), likely a data-entry error.
6. **2013 gmc yukun lx** (470,000 mi, excellent): listed $11,000; not extreme on price alone, but IF score 0.74 -- attribute combination is unusual (age/odometer/mileage-per-year mix), likely a data-entry error.
7. **2014 chevrolet express cutaway 3500** (376,496 mi, condition missing): listed $9,499; not extreme on price alone, but IF score 0.74 -- attribute combination is unusual (age/odometer/mileage-per-year mix), likely a data-entry error.
8. **2013 chevrolet express 1500 awd 5.3l** (410,384 mi, excellent): listed $10,700; not extreme on price alone, but IF score 0.74 -- attribute combination is unusual (age/odometer/mileage-per-year mix), likely a data-entry error.
9. **2016 gmc sierra 3500hd sle** (321,649 mi, good): listed $27,968; not extreme on price alone, but IF score 0.74 -- attribute combination is unusual (age/odometer/mileage-per-year mix), likely a data-entry error.
10. **2016 volvo vnl 670** (499,232 mi, excellent): listed $39,999; not extreme on price alone, but IF score 0.74 -- attribute combination is unusual (age/odometer/mileage-per-year mix), likely a data-entry error.

## Business use

- **HIGH (49 listings): strong mispriced + structural.** Route to trust-and-safety BEFORE the listing goes live. Two independent signals agree: implausible price AND weird attribute combination.
- **Strong mispriced (any direction):** underpriced -> fraud review; overpriced -> seller nudge. Extreme enough that model error is an unlikely explanation.
- **Moderate mispriced:** human-review queue only. At MAPE ~37%, a moderate flag may just be the model missing a rare trim -- do NOT auto-action.
- **Structural only:** likely data-entry error -- ask the seller to confirm year and mileage before publishing.

> By design, these flags are a separate deliverable and are NOT fed back into the Phase 3 training pipeline.
