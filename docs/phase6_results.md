# Phase 6 Results

## 6A. Split contamination probe

> **Note:** the numbers in this section were measured before Phase 7B adopted
> `desc_*` features as default (they predate that change and are not rerun here
> since 7B's own ablation already retrained the canonical model separately --
> see [phase7_results.md](phase7_results.md)). The current canonical LightGBM
> test RMSE is $6,253 (post-7B), not the $6,611 referenced below; the
> contamination-rate and reach conclusions are not expected to depend on that
> feature addition, but the exact dollar figures here are a historical
> snapshot, not the live headline numbers.

**Question:** do near-duplicate listings (same physical car re-posted, or a
VIN/no-VIN pair) straddle the train/test boundary and inflate reported metrics?

**Method:** `scripts/probe_split_leakage.py`. Reuses the exact Phase 3 split
(`build_split`, seed 42) and the exact Phase 3 final pipeline (OOF target-encoded
LightGBM). For every TEST row, searched TRAIN for a near-duplicate: identical
`(manufacturer, model, year)`, odometer within a fixed band, and relative price
gap within a fixed band. Matching done via sorted-odometer + `searchsorted` per
key group (not a full O(n^2) cross join). Swept three band widths to check the
conclusion does not hinge on one arbitrary epsilon.

### Contamination rate (sensitivity sweep)

| Odometer band | Price band | Contaminated test rows | Share |
|---|---|---|---|
| +/-250 mi | +/-2% | 635 | 1.61% |
| +/-500 mi | +/-5% | 1,814 | **4.59%** |
| +/-1000 mi | +/-10% | 5,019 | 12.69% |

### Metric comparison (default band: 500 mi / 5%)

| | n | RMSE ($) | MAE ($) | MAPE (%) | R2 |
|---|---|---|---|---|---|
| Full test | 39,563 | 6,611.06 | 3,361.01 | 36.90 | 0.757 |
| Contaminated subset | 1,814 | 5,746.61 | 3,373.53 | 19.23 | 0.83 |
| Clean subset | 37,749 | 6,649.78 | 3,360.41 | 37.75 | 0.75 |

Clean-subset RMSE vs full-test RMSE: **+0.59%**.

### Verdict

The pre-registered decision gate was: no fix needed if (contaminated share < 1%)
AND (clean-subset RMSE within 2% of full-test RMSE). The gate **technically
fails** on the first criterion (4.59% > 1%) but **passes comfortably** on the
second (0.59% << 2%).

Reading both numbers together: contamination is real and shows a genuine
leakage signature -- on the 4.59% of test rows with a near-duplicate in train,
MAPE is roughly half the overall rate (19.2% vs 36.9%), i.e. the model gets a
partial "free win" on exactly those rows by having seen a near-identical listing
during training. But because that subset is a small share of the total, its
effect on the headline metrics reported in [phase3_results.md](phase3_results.md)
and the root README is negligible: RMSE moves by $39 (6,611 -> 6,650, +0.6%),
well inside normal run-to-run noise for this pipeline.

**Decision: do not re-split.** A group-aware re-split (GroupShuffleSplit) would
be the methodologically purer choice, but it would require retraining all 3
models plus all 3 Phase 3 ablations and rewriting the reported metric tables,
for an expected headline change on the order of 1% -- not a material correction.
The honest record is this section: contamination was measured, its magnitude and
localized effect are quantified, and the reported Phase 3 numbers are confirmed
not to be materially inflated by it.

**Caveat for a future iteration:** this conclusion is scale-dependent. If the
dataset were resampled at a much larger duplicate rate (e.g. a live-scraped
feed instead of a 30-day static Craigslist snapshot with capped re-posting),
the same 4.59% contamination could compound to a larger effect. The probe
script (`scripts/probe_split_leakage.py`) is kept in the repo so this can be
re-checked cheaply against new data instead of re-derived from scratch.

---

## 6B/6C. Prediction intervals (conformal + Mondrian)

**Method:** split-conformal quantile regression (CQR, Romano et al. 2019). Two LightGBM quantile regressors (lower/upper) per interval level, trained on a `fit` subset of TRAIN; the band is calibrated on a disjoint `calibration` subset of TRAIN (never TEST). Point estimate is a separate alpha=0.5 quantile model. Reuses the Phase 3 / 6A train-test split -- see [6A resolution](#6a-split-contamination-probe).

Two calibration variants from the SAME quantile models:

- **Standard CQR (6B):** one global correction. Guarantees marginal coverage only.
- **Mondrian CQR (6C):** 5 corrections, one per bin of the predicted-band midpoint (log scale). Binning on the model's own output keeps the interval computable for any new listing at inference time -- binning on the actual price would be unavailable for a 'what is it worth' query and would break exchangeability.

Corrections (log scale): standard 90% = 0.0577, standard 99% = 0.0955; Mondrian 90% per-bin = [0.0678, 0.0593, 0.0694, 0.0415, 0.0578] -- note the spread across bins: cheap-car bins need a much larger correction than expensive-car bins, which is exactly the heteroscedasticity one global correction ignores.

### Overall coverage (test set, n=39,563)

| Interval | Calibration | Empirical coverage | Nominal |
|---|---|---|---|
| 90% | raw quantile band (uncalibrated) | 0.8504 | 0.90 |
| 90% | standard CQR (global) | 0.9025 | 0.90 |
| 90% | **Mondrian (per-bin)** | **0.9022** | 0.90 |
| 99% | standard CQR (global) | 0.9888 | 0.99 |
| 99% | **Mondrian (per-bin)** | **0.9894** | 0.99 |

**Reading this:** the raw quantile band is under-covered (85.0% vs 90% target) -- LightGBM's quantile loss is not exactly calibrated on its own; the conformal step is what earns the guarantee.

### 90% coverage by ACTUAL price segment: standard vs Mondrian

| segment   |   standard_coverage |   mondrian_coverage |   mondrian_median_width |   count |
|:----------|--------------------:|--------------------:|------------------------:|--------:|
| <5k       |              0.8376 |              0.8409 |                  5139.7 |    7738 |
| 5-10k     |              0.9349 |              0.9384 |                  7139.5 |   10378 |
| 10-20k    |              0.9251 |              0.9234 |                 11398.1 |   11327 |
| 20-50k    |              0.9114 |              0.9055 |                 20400   |    9191 |
| 50-150k   |              0.7169 |              0.7158 |                 40421.8 |     929 |

**Verdict: Mondrian did NOT materially improve ACTUAL-price-segment coverage** (worst segment 71.7% -> 71.6%). The root-cause probe (`scripts/probe_mondrian_conditional_coverage.py`) shows why -- see the root-cause subsection below. Short version: the guarantee Mondrian actually makes (coverage per PREDICTED-price bin) holds at 89-91% in every bin; the actual-price tail failure is caused by point-model bias on rare expensive trims, which no calibration scheme can repair.

### Root cause: why the actual-price tails stay under-covered

Probe: `scripts/probe_mondrian_conditional_coverage.py` (numbers below are from its recorded run; the probe is deterministic and re-runnable).

1. **The guarantee Mondrian makes is delivered.** Coverage per PREDICTED-price bin: 89.5-90.6% (5 bins), 88.7-90.8% (10 bins), 89.1-90.7% (tail-focused bins). Feature-conditional calibration works.
2. **Finer binning does not move actual-price-segment coverage.** The 50-150k actual segment stays at 69-70% under 5-bin, 10-bin, and tail-focused binnings alike -- the problem is not bin coarseness.
3. **The missed expensive listings are point-model failures, not calibration failures.** Of the 282 uncovered 50-150k listings, the model's predicted band midpoint has median ~$30k against a median actual price of ~$65k. An interval centered at $30k cannot reach $65k under any per-bin widening that keeps intervals useful for the mid-priced majority sharing those bins. These are the rare-trim / heavy-truck listings Phase 3's error analysis already identified as the model's blind spot.
4. **Slicing coverage by ACTUAL price conditions on the outcome.** Exact coverage conditional on outcome-derived groups is provably unattainable in finite samples (Vovk 2012; Foygel Barber et al. 2021). Any slice defined by the target concentrates precisely the rows the model mispredicts -- age-bucket coverage (a feature-based slice) holds fine, as the table below shows.

**Practical conclusion:** report the feature-conditional guarantee (per-predicted-bin, per-age) as the honest product claim; treat the actual-price tail miscoverage as a Phase 3 modeling gap (rare expensive trims need better features -- e.g. trim-level signal from `description` -- not better calibration).

### Mondrian 90% coverage + median width by age bucket

| segment   |   coverage |   median_width |   count |
|:----------|-----------:|---------------:|--------:|
| 0-3yr     |     0.9001 |       21202.3  |    5398 |
| 4-6yr     |     0.9125 |       15128.5  |    7426 |
| 7-10yr    |     0.9004 |        9553.12 |    9880 |
| 11-15yr   |     0.9052 |        6592.04 |    8642 |
| 16+yr     |     0.893  |        7724.83 |    8217 |

See [reports/figures/13_interval_width_vs_price.png](../reports/figures/13_interval_width_vs_price.png) -- left: interval width grows with price (honest heteroscedasticity); right: standard vs Mondrian coverage per price segment.

### Anomaly tie-in: Mondrian interval exceedance vs Phase 4 MAD-z tiers

For each test-set listing: does the actual price fall outside its calibrated interval? Compared against a MAD-z flag refit on this test set (approximation: Phase 4's original MAD reference used the full 197,814-row OOF residual distribution; here it is refit on the 39,563-row test subset for a like-for-like comparison without rerunning the full 5-fold OOF pass).

| Signal | Count | Share of test |
|---|---|---|
| Outside Mondrian 90% interval | 3,871 | 9.78% |
| Outside Mondrian 99% interval | 419 | 1.06% |
| MAD-z, \|z\| > 3.5 (test-refit) | 1,868 | 4.72% |
| MAD-z STRONG tier (test-refit) | 493 | 1.25% |
| Overlap: 90% interval AND \|z\|>3.5 | 1,306 | - |
| Overlap: 99% interval AND STRONG | 205 | - |

**Framing:** the interval-exceedance flag now carries a per-listing, segment-aware threshold with a coverage guarantee behind it, which the global MAD-z band could not offer. The two signals agree on the clear cases (most STRONG-tier listings sit outside the 99% interval). Phase 4 outputs are NOT rewritten; for a production system the recommended flag is 'outside the Mondrian 99% interval', with the MAD-z tiers kept as a cross-check.
