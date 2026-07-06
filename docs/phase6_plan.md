# Phase 6 Plan -- Split Integrity Probe + Prediction Intervals

Two items promoted from the post-handoff critical review (only items scoring >= 7/10):

- **6A. Split contamination probe (potential real defect):** near-duplicate listings may
  straddle the train/test boundary, inflating reported metrics. Probe first; fix only if proven.
- **6B. Prediction intervals (quantile LightGBM + conformal calibration):** turn the point
  model into an interval model. Side effect: gives a cleaner, heteroscedasticity-aware
  framing for the Phase 4 anomaly threshold.

Everything below follows the standard protocol: probe -> prove -> fix -> re-prove -> document.

> **Status: 6A and 6B executed.** Results in [phase6_results.md](phase6_results.md).
> 6A: contamination measured, decision was to skip the re-split (not material). 6B:
> conformal intervals built; marginal coverage holds, per-price-segment coverage does
> not. **6C approved as follow-up:** close that gap with Mondrian calibration.

---

## 6C. Mondrian (group-conditional) conformal calibration

**Why:** 6B's own evaluation showed split-conformal under-covers the price tails
(70.2% on 50-150k, 82.2% on <5k vs the 90% target). Mondrian conformal fits a
separate correction term per group, restoring coverage group by group.

**Key design constraint -- binning variable must be available at inference.**
Groups CANNOT be defined by actual listed price: for "what is this car worth" the
actual price does not exist yet, and conditioning calibration on the ground truth
would break exchangeability. Instead, bin on the **midpoint of the raw quantile
band** (log scale) -- a quantity the interval model itself produces for any row.
Bin edges are quantiles of that midpoint computed on the calibration set (5 bins,
~6.3k calibration rows per bin -- enough for a stable per-bin conformal quantile
even at alpha=0.01).

**Steps:**
1. `MondrianConformalIntervalModel` in `src/models/intervals.py` (subclass of
   `ConformalIntervalModel`): per-bin corrections fit on calibration, applied by
   digitizing each new row's predicted-band midpoint into the same bins.
2. `scripts/predict_intervals.py`: fit Mondrian 90% and 99% alongside standard CQR;
   report per-price-segment coverage side by side (standard vs Mondrian).
3. Honest caveat to verify in the numbers: Mondrian guarantees coverage per
   PREDICTED-price bin; the evaluation segments are ACTUAL-price bins. Where the
   model badly mispredicts (the <5k junk-heavy segment), improvement may be
   partial. Report whatever the numbers say.
4. Update figure 13 (side-by-side coverage bars), 6B/6C sections in
   phase6_results.md, README interval section.

**Success criterion:** worst actual-price-segment coverage moves materially toward
0.90 vs standard CQR's 70.2%, with marginal coverage preserved (~90%).

> **Status: executed -- success criterion NOT met, root cause proven instead.**
> Mondrian delivered its actual guarantee (89-91% coverage per PREDICTED-price bin)
> but actual-price-tail coverage did not move (70.2% -> 69.6%). Root-cause probe
> (`scripts/probe_mondrian_conditional_coverage.py`) proved why: the uncovered
> expensive listings are point-model under-predictions (predicted mid ~$30k vs
> actual ~$65k, medians) -- unreachable by any calibration; and outcome-conditional
> coverage is provably unattainable in general (Foygel Barber et al. 2021). The
> honest deliverable is the feature-conditional guarantee + the documented root
> cause; the tail gap is a Phase 3 modeling issue (rare-trim signal), not a
> calibration issue. Full analysis: [phase6_results.md](phase6_results.md).

---

## 6A. Split contamination probe

### Why this might be a real defect

Current state of the pipeline:

- `src/preprocess/cleaner.py::_deduplicate`: VIN exact-dedup for VIN'd rows; fingerprint
  dedup `(manufacturer, model, price, odometer)` with **exact equality**, applied **only to
  no-VIN rows**.
- `src/models/dataset.py::build_split`: plain `train_test_split` stratified by price decile.
  No group awareness.

Escape routes for a re-posted listing (same physical car, two rows):

1. Re-post with price changed by any amount (e.g. $19,500 -> $19,000): fingerprint differs,
   both rows survive, random split can put one in train and one in test.
2. Re-post where one copy has a VIN and the other does not: VIN dedup never compares them,
   fingerprint dedup never sees the VIN'd row.
3. Odometer updated between posts (car still being driven): fingerprint differs.

If such pairs exist at meaningful volume, the model memorizes the train copy and gets the
test copy nearly free -> RMSE/MAE/MAPE optimistic, and the A3 encoding ablation ordering
could also be affected (target encoding benefits most from near-dupe leakage).

### Step A1 -- Probe script (no pipeline changes)

`scripts/probe_split_leakage.py`:

1. Rebuild the exact Phase 3 split (same seed, same code path via `build_split`).
2. For each TEST row, search TRAIN for a near-duplicate:
   - exact match on `(manufacturer, model, year)`, AND
   - `|odometer_test - odometer_train| <= 500` miles, AND
   - relative price gap `|p_t - p_tr| / max(p_t, p_tr) <= 0.05`.
   Implemented via a merge on the exact keys then filtering, not O(n^2).
3. Report:
   - count and share of contaminated test rows;
   - metric comparison: LightGBM test RMSE/MAE/MAPE/R2 on (a) full test set,
     (b) contaminated subset, (c) clean subset.
4. Sensitivity: repeat with a stricter (250 mi / 2%) and looser (1000 mi / 10%) band so the
   conclusion does not hinge on one arbitrary epsilon.

### Decision gate (agreed thresholds -- proposed, confirm before run)

- **Contaminated share < 1% of test rows AND clean-subset RMSE within ~2% of full-test RMSE:**
  no fix. Document the probe result in this file as a defense note. 6A ends here.
- **Otherwise:** proceed to A2.

### Resolution (executed)

Probe run: contamination 4.59% (500mi/5% band) -- gate's share criterion fails,
but clean-subset RMSE is only +0.59% vs full test -- gate's metric criterion
passes comfortably. **User decision: skip A2, document as a defense note**
(option B) -- the metric impact is not material even though contamination is
measurable. Full numbers and reasoning: [phase6_results.md](phase6_results.md#6a-split-contamination-probe).
Cross-referenced from `phase3_results.md` and the root README. Step A2 below
is left as a reference for what a group-aware re-split would have involved,
in case a future dataset iteration shows a larger effect.

### Step A2 -- Fix (only if gate triggers) -- NOT EXECUTED, see Resolution above

1. Build a `group_id` in `src/models/dataset.py`: connected duplicates under the near-dupe
  definition above (exact `(manufacturer, model, year)` key + odometer/price bands). Rows
  sharing a group never straddle the split (`GroupShuffleSplit`, `random_state=42`).
2. Retrain all 3 models + rerun the 3 ablations with the group-aware split.
3. Update `docs/phase3_results.md` and root `README.md` metric tables. Old numbers stay in
   the doc, struck through with a note -- honest correction, not silent replacement.
4. Regression test: `tests/test_split_integrity.py` asserting zero near-dupe pairs across
   the boundary under the probe definition.

### 6A verification

Re-run the A1 probe script after the fix; contaminated count must be 0. That is the
"prove it is solved" step.

---

## 6B. Prediction intervals (quantile LightGBM + split conformal)

### Design

1. **Quantile models:** three LightGBM regressors with `objective="quantile"` at
   alpha = 0.05, 0.50, 0.95, same feature pipeline as the final Phase 3 model
   (OOF target encoding etc.), trained on the (possibly re-split, post-6A) train set.
2. **Conformal calibration (CQR, split-conformal):** carve a calibration set (~20%) out of
   TRAIN (never test). Compute CQR nonconformity scores on calibration, take the
   (1-alpha)-quantile correction, widen/narrow the raw quantile band accordingly.
   Target nominal coverage: 90%.
3. **New module:** `src/models/intervals.py` (class `ConformalIntervalModel`: fit /
   calibrate / predict_interval). Entry point: `scripts/predict_intervals.py`. Notebook and
   scripts import from src, per repo rules.

### Evaluation (on the held-out test set, dollar scale)

- Empirical coverage vs nominal 90% -- overall AND per price segment / age bucket
  (marginal coverage can hide segment-level failure; this is the table a reviewer checks).
- Mean and median interval width, per segment. Expect wider intervals on cheap/old cars --
  that is the heteroscedasticity showing up honestly instead of being hidden by a global z.
- Sanity: raw (uncalibrated) quantile coverage vs conformalized coverage, to show the
  calibration step actually does something.

### Tie-in to Phase 4 anomaly framing (documentation-level, no pipeline rewrite)

- Compute for each listing whether the actual price falls outside its calibrated 90% (and a
  wider 99%) interval. Compare this flag set against the existing MAD-z STRONG/MODERATE
  tiers: overlap counts + examples where the two disagree.
- This addresses the "global z ignores segment-dependent residual spread" critique: interval
  width adapts per listing. We do NOT rewrite Phase 4 outputs; we add a section to
  `docs/phase6_results.md` showing the comparison, and note it as the recommended framing
  for a production system.

### 6B deliverables

- `src/models/intervals.py`, `scripts/predict_intervals.py`, coverage tables + one figure
  (interval width vs price) in `reports/figures/`.
- `docs/phase6_results.md` with: coverage tables, width analysis, anomaly-flag comparison.
- README: short new section (Turkish) with the headline coverage number.

---

## Execution order

1. A1 probe -> report result -> decision gate (user approval).
2. A2 fix + metric regeneration (only if gate triggers) -> user approval on updated numbers.
3. 6B build -> evaluate -> document -> user approval.

## Out of scope for Phase 6

Items scored < 7 in the review (bootstrap CIs, MNAR ablation, description features,
shrinkage encoding, SHAP, predict.py, broader test suite) -- parked, re-evaluated after 6B.
