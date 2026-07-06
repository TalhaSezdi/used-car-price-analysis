# Phase 7 Plan -- Junk-Listing Filter + Description Trim Features

Motivated by the Phase 6C root cause and the description recon probe
(`scripts/probe_description_signal.py`, findings recorded below in "Evidence").
Two workstreams, executed IN ORDER because 7A changes the dataset 7B trains on --
running them together would make impact attribution impossible:

- **7A. Junk / non-sale listing filter (cleaning-level):** remove listings that are
  not actual vehicle sales. Targets the 6C expensive-tail failure.
- **7B. Leakage-free trim/equipment features from `description` (model-level):**
  capture the within-group ~16.5% trim premium the current features miss.
  Targets general RMSE/MAPE.

Attribution chain: baseline (current) -> after 7A (same features) -> after 7A+7B.
Each step gets its own metrics row; nothing is claimed without a number.

## Evidence (from the recon probe, alignment-verified)

- `description` usable on 100.0% of cleaned rows (median 1,013 chars).
- Within the same (manufacturer, model, year) group, listings mentioning a
  trim/package keyword carry a **median +16.5% price premium** (1,711 groups) --
  signal the current feature set cannot see. Caveat: probe keywords lacked word
  boundaries, so this estimate is noise-diluted; 7B fixes the matching.
- Sampled under-predicted expensive listings (50-150k, predicted < 60% of actual):
  the majority are NOT missing-trim cases but junk -- non-vehicle items (kitchen
  equipment at $60k), "wanted to buy" ads, commercial-equipment trucks (forestry
  bucket truck), and price-entry errors. No trim feature fixes those; a cleaning
  rule does.

---

## 7A. Junk / non-sale listing filter

### Design

New cleaning step `_filter_junk_listings` in `src/preprocess/cleaner.py`, running
BEFORE `description` is dropped (requires reordering: description stays through
filtering, is dropped at the end of the cleaning pipeline as before -- the
processed parquet still contains NO raw text).

Candidate rules (high-precision regex on lowercased description; each rule gets a
counter in `CleaningReport` and is applied only if manual inspection confirms
precision):

1. **Wanted / buyer ads (not sales):** patterns like "looking to buy",
   "we buy", "cash for your", "wanted!" near the start of the text.
2. **Non-vehicle items:** no vehicle-sale vocabulary at all AND `model` missing --
   conservative; catches the "$60k kitchen equipment" class.
3. **Commercial equipment vehicles:** "bucket truck", "boom lift", "forestry unit",
   "cutaway", "box truck with liftgate" etc. Decision to confirm at the gate:
   these ARE vehicles but not consumer-marketplace comparables; proposal is to
   filter them OUT and state the scope as consumer vehicles ("what is this car
   worth" product), documenting the count.

### Step gate (user approval required before retraining anything)

Probe script `scripts/probe_junk_rules.py` reports, per rule: match count, share,
and a 15-listing random sample (year/make/model/price + description snippet) for
manual precision review. Rules that look imprecise get dropped or tightened HERE,
before they touch the pipeline.

### Execution after gate

1. Implement approved rules in `DataCleaner` + update `CleaningReport`.
2. Rerun `clean_data.py` -> new `cleaned.parquet` (row count change documented).
3. Rerun `train.py` -> new model metrics. Compare vs baseline in a
   before/after table (overall + by price segment; the 50-150k segment is the
   one 7A exists for).
4. Rerun `predict_intervals.py` -> does the 50-150k coverage gap shrink?
5. Record everything in `docs/phase7_results.md`.

### Success criteria (7A)

- Each shipped rule shows >= ~90% precision on its inspected sample.
- 50-150k segment MAE/MAPE improves; overall metrics do not degrade materially.
- If metrics do NOT improve: record the negative result, keep the rules only if
  precision review justifies them on data-quality grounds alone.

## 7B. Description trim/equipment features (leakage-free)

### Design

New transformer `DescriptionFeatureExtractor` in `src/features/` -- stateless,
row-wise, no fitting on the target and no cross-row statistics, so it can run at
cleaning time with zero leakage risk:

- `desc_trim_luxury` (0/1): word-boundary match on curated luxury-trim names
  (denali, platinum, king ranch, laramie, lariat, limited, overland, rubicon,
  trd, z71, big horn, sahara, touring...).
- `desc_equip_count` (int): count of distinct equipment keywords present
  (leather, sunroof, navigation, heated seats, tow package, 4x4...).
- `desc_len_log` (float): log1p(description length) -- listing-quality proxy.

Hard leakage rules: alphabetic keywords only, NO digit extraction of any kind
(prices, MSRP, stock numbers stay out by construction); keyword list fixed a
priori, never derived from target statistics.

Wiring: computed during `clean_data.py` while description is available, stored as
numeric columns in the parquet; `description` itself still never reaches the
model. Added to `NUMERIC_FEATURES` in `src/models/dataset.py`.

### Evaluation

- **Ablation A4** in `train.py`: final LightGBM with vs without `desc_*` features,
  same split, same pipeline. RMSE/MAE/MAPE/R2 + error by price segment.
- Sanity: gain-importance of the new features reported; if `desc_trim_luxury`
  shows near-zero gain, the negative result is recorded (precedent: 6C).

### Success criteria (7B)

- A4 shows a real improvement (beyond run noise) in RMSE or MAPE, or a clear
  improvement in the expensive-segment error. Otherwise: negative result,
  features documented but reverted from the default feature list.

## Doc/metric ripple (explicit scope)

Retraining changes reported numbers. In scope: regenerate
`docs/phase3_results.md` (train.py) and review its template narrative for stale
hardcoded numbers; update README metrics table and interval section; rerun
`detect_anomalies.py` and `run_eda.py` so a fresh clone reproduces everything.
Historical phase docs' conclusions are NOT rewritten; `phase7_results.md` records
the deltas.

## Execution order

1. 7A probe -> user gate on rules.
2. 7A implement + full rerun -> before/after table -> user approval.
3. 7B implement + A4 ablation -> user approval.
4. Doc/README sweep.
