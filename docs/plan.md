# Project Plan — Used Car Market Intelligence

## Project Structure
```
Used car/
├── data/
│   ├── raw/          vehicles.csv (not committed; data/raw/README.md placeholder)
│   └── processed/    cleaned.parquet (generated)
├── src/
│   ├── config.py     shared constants (RANDOM_STATE, segment bins, thresholds)
│   ├── preprocess/   cleaning pipeline
│   ├── features/     feature engineering
│   ├── models/       training, encoders, conformal intervals
│   ├── evaluation/   metrics, plots, EDA insights, markdown-report helpers
│   └── anomaly/      residual-based + isolation forest, tiering, diagnostics
├── scripts/
│   ├── clean_data.py
│   ├── run_eda.py
│   ├── train.py
│   ├── detect_anomalies.py
│   ├── predict_intervals.py
│   ├── ablation_description_features.py
│   └── check_consistency.py / probe_*.py   (diagnostic, not part of the run order)
├── tests/            pytest suite mirroring src/ (Phase 8)
├── notebooks/        EDA + presentation
├── reports/          exported figures, suspicious listings report
├── docs/             this file
├── CLAUDE.md
├── requirements.txt
└── README.md
```

---

## Phase 1 — Data Cleaning & Feature Engineering
**Goal:** produce a clean, analysis-ready dataset at `data/processed/cleaned.parquet`.  
**Status:** DONE — 426,880 -> 197,814 rows (46.3% retention). Dedup split: VIN exact 130,857; no-VIN fingerprint 32,826. Fingerprint dedup restricted to no-VIN rows to avoid collapsing distinct VINs.

### Steps

**1.1 Setup**
- Move `vehicles.csv` -> `data/raw/vehicles.csv`
- Create `requirements.txt` with pinned versions
- Create `src/__init__.py` files for package imports

**1.2 Initial Audit** (`src/preprocess/audit.py`)
- Row/column counts, dtypes
- Missing value counts & percentages per column
- `price`, `odometer`, `year` basic distributions (min, max, percentiles)
- Document findings to `docs/phase1_audit.md`

**1.3 Cleaning Pipeline** (`src/preprocess/cleaner.py` — OOP, `DataCleaner` class)
In this order:
1. Drop non-predictive columns: `id`, `url`, `region_url`, `image_url`, `description`, `county`
2. Cast types: `year` -> int, `price`/`odometer` -> float, `posting_date` -> datetime
3. Price filter: drop rows where `price < 500` or `price > 150_000` (document rationale)
4. Year filter: drop rows where `year < 1970` or `year > 2022` (data collected ~2021)
5. Odometer filter: drop rows where `odometer <= 0` or `odometer > 500_000`
6. Title status: keep only `clean` and `rebuilt`; drop salvage/missing/lien/parts only
7. Duplicate removal: deduplicate by VIN (where not null), then by (manufacturer, model, price, odometer) fingerprint
8. Drop rows still missing `price`, `year`, `odometer`, `manufacturer` (core features)

**1.4 Feature Engineering** (`src/features/engineer.py` — `FeatureEngineer` class)
- `age` = `posting_year` - `year`  (posting_year extracted from posting_date; NEVER use current year)
- `mileage_per_year` = `odometer` / `age` (clip age >= 1 to avoid division by zero)
- `log_price` = `log1p(price)` — this is the model target
- `log_odometer` = `log1p(odometer)`
- Encoding strategy documented (not applied here — applied inside model pipeline):
  - High cardinality: `model` -> target/frequency encoding
  - Low cardinality: `condition`, `fuel`, `transmission`, `drive`, `type`, `title_status`, `cylinders`, `paint_color`, `state` -> one-hot or ordinal

**1.5 Output**
- Write `data/processed/cleaned.parquet`
- Print summary: original row count -> final row count -> rows dropped (%), columns kept

**1.6 Entry Point** (`scripts/clean_data.py`)
- Runs 1.3 + 1.4 end to end, writes output

### Success Criteria
- `cleaned.parquet` loads without errors
- `price` range: 500–150,000; no nulls in core columns
- `age` values are all positive and reasonable (1–55)
- Row retention documented (expect ~50–70% of original after aggressive cleaning)

---

## Phase 2 — EDA with Business Insights
**Status:** DONE — 11 figures in reports/figures/, insights in docs/phase2_insights.md.
Core (01-07), experimental (08 heaping, 09 confound), quant-grade (10 age x odometer interaction,
11 missingness structure + model cardinality). Documented caveats: age~year corr = -1.00 (drop one in
linear baseline); state spread confounded with vehicle mix; single-month snapshot (30 days) -> random
split justified in Phase 3. These directly set Phase 3 decisions (encoding, imputation, collinearity).
Reusable plot logic in src/evaluation/plots.py, orchestrated by scripts/run_eda.py.
Key findings: log target justified (skew 2.06 -> -0.43); depreciation ~47% by yr5, ~72% by yr10;
odometer strongest numeric driver (-0.51), flattens after ~150k mi; mileage_per_year near-zero linear
signal; 3.04x state median spread; clear brand tiers.

Key questions to answer:
- Price distribution by manufacturer, type, condition, fuel
- Depreciation curves (price vs age by top manufacturers)
- Regional price differences (state-level)
- Mileage vs price relationship and breakpoints
- What factors correlate most with price?

Each chart: visualization + 1-2 sentence business insight (what this means for a used-car marketplace).

---

## Phase 3 — Price Prediction Model
**Status:** DONE — LightGBM best (RMSE $6,611, MAE $3,361, MAPE 36.9%, R2 0.76). 3 ablations completed.
Gain-based importance sane: age (45%) > model (18%) > odometer (9%). Error analysis by age/price/brand documented.
Self-review caught and fixed 3 issues: (1) target encoder discarded its KFold OOF result and leaked the
train target -> now genuine OOF encoding; (2) LightGBM early-stopped on the test set -> now on a train-carved
val split; (3) split-count importance -> gain-based. Fix (1) reversed the A3 conclusion (target encoding
now wins). Full results in docs/phase3_results.md.

Philosophy: at every decision point, document the alternative we did NOT take and run an
ablation to show the consequence. A data lead asks "why X and not Y?" -- we answer with numbers.

### Decisions, alternatives, and planned evidence

| Decision | We choose | Alternative(s) | How we prove it |
|---|---|---|---|
| Target | `log1p(price)` | raw price; Box-Cox | Ablation A1: raw vs log, compare RMSE/MAPE in $ |
| Collinearity | keep `age`, drop `year` | keep both; keep year | Ablation A2: linear w/ both -> show VIF / unstable coefs |
| `model` encoding | KFold target encoding + smoothing | frequency enc; drop model; collapse rare->other | Ablation A3: target vs frequency vs drop, RMSE delta |
| Missing values | impute + missing-indicator | drop rows; model-native NaN | note + optional check on LightGBM native NaN |
| Split | random 80/20, stratified by price decile | temporal (impossible, 30-day); group-by-model | justified by Phase 2 (single-month snapshot) |
| Model family | Linear -> RF -> LightGBM | SVR/KNN (too slow at 200k) | full metric table across the three |

### Steps
1. `src/models/dataset.py` — assemble model matrix: select features, drop leakage
   (`VIN`, `region`, `lat/long` redundant w/ state, `year` collinear), define X/y, split.
2. `src/models/encoders.py` — leakage-safe encoding (fit on train fold only; KFold target
   encoding for `model`; one-hot / ordinal for low-card; missing-indicator).
3. `src/models/train.py` logic + `scripts/train.py` entry point — train 3 models, CV-tune LightGBM.
4. `src/evaluation/metrics.py` — RMSE, MAE, MAPE, R2 on $ scale (expm1 inverse).
5. Ablations A1-A3 — small, scripted, results written to `docs/phase3_results.md`.
6. Interpretability — LightGBM feature importance; sanity check age/odometer dominate.
7. Error analysis — residuals by price segment / brand / age; document failure modes.

### Success criteria
- LightGBM beats linear baseline on RMSE and MAPE.
- Feature importance is sane (age, odometer, model/brand on top).
- Every alternative in the table has a documented number, not just a claim.

---

## Phase 4 — Anomaly Detection
**Status:** DONE (revised after self-review). All 197,814 listings scored via leakage-free 5-fold
OOF predictions, then tiered:
- Residual STRONG (|z|>5 AND |pct|>85%): 2,197 -- outside plausible model-error band.
- Residual MODERATE (|z|>3.5, not strong): 5,362 -- confound-aware, needs human review.
- Isolation Forest (structural, contamination 0.01): 1,979.
- **HIGH tier (strong residual + IF): 53** -- the highest-confidence action set.

Three fixes applied during self-review:
1. Removed `log_price` from IF features. Independence check: corr(if_score, |z|) dropped 0.336 -> 0.118;
   the residual & IF overlap dropped from an inflated 639 to a genuinely independent 220. IF now uses
   only structural features so the two signals are orthogonal by design.
2. Framed the z threshold as OPERATIONAL, not statistical. Added fat-tail table (|z|>3.5 is ~80x the
   Gaussian rate) and a confound section: with MAPE ~37%, moderate flags may just be model error.
3. Category-based ranking: top-10 underpriced + top-10 overpriced + top-10 structural-only exported
   separately. Overpriced tail exposed classic data-entry-error patterns (e.g. $123,456 on a 1997
   Tahoe) that a pure |z| ranking hid because log residuals are asymmetric.
Alternatives proven with numbers (docs/phase4_results.md). No accuracy metrics invented (no
ground-truth labels), per CLAUDE.md.

Goal: flag suspicious listings (mispriced, likely scam, or data-entry error). No ground-truth
fraud labels exist -> we do NOT invent accuracy/precision/recall. We rank the top-N flagged
listings and justify each qualitatively (per CLAUDE.md).

Two complementary signals:
- **Residual-based** (uses the price model): a listing priced far from its predicted value.
  Actual << predicted -> "too good to be true", scam/fraud or hidden defect signal.
  Actual >> predicted -> data-entry error / spam / rare trim.
- **Isolation Forest** (unsupervised, ignores the price model): structurally weird
  attribute combinations (impossible age/odometer/price mixes that survived cleaning).

Listings flagged by BOTH are the highest-priority "hard" anomalies.

### Decisions, alternatives, and planned evidence

| Decision | We choose | Alternative(s) | How we prove it |
|---|---|---|---|
| Residual scale | log-space residual (approx pct error) | dollar residual | dollar residuals are heteroscedastic; plot residual vs price -> funnel shape |
| Prediction source | out-of-fold (`cross_val_predict`, 5-fold) | in-sample train predictions | in-sample residuals are biased small; compare residual std in-sample vs OOF |
| Flag threshold | robust z via MAD, \|z\| > 3.5, both tails | fixed percentile; std-based z | MAD is robust to the very outliers we are hunting (std is inflated by them) |
| Unsupervised model | Isolation Forest | LOF, One-Class SVM | scalability at ~200k rows; report fit runtime; LOF ~O(n^2), OC-SVM impractical |
| Contamination | explicit 0.01 (~1%) | auto; 0.05 | documented, tunable; keeps the review list human-sized |
| Final ranking | union of both, intersection = top priority | single method only | two methods catch different failure modes (mispriced vs structurally impossible) |

### Steps
1. `src/anomaly/detector.py` — `ResidualAnomalyDetector` (OOF preds -> log residual -> MAD z ->
   directional flag) and `IsolationForestDetector` (fit on engineered numeric space). OOP, leakage-safe.
2. `scripts/detect_anomalies.py` — entry point: load cleaned data, reuse the Phase 3 pipeline to get
   OOF predictions for EVERY row, compute residuals, run Isolation Forest, merge flags, rank, export.
3. Reuse `FeaturePreprocessor` / model from Phase 3 -- do NOT redefine model logic here.
4. Figures (add to `src/evaluation/plots.py`): residual distribution with flagged tails; predicted-vs-actual
   scatter with flagged points colored; IF score distribution.
5. Outputs:
   - `reports/suspicious_listings.csv` — ranked top-N with price, predicted, residual %, IF score, key attrs.
   - `docs/phase4_results.md` — methodology, the alternatives table filled with numbers, and a
     hand-written why-suspicious note on the top ~10 listings.

### Leakage guard (the key rigor point)
Residuals MUST come from out-of-fold predictions. A model scoring its own training rows has
artificially small residuals -> real anomalies hide and the residual scale is biased. Same lesson as
the Phase 3 OOF target-encoding fix, applied to scoring. We score all 197,814 rows via 5-fold
`cross_val_predict` so every listing is judged by a model that did not train on it.

### Success criteria
- Every listing gets a residual z-score and an IF score, all from leakage-free predictions.
- Top-N flagged list is human-inspectable and each has a defensible reason.
- Anomaly flags are NOT fed back into the Phase 3 training pipeline (kept as a separate deliverable).

---

## Phase 5 — Final Packaging
**Status:** DONE.
- Root README.md written in Turkish (user decision; CLAUDE.md language rule updated accordingly):
  metrics table, 5 headline findings each linked to its figure/doc, run order, honesty notes.
  All 13 relative links verified to resolve.
- notebooks/presentation.ipynb: 21 cells (Turkish markdown narrative, English ASCII code).
  Imports thresholds from src/, loads cleaned.parquet + figures + suspicious_listings.csv
  (never re-runs pipeline, never redefines logic). Executed end-to-end: 9 code cells, 0 errors.
- Housekeeping: scipy==1.17.1 pinned (was imported in Phase 4 but unpinned); redundant root
  vehicles.csv deleted after byte-size match vs data/raw/ copy (1,447,955,215 bytes both).

Final lead-eye review pass (post-completion):
- CRITICAL fix: requirements.txt pinned versions did NOT match the environment that produced
  the reported metrics (e.g. lightgbm 4.3.0 pinned vs 4.6.0 actual, sklearn 1.4.2 vs 1.8.0).
  Rewritten to the exact installed versions; the reproducibility claim now holds.
- Removed dead dependencies (xgboost, category_encoders -- never imported anywhere);
  replaced unverifiable jupyter==1.0.0 pin with the actual notebook execution stack
  (ipykernel/nbformat/nbclient); notebook ships pre-executed so any frontend can view it.
- CLAUDE.md aligned with reality: scripts list (evaluate.py never existed; run_eda.py +
  detect_anomalies.py do) and dependency list updated.
- README: 2 Turkish wording fixes, Python version stated, phase1_audit link relabeled
  (it is the raw audit; cleaning rules live in plan.md -- separate link added). All 14
  relative links verified.

Audience: a data team lead with ~10 minutes. What they see must be honest,
skimmable, and every claim must trace to a number or a figure.

### Deliverables
1. **Root `README.md`** — reviewer-first, ~1 page:
   - What / why in 2-3 sentences.
   - How to run: env setup + `clean_data.py -> train.py -> detect_anomalies.py` order.
   - Final metrics table (Linear / RF / LightGBM, 4 metrics).
   - 3-5 headline findings, each with one link to the figure/doc that proves it.
   - Links to `docs/phase{1,2,3,4}_*.md` for depth.
   - Narrative in TURKISH (user decision -- the reviewing data lead is Turkish);
     code blocks, commands, and metric names stay English ASCII.

2. **`notebooks/presentation.ipynb`** — the story deck.
   - Structure: dataset intro -> cleaning decisions -> EDA insights -> modeling
     with ablations -> anomaly detection -> conclusion.
   - IMPORTS `src/` classes/functions; NEVER redefines cleaning or modeling logic
     (CLAUDE.md rule -- notebooks are presentation only).
   - Loads `data/processed/cleaned.parquet` (does NOT re-run the full clean).
   - Loads figures from `reports/figures/` and top listings from
     `reports/suspicious_listings.csv` so it stays under a couple of minutes to run.
   - Each section: figure + 1-2 sentence business takeaway (matches Phase 2 style).

3. **Housekeeping**
   - `requirements.txt` sanity: pin every import actually used (scipy was used in
     Phase 4 -- verify it's listed).
   - Delete any stray root-level `vehicles.csv` if the copy in `data/raw/` is
     authoritative (Phase 1 rule).
   - Confirm end-to-end reproducibility path in README works from a fresh clone.

### Decisions to lock in

| Decision | Choice | Alternative | Reason |
|---|---|---|---|
| Notebook count | one story notebook | separate EDA / model / anomaly | CLAUDE.md: one presentation notebook |
| README length | ~1 page + links | multi-section deep-dive | reviewer has 10 min; depth lives in docs/ |
| Notebook re-runs pipeline? | NO -- loads artifacts | re-runs cleaning/training | keeps notebook fast + guarantees consistency with reports/ |
| Language of artifacts | Turkish narrative, English code | all-English | reviewer is a Turkish data lead; code stays ASCII per CLAUDE.md |
| CI / Makefile | skip | add | overkill for a portfolio; README order is enough |

### Success criteria
- A reviewer opening only README.md gets the story and the numbers in 10 minutes.
- Notebook runs end-to-end on a clean checkout in ~2 min (loading, not training).
- No business logic redefined in the notebook -- every function comes from `src/`.
- Every headline claim in the README links to its source of truth.

---

## Phase 6 — Live Presentation Deck
**Status:** DONE.
- reports/presentation.pptx: 15 slides (14 + 1 backup Q&A), 16:9, Cherry Bold palette
  (navy dark slides for title/thesis/leakage-story/conclusion), Cambria/Calibri.
- Speaker notes on every slide (Turkish, ~60-90s each, with anticipated-question answers);
  duplicated to docs/speaker_notes.md for rehearsal.
- Built with python-pptx (no node/LibreOffice on this machine); native editable table + 2
  native bar charts (gain importance, MAPE by segment); 4 figures embedded from reports/figures/.
- QA: all 15 slides exported to PNG via PowerPoint COM and visually inspected. One fix cycle:
  title-slide stat wording, Tahoe "+%4,564" -> "~47 kat" (Turkish decimal-comma ambiguity),
  "Nis-May" -> "Nisan-Mayis". Re-verified fixed slides.

Context: 15-20 min live technical presentation to the arabam.com data team lead.
Deliverables: `reports/presentation.pptx` + speaker notes per slide (inside the pptx notes
pane, and a separate `docs/speaker_notes.md` for rehearsal).
Language: Turkish narrative, English figures/code (consistent with project artifacts).
Figures embedded from `reports/figures/` (no regeneration).

Slide plan (14 + 1 backup):
1. Title -- project, one-line summary.
2. Problem & data -- 426,880 listings, 4 deliverables, why it is hard (dirty data, 20k
   model cardinality, no fraud labels).
3. Methodology thesis -- "why X and not Y?" answered with ablations, not claims.
   This is the differentiator slide.
4. Cleaning decisions -- rule table, 46.3% retention, the dedup trap story (271 real
   cars saved by a probe script).
5. EDA I -- front-loaded depreciation (fig 02) + age x odometer interaction (fig 10):
   the "a tree model will be needed" prediction.
6. EDA II -- confound analysis (fig 09): VIN premium is mostly age; why univariate
   rules mislead.
7. Model setup -- log target, OOF target encoding, stratified split, leakage guards.
8. Results -- 3-model table, LightGBM RMSE $6,611 / MAPE 36.9% / R2 0.76; gain
   importance age 45% > model 18% > odometer 9%.
9. Ablations -- A1 (log loses RMSE, wins MAPE by 13pp -> business metric priority),
   A2 (collinearity), A3 (encoding).
10. The leakage story -- self-caught encoder bug reversed A3's conclusion. Strongest
    live moment: found, proved, fixed, documented.
11. Error analysis -- where the model is weak (<$5k MAPE ~109%, pickup brands) and why.
12. Anomaly detection -- two INDEPENDENT signals (why log_price was removed from IF:
    corr 0.336 -> 0.118), tiered output (fig 12).
13. Suspicious listing examples -- $549 Ram 3500, $123,456 Tahoe, 473k-mi Silverado;
    mapped business actions.
14. Conclusion -- business recommendations + honest limits.
15. (Backup) Anticipated questions -- why not XGBoost / temporal split / Box-Cox;
    why MAPE is high; prepared answers.
