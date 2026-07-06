# CLAUDE.md

This file provides strict guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 0. Communication & Persona (CRITICAL)
- Speak like a direct, pragmatic senior data scientist discussing a problem with a peer.
- Keep explanations extremely concise, practical, and straight to the point.
- NO assumptions. Never guess. If something is missing, stop and ask.
- DO NOT use emojis anywhere (not in code, not in comments, not in chat responses).
- DO NOT use Turkish characters in code, variables, function names, or code comments. Use standard English ASCII only.
- Language split (decided in Phase 5): NARRATIVE artifacts aimed at the reviewer (root README.md, notebook markdown cells) are written in Turkish, since the reviewing data lead is Turkish. CODE stays English ASCII everywhere: code cells, comments, identifiers, figure labels, and docs/ phase documents already written in English stay as they are.
- This project is a portfolio piece intended to be reviewed by a data team lead. Every artifact (code, notebook, README) must be clean, readable, and defensible in a technical review.

## 1. Agent Workflow Protocol (Strict Execution)
You must follow this exact sequence for ANY task, feature, or analysis step:
1. **Plan:** Create or update a plan document under the docs/ directory. Outline the phases and steps required.
2. **Wait for Approval:** Ask the user for explicit approval on the plan. Do NOT write code yet.
3. **Phase Execution:** Execute the plan phase by phase.
4. **Code:** Write the code for the current phase based on the approved plan.
5. **Verify:** Run the code, inspect outputs, and validate results (shapes, distributions, metrics) before declaring a phase done.
6. **Fix & Record:** Fix any issues. Once a phase is successful, update the plan document in docs/ with the execution results. Move to the next phase only after user approval.

## 2. Problem Solving Framework
When facing an issue, bug, or metric degradation, follow this scientific method without making assumptions:
1. **Define the problem:** What is failing?
2. **Prove it exists:** Write a minimal script to reproduce the exact error or the surprising result.
3. **Find the root cause:** Investigate why it happens (data leakage, null values, type mismatch, outliers, wrong encoding, etc.).
4. **Solve:** Implement the fix.
5. **Prove it is solved:** Re-run the exact reproduction script from step 2 and show it now passes.

## 3. Project Architecture & Coding Standards
All reusable logic must be written following Object-Oriented Programming (OOP) principles. Code must be modular, reusable, and follow industry standards (PEP8, type hinting, docstrings). Notebooks are for narrative and presentation only; business logic lives in src/.

### Code cleanliness rules (strict)
- **src/ = generic, parametric, reusable.** Every function/class in src/ must be generic: no hardcoded column names, file paths, magic numbers, or dataset-specific literals baked into the function body. Anything that can vary (column name, threshold, path, model hyperparameter) must be a parameter with a sensible default, not a hardcoded value. A function should work if handed a different but structurally similar dataframe.
- **One responsibility per module, grouped by folder.** Each function/class lives in the src/ subpackage matching its responsibility (preprocess/, features/, models/, evaluation/, anomaly/) — never dump unrelated logic into one file, and never duplicate a helper across subpackages. If similar logic is needed in two places, extract it once and import it.
- **scripts/ = thin orchestration only.** Files under scripts/ must only: import the required classes/functions from src/, wire them together (instantiate, call, save output), and handle CLI-level concerns (argparse, logging, file paths for this specific run). No business logic, no data transformation code, no model-fitting internals directly inside a script — that all belongs in src/. A script should read like a short, readable pipeline of calls.
- **No dead code, no commented-out blocks, no speculative parameters.** Do not add a parameter, flag, or branch for a case that is not currently used. Delete unused imports/functions rather than leaving them "just in case."
- **Docstrings on every public function/class in src/**: one-line summary, Args, Returns (Google or NumPy style, pick one and stay consistent across the whole src/ tree).

Project structure:
- docs/ -> Planning documents, phase tracking, and results.
- data/
  - raw/ -> Original untouched data (vehicles.csv).
  - processed/ -> Cleaned / feature-engineered datasets (generated, not committed if large).
- src/ -> Core modular packages.
  - config.py -> Shared constants (RANDOM_STATE, price/age segment bins, anomaly z-threshold, interval alpha) -- single source of truth, imported by every module below instead of being redefined locally.
  - preprocess/ -> Data loading, cleaning, missing-value handling, outlier filtering, type casting.
  - features/ -> Feature engineering classes and transformers (age, encoders, buckets).
  - models/ -> Model definitions, training logic, and wrappers.
  - evaluation/ -> Metrics, error analysis, plotting utilities, EDA insights, and markdown-report helpers (metrics.py, plots.py, insights.py, reporting.py).
  - anomaly/ -> Anomaly detection logic (residual-based flagging, isolation forest, tiering, fat-tail diagnostics).
- scripts/ -> Executable entry points (run in this order).
  - clean_data.py -> Run the full cleaning pipeline, write processed dataset.
  - run_eda.py -> Generate all EDA figures and docs/phase2_insights.md.
  - train.py -> Train 3 models + 3 ablations, error analysis, docs/phase3_results.md.
  - detect_anomalies.py -> OOF anomaly scoring, reports/suspicious_listings.csv.
  - predict_intervals.py -> Conformal + Mondrian prediction intervals, docs/phase6_results.md (6B/6C).
  - ablation_description_features.py -> Ablation A4 (description-derived features), docs/phase7_results.md (7B).
  - Remaining scripts (check_consistency.py, probe_*.py) are diagnostic/one-off tools per the Problem Solving Framework (section 2), not part of the run order above.
- notebooks/ -> EDA and presentation only. No business logic here.
- reports/ -> Exported figures and the written business-insight findings.
- tests/ -> pytest suite mirroring src/ (preprocess/, features/, models/, evaluation/, anomaly/, scripts/). Every src/ module gets its tests written in the same step it's changed, not deferred.

## 4. Project Goal & Data
Build an end-to-end used-car analysis and price-prediction project on a large Craigslist listings dataset. The project has four deliverables, in order:
1. **Data Cleaning & Feature Engineering** — a defensible, documented cleaning pipeline.
2. **EDA with Business Insights** — every chart is followed by a "what this means for a used-car marketplace" interpretation, not just a description.
3. **Price Prediction Model (regression)** — predict listing `price` from vehicle attributes.
4. **Anomaly Detection** — flag suspicious listings (mispriced, likely fraudulent, or data-entry errors) using the price model residuals and unsupervised methods.

Goal: Given a vehicle's attributes (year, make, model, mileage, condition, etc.), predict its listing price. This is a **regression** problem.

Data source: `vehicles.csv` (~426,000 Craigslist used-vehicle listings, US market). The raw file currently sits at the repository root; moving it to `data/raw/` is part of Phase 1. Never modify the raw file in place — cleaning always writes to `data/processed/`.

Columns: id, url, region, region_url, price, year, manufacturer, model, condition, cylinders, fuel, odometer, title_status, transmission, VIN, drive, size, type, paint_color, image_url, description, county, state, lat, long, posting_date.

### Critical Data Rules
- **DROP identifier / non-predictive columns** before modeling: id, url, region_url, VIN, image_url, county (almost fully empty). These carry no generalizable signal and some are unique-per-row.
- **LEAKAGE — description text:** the `description` field frequently contains the price, MSRP, or dealer stock numbers. Do NOT feed raw description into the price model without stripping numeric/price content. Prefer excluding it from the model entirely unless a controlled, leakage-free feature is extracted.
- **REFERENCE DATE TRAP:** the data was collected around 2021 (`posting_date`). Vehicle age MUST be computed as `posting_year - year` (or relative to a fixed reference year such as 2021), NEVER relative to today's real-world date. Using the current year silently corrupts the age feature.
- **Price filtering:** `price` has junk values (0, 1, and absurdly high entries). Define and document explicit lower/upper bounds (e.g. drop price <= a floor and above a high percentile). State the thresholds and the rationale.
- **Odometer & year filtering:** remove impossible values (e.g. year in the future relative to posting, odometer of 0 on old cars, extreme mileage outliers). Document every rule.
- **Missing data:** a large share of rows have missing fields. Document the strategy per column (drop vs impute) and why. Do not silently drop half the dataset without justifying it.
- **Duplicates:** `id` is unique per posting, so real duplicates are re-posted listings — deduplicate by VIN where present, and by (description, price, model) fingerprint otherwise. Document how many rows were removed.
- **Fit on train only:** all imputers, scalers, and encoders must be fitted on training data only, inside a pipeline, to avoid leakage into the test set.

### Key Features to Engineer
- `age` = posting_year - year (see reference-date rule above).
- `odometer` transforms / buckets; consider mileage-per-year (`odometer / age`).
- Log-transform of the target price (right-skewed) — see Modeling.
- High-cardinality categoricals (`model` has thousands of values, `manufacturer` dozens): use frequency or target encoding, NOT naive one-hot on `model`.
- Low-cardinality categoricals (condition, fuel, transmission, drive, type, title_status, cylinders, paint_color): one-hot or ordinal where a natural order exists (e.g. condition).
- Optional geographic signal from `state` (regional price differences), encoded, not raw lat/long unless justified.

## 5. Modeling Strategy
This is a price-regression problem. The target is right-skewed, so distribution handling matters.

### Approach
1. **Target transform:** train on `log1p(price)`; invert with `expm1` before reporting error metrics in real dollar units.
2. **Baseline:** Linear Regression (interpretable reference). Sanity-check coefficient signs (older/higher-mileage -> lower price).
3. **Advanced:** tree-based models — Random Forest, then gradient boosting (XGBoost / LightGBM). Tune hyperparameters via cross-validation.
4. **Model comparison:** compare all models on the same held-out test set with the same metrics. Justify the final choice with numbers, not preference.
5. **Interpretability:** feature importances (and optionally SHAP) on the final model. Age, mileage, make/model, and condition should dominate — if they do not, investigate for a data or encoding bug.

### Metrics (in priority order, reported on the original price scale)
1. **RMSE** (primary — penalizes large dollar errors).
2. **MAE** (robust, easy to communicate: "average error in dollars").
3. **MAPE** (percentage error — best for business framing).
4. **R^2** (variance explained, secondary).
- Always report metrics after inverting the log transform, in dollars. Do NOT report error only in log space.

### Validation
- Hold out a proper test set (random split is acceptable here; each row is an independent listing). Use cross-validation on the training set for tuning.
- Do error analysis: where is the model worst (price segment, brand, age)? Document failure modes — this is what a reviewer looks for.

### Anomaly Detection (Deliverable 4)
Runs AFTER the price model is finalized. Two complementary approaches:
1. **Residual-based:** listings where |actual - predicted| (in log space or percentage terms) exceeds a documented threshold are flagged as mispriced. A car listed far below its predicted value is a fraud/scam signal; far above is a data-entry or spam signal.
2. **Unsupervised:** Isolation Forest (or similar) on the engineered feature space to catch structurally weird listings (impossible year/odometer/price combinations that survived cleaning).
- There are no ground-truth fraud labels. Do NOT invent accuracy metrics; evaluate by inspecting and documenting the top-N flagged listings with a qualitative explanation of why each looks suspicious.
- Deliver a ranked "suspicious listings" report in reports/ — this is the business-facing output.
- Keep anomaly flags OUT of the price model's training data pipeline decisions; do not retroactively drop training rows based on model residuals without documenting it as a separate, justified iteration.

## 6. Environment
Windows + PowerShell. Python 3.10+ virtual environment. Activate before running anything:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Key dependencies: pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn, scipy, pyarrow. (ipykernel/nbformat/nbclient for notebook execution; xgboost and category_encoders were dropped -- never imported, LightGBM chosen as the GBDT and encoders are custom in src/models/encoders.py.) Test suite: pytest, pytest-cov (`python -m pytest` from the repo root; tests/ mirrors src/, no dependency on the real 426k-row CSV).

Caveat: this repository lives inside a OneDrive-synced folder with non-ASCII characters in the path. If venv or pip misbehaves (file locks, path errors), create the venv OUTSIDE OneDrive (e.g. `C:\venvs\usedcar`) and point to it — do not fight sync issues silently.

### Reproducibility (mandatory)
- Every random operation (train/test split, CV, model seeds, sampling) uses a fixed `random_state = 42`. No exceptions.
- `requirements.txt` must pin exact versions (`pandas==X.Y.Z`), so a reviewer can reproduce results.
- Scripts must be re-runnable end to end: `clean_data.py -> train.py -> evaluate.py` from a fresh clone must reproduce the reported metrics.

## 7. Final Packaging (last phase, before handoff)
The project is handed to a data team lead for review. The final phase produces:
- A root `README.md`: project summary, how to run, final metrics table, and 3-5 headline business insights. Written for a reviewer who has 10 minutes.
- `reports/` contains exported figures and the suspicious-listings report.
- One presentation notebook in `notebooks/` that tells the full story (cleaning decisions -> insights -> model comparison -> anomalies), importing logic from src/, never redefining it.

## 8. Out of Scope
- No deep learning / NLP on descriptions unless explicitly requested (and only leakage-free).
