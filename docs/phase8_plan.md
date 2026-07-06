# Phase 8 Plan -- Code Quality Refactor + Test Suite

## Motivation

CLAUDE.md section 3 was tightened with explicit "Code cleanliness rules": src/
must be generic/parametric (no hardcoded columns/paths/magic numbers), one
responsibility per subpackage, scripts/ must be thin orchestration only,
docstrings on every public function/class, no dead code. A full read-only audit
of src/ (11 files, ~1,400 lines) and scripts/ (12 files, ~2,700 lines) was run
against these rules (two independent agent passes). This phase executes the
fixes the audit surfaced, adds the test suite the project has never had (zero
`tests/` directory, zero pytest dependency today), and updates CLAUDE.md/plan.md
where they've drifted from what actually got built in phases 6-7.

**This is a structural refactor, not a modeling change.** No feature, threshold,
or hyperparameter default may change value -- only where it lives and how it's
named/parametrized. Phases 1-7 results (RMSE $6,611 / MAPE 36.9% baseline,
A4 desc_* improvement, anomaly tiers, conformal intervals) must remain
reproducible from the exact same defaults.

## Decisions locked in (per user approval, 2026-07-07)

| Decision | We choose | Alternative | Reason |
|---|---|---|---|
| Scope | src/ (all 5 subpackages) + the 6 real pipeline scripts (`clean_data.py`, `run_eda.py`, `train.py`, `detect_anomalies.py`, `predict_intervals.py`, `ablation_description_features.py`) | Also refactor the 6 probe_*/check_consistency.py scripts | Probes are closed, documented, one-off investigations (Problem Solving Framework artifacts) -- refactoring them touches code tied to closed phase docs for no metric benefit. `check_consistency.py` doesn't cleanly fit any existing src/ subpackage; deferred, not in scope. |
| Docstring style | Google (`Args:` / `Returns:` / `Raises:`) | NumPy (`Parameters` / `Returns` underlined sections) | Only 2 of 11 src/ files currently use NumPy style (`metrics.py`, part of `encoders.py`); Google is the smaller migration and more compact. |
| Behavior verification | Unit tests only, no full 426k-row pipeline rerun | Full rerun of clean_data->train->detect_anomalies->predict_intervals, diff every reported metric | Avoids multi-minute LightGBM retraining cycles during a pure structural refactor. **Accepted risk, stated explicitly below.** |
| Shared constants | New `src/config.py` (RANDOM_STATE, price/age segment bins, anomaly z-threshold, interval alpha) | Keep per-module local constants, just parametrize them | Removes the 5x independent `RANDOM_STATE = 42` redefinition and 4x duplicated price-bin literal the audit flagged as drift risk. |

### Accepted risk: no full-data rerun

Since verification is unit-tests-only, the currently committed
`reports/suspicious_listings.csv`, `docs/phase3_results.md`,
`docs/phase4_results.md`, `docs/phase6_results.md`, and README metrics table are
**not regenerated** as part of this phase -- they stay exactly as last produced.
Unit tests pin default values and behavior at the function/class level (e.g.
"age = REFERENCE_YEAR - year when posting_date absent", "price filter keeps
[500, 150000]"), which catches an accidental default change, but does not
re-confirm the actual end-to-end dollar metrics still reproduce. Recommend the
user manually re-run the 4 pipeline scripts once at their own convenience after
this phase lands, as a real-world sanity check -- optional, not a Phase 8 success
criterion.

---

## Audit summary (condensed -- full detail was gathered via two read-only agent audits of src/ and scripts/)

### src/ findings
- **Dead code (2 items, confirmed unused, safe to delete):** `src/preprocess/audit.py::DROP_COLS` (drifted duplicate of `cleaner.py`'s list, never referenced); `src/models/dataset.py::DROP_BEFORE_MODEL` (superseded by the whitelist approach in `select_features`, never referenced).
- **`RANDOM_STATE = 42` independently redefined in 5 files:** `models/dataset.py`, `models/encoders.py`, `models/intervals.py`, `models/train.py`, `anomaly/detector.py`.
- **Duplicated LGBM-fit-with-early-stopping helper:** `models/train.py::_fit_lgbm` and `models/intervals.py::_fit_lgbm_quantile` are near-identical (`intervals.py`'s own docstring says "Mirrors src.models.train._fit_lgbm").
- **Evaluation-shaped functions outside evaluation/:** `coverage()` / `coverage_by_segment()` live in `models/intervals.py`, structurally identical in spirit to `evaluation/metrics.py::error_by_segment`.
- **Duplicated magic numbers with no shared source:** `z_threshold=3.5` in both `anomaly/detector.py` and `evaluation/plots.py::plot_anomaly_overview`; `alpha=0.10` (`intervals.py`) vs `nominal=0.90` (`plots.py`) express the same concept independently.
- **Inconsistent parametrization within single classes/files** (not just across files): `DataCleaner` parametrizes thresholds but not `DROP_COLS`/`CORE_COLS`/the cast-types column list; `FeatureEngineer` has zero constructor parameters (`REFERENCE_YEAR`, `AGE_MIN` are unreachable module globals -- the single highest-risk item, since `REFERENCE_YEAR` is exactly CLAUDE.md's named "reference date trap" constant); `dataset.py::select_features` parametrizes nothing while `build_split` in the same file parametrizes everything; `plots.py` parametrizes some per-function thresholds but not others.
- **Docstrings:** inconsistent style (NumPy in 2 files, prose elsewhere, several public functions with zero docstring -- notably `train_linear`/`train_rf`/`train_lgbm`, the three core model-training entry points).
- **Minor:** mutable default argument in `DataCleaner.__init__` (`valid_title_status: set = VALID_TITLE_STATUS`); a few `y: pd.Series = None` type hints that should be `pd.Series | None = None`.
- **No commented-out code / TODO markers anywhere in src/.**

### scripts/ findings
- CLAUDE.md's canonical scripts list (section 3) only names 4 files; `predict_intervals.py` and `ablation_description_features.py` are real, re-runnable Phase 6/7 deliverables missing from that list (doc drift, fixed in 8.13).
- `clean_data.py` is already a clean orchestration template -- no changes needed.
- `run_eda.py::compute_insights()` (~100 lines): real groupby/correlation/pivot statistics computed directly in the script body, zero calls into `src/evaluation`.
- `train.py`: inline feature-importance gain-table extraction; hardcoded (unnamed) price/age bin literals; three separate "compare metric to threshold, branch to canned verdict paragraph" functions (`_a1_note`, `_a3_note`, `_model_comparison_note`).
- `detect_anomalies.py::in_sample_residual_std()`: a **complete model-fitting routine written directly in the script** (imports lightgbm inline, fits an `LGBMRegressor`) -- the clearest single "no model-fitting internals in a script" violation found. Also: STRONG/MODERATE tiering logic (a real business rule, `np.select`) lives inline; a fat-tail Gaussian-expected-count probe is computed inline.
- `predict_intervals.py`: reimplements the STRONG-tier formula from `detect_anomalies.py` independently (same constants, no shared function); its own `_segment_verdict()` duplicates the threshold-verdict pattern; its idempotent "read doc, split on header, rewrite" section-replace code is copy-pasted into `ablation_description_features.py`.
- `ablation_description_features.py`: same doc-section-replace duplication; its own bespoke verdict-threshold block; architecturally a 4th ablation built as a standalone script instead of a function alongside A1-A3 in `train.py`.
- Cross-cutting: the price-segment bin `[0,5000,10000,20000,50000,150000]` and age-bucket bin `[0,3,6,10,15,60]` are retyped as inline literals (not shared constants) in `train.py`, `predict_intervals.py`, `ablation_description_features.py`, and (age, with different labels) `run_eda.py`.

---

## Testing infrastructure (new)

- Add to `requirements.txt` (dev-only, still pinned): `pytest==8.4.2`, `pytest-cov==6.0.0` (exact versions confirmed at install time against what actually resolves in the venv -- CLAUDE.md's reproducibility rule applies to test deps too).
- New `tests/` tree mirroring `src/`: `tests/preprocess/`, `tests/features/`, `tests/models/`, `tests/evaluation/`, `tests/anomaly/`, `tests/conftest.py` (shared fixtures: small synthetic `pd.DataFrame`s, no file I/O against the real 426k-row CSV, no reliance on wall-clock or unseeded randomness -- every seeded call passes `random_state` explicitly, matching CLAUDE.md's reproducibility rule).
- `tests/scripts/test_imports.py`: one lightweight test per pipeline script asserting it imports without error -- catches broken wiring from the extraction work in 8.12 without running the real pipeline (respects the "unit tests only" verification decision).
- `pyproject.toml` or `pytest.ini` (whichever is less invasive given no existing packaging config): `testpaths = tests`, rootdir pinned.
- Every src/ module gets its test file written in the SAME sub-phase it's refactored in (not deferred to one giant test-writing phase at the end) -- this is what makes each sub-phase individually verifiable per CLAUDE.md's Agent Workflow Protocol ("Verify... before declaring a phase done").

---

## src/config.py design (new module, written first -- everything else imports from it)

```python
RANDOM_STATE: int = 42

PRICE_SEGMENT_BINS: list[float] = [0, 5_000, 10_000, 20_000, 50_000, 150_000]
PRICE_SEGMENT_LABELS: list[str] = ["<5k", "5-10k", "10-20k", "20-50k", "50-150k"]

AGE_BUCKET_BINS: list[float] = [0, 3, 6, 10, 15, 60]
AGE_BUCKET_LABELS: list[str] = ["0-3", "3-6", "6-10", "10-15", "15+"]

ANOMALY_Z_THRESHOLD: float = 3.5
INTERVAL_ALPHA: float = 0.10  # 90% nominal coverage
LGBM_QUANTILE_N_ESTIMATORS: int = 3000
```
Values copied verbatim from current defaults -- this is a rename/relocation, not
a value change. Every module that currently hardcodes one of these imports it
from here instead; every function/constructor parameter that currently defaults
to a bare literal now defaults to the config constant (still overridable).
`tests/test_config.py` asserts each value's type and, via `inspect.signature`,
that every consumer function's default still resolves to the config value (this
is the automated guard against the exact "silent drift across 5 files" pattern
the audit found).

---

## Execution plan (ordered by risk, lowest first; each numbered step is a checkpoint -- run its tests, then move on)

**8.1 -- `src/config.py`** (new file, no behavior change yet since nothing imports it)
Write the module + `tests/test_config.py`.

**8.2 -- `src/evaluation/plots.py`** (zero model-behavior risk -- pure visualization)
Parametrize every hardcoded bin/list/threshold identified in the audit
(`plot_confound_check` age buckets, `plot_age_odometer_interaction` bins,
`plot_value_heaping` "interesting" endings, `plot_missingness_and_cardinality`
miss_cols + cardinality bins, `plot_correlation_heatmap` numeric column list,
`TOP_N_MANUFACTURERS`) as function parameters with current values as defaults
(sourced from `config.py` where the concept is shared, e.g. `z_threshold`
defaults to `config.ANOMALY_Z_THRESHOLD`, `nominal` defaults to
`1 - config.INTERVAL_ALPHA`). Convert docstrings to Google style.
Write `tests/evaluation/test_plots.py` (smoke tests: figure returned with
expected axes, `_save` creates parent dirs / no-ops on `None` path, deterministic
sampling with fixed `random_state`, no exception on edge-case input like a
manufacturer with a gap in its age range).

**8.3 -- Dead code removal**
Delete `preprocess/audit.py::DROP_COLS` and `models/dataset.py::DROP_BEFORE_MODEL`
after re-confirming zero references (grep). No test needed beyond existing
import-smoke coverage.

**8.4 -- `RANDOM_STATE` consolidation**
Remove the 5 local `RANDOM_STATE = 42` constants; import from `config.py`.
Extend `tests/test_config.py` with the signature-default-matches-config check
described above.

**8.5 -- Unify the duplicated LGBM-fit-with-early-stopping helper**
Extract one shared function (`src/models/train.py::fit_lgbm_with_early_stopping`,
parametrized by objective/params/n_estimators/random_state/test_size=0.1) used by
both `train.py`'s point-regression path and `intervals.py`'s quantile path.
Write `tests/models/test_train.py` covering the shared helper (deterministic
with fixed seed, correct validation-split proportion) before touching
`intervals.py`'s call site, so the extraction is tested against the original
behavior first.

**8.6 -- Relocate `coverage()` / `coverage_by_segment()`**
Move from `models/intervals.py` into `src/evaluation/metrics.py` (same module as
`error_by_segment`, same responsibility); update the one import site in
`predict_intervals.py`. Add to `tests/evaluation/test_metrics.py`.

**8.7 -- `src/preprocess/cleaner.py` parametrization** (touches the cleaning pipeline -- higher care)
`DROP_COLS`/`CORE_COLS`/the cast-types categorical-column list become
`__init__` parameters defaulting to the current lists; fix the mutable-default
`valid_title_status` argument (`None` sentinel pattern). Write the full test
list from the audit: price/year/odometer bound filters (inclusive boundaries),
VIN-exact dedup, fingerprint dedup restricted to no-VIN rows, title-status
keep/drop incl. null passthrough, core-null drop, empty-df `retention_pct()`
edge case. These tests are what proves the parametrization defaults are
byte-identical to current behavior.

**8.8 -- `src/models/dataset.py::select_features` parametrization**
Add `numeric_features`/`categorical_features` parameters defaulting to the
current `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` lists. Write
`tests/models/test_dataset.py`: leakage columns excluded, missing-column
tolerance, `build_split` size ratios and test-set stability across
`val_size_of_remainder` changes, `split_calibration` disjointness.

**8.9 -- `src/features/engineer.py` parametrization** (highest risk -- reference-date trap)
Add `__init__(self, reference_year: int = 2021, age_min: int = AGE_MIN)`;
`_add_age` uses `self.reference_year`/`self.age_min` instead of module globals.
Write `tests/features/test_engineer.py` BEFORE changing the method body,
pinning exact expected `age` values for known year/posting_date fixtures --
this test must fail against a naive `datetime.now().year` implementation and
pass against the documented `2021` default, so it stands as a permanent guard
against the reference-date trap regressing. Also covers: posting_date present
vs absent, negative-age clipping to `age_min`, `mileage_per_year` division,
cylinder-string extraction, description-feature passthrough/no-op when the
column is absent.

**8.10 -- `src/models/encoders.py` cleanup**
Expose the `SafeTargetEncoder` KFold shuffle seed as a constructor parameter
(default `config.RANDOM_STATE`); fix `y: pd.Series | None = None` type hints;
extract the repeated high-card-columns-fillna block in `FeaturePreprocessor`
into a private helper. Write `tests/models/test_encoders.py`: OOF encoding
doesn't leak a row's own target, smoothing shrinks rare categories toward the
global mean, unseen-category fallback, frequency-encoder correctness, missing
indicator only varies for columns that had nulls, target-vs-frequency method
switch produces different but deterministic encodings.

**8.11 -- Docstring pass (Google style), remaining gaps**
Apply Args/Returns/Raises to every remaining public function/class without one
(notably `train_linear`/`train_rf`/`train_lgbm`, `DataCleaner`/`FeatureEngineer`
class-level docs, `ConformalIntervalModel`/`MondrianConformalIntervalModel`
constructors). Purely additive, no behavior change, last in the src/ pass per
the audit's own risk ordering.

**8.12 -- scripts/ thinning** (the 6 in-scope scripts)
- `run_eda.py`: extract `compute_insights()` into a new
  `src/evaluation/insights.py::compute_eda_insights(df) -> dict` (parametric,
  generic); script calls it and writes the markdown.
- `train.py`: extract gain-importance extraction into
  `src/evaluation/metrics.py::gain_importance_table(model, top_n=15)`; import
  price/age bins from `config.py`; consolidate the three verdict-from-threshold
  functions into one shared `src/evaluation/reporting.py::threshold_verdict(...)`.
- `detect_anomalies.py`: move `in_sample_residual_std()` into
  `src/models/train.py` (or `src/anomaly/detector.py` -- decide at implementation
  time based on which reads more naturally, document the choice); move
  STRONG/MODERATE tiering into a new `ResidualAnomalyDetector.tier()` method
  (thresholds sourced from `config.py` or explicit method params); move the
  fat-tail Gaussian-expected-count check into `src/anomaly/` or
  `src/evaluation/`.
- `predict_intervals.py`: import segment bins from `config.py`; call
  `ResidualAnomalyDetector.tier()` (from 8.12's detect_anomalies work) instead
  of reimplementing the STRONG-tier formula; use the shared `threshold_verdict`
  helper; use a new `src/evaluation/reporting.py::replace_doc_section(path,
  header, new_content)` for the idempotent markdown-section rewrite (shared with
  `ablation_description_features.py`).
- `ablation_description_features.py`: use the same shared `threshold_verdict`
  and `replace_doc_section` helpers.
- Every newly extracted src/ function gets a unit test (insights, gain-importance
  table, `threshold_verdict`, `replace_doc_section`, `.tier()`).
- `tests/scripts/test_imports.py`: one import-smoke test per script (all 6),
  confirming the extraction didn't break wiring, without a full data run.

**8.13 -- Docs sync**
Update CLAUDE.md section 3's project-structure list to include
`predict_intervals.py`, `ablation_description_features.py`, and `src/config.py`.
Update `docs/plan.md`'s "Project Structure" tree the same way. No change to
historical phase3-7 results docs' conclusions (numbers stay as last generated,
per the accepted-risk note above).

**8.14 -- Final review**
Full `pytest` run green. Re-grep for the specific violations the audit found
(dead constants, duplicated `RANDOM_STATE`, duplicated LGBM-fit scaffolding) to
confirm each is actually gone. Write `docs/phase8_results.md` recording: test
count added, files touched, and the accepted-risk note restated (no full-data
rerun was performed).

---

## Success criteria

- `pytest` passes, 0 failures, covering every src/ module touched above.
- Zero references remain to the two dead constants, the 5 independent
  `RANDOM_STATE` definitions, or the duplicated LGBM-fit scaffolding (grep-verified).
- Every src/ public function/class has a Google-style docstring.
- Every parametrized function's default value is unchanged from what's
  currently hardcoded (this is what "behavior-preserving" means operationally
  here, given the no-full-rerun decision).
- All 6 in-scope scripts import cleanly and contain no business logic beyond
  instantiate/call/save + CLI concerns.
- CLAUDE.md and docs/plan.md project-structure sections match what's actually
  in the repo.

## Out of scope (deferred, not forgotten)

- `check_consistency.py`, `probe_attrition.py`, `probe_description_signal.py`,
  `probe_junk_rules.py`, `probe_mondrian_conditional_coverage.py`,
  `probe_split_leakage.py` -- left exactly as-is. If ever revisited: the audit
  already identified `check_consistency.py` as 100% extractable business logic
  with no current src/ home, and `probe_attrition.py`/`probe_split_leakage.py`
  as containing fully generic, already-parametric statistical helpers stranded
  in scripts/.
