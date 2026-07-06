# Phase 8 Results -- Code Quality Refactor + Test Suite

**Status:** DONE. All 14 sub-steps (8.0-8.13) executed and committed
individually; this document is the final review (8.14).

## Summary

122 tests added across 19 test files (previously: zero tests, zero `tests/`
directory, zero pytest dependency). Every src/ module touched in this phase
got its tests written in the same commit as the change, per CLAUDE.md's
Agent Workflow Protocol -- not deferred to one giant test-writing pass at
the end.

## What changed, by sub-step

| Step | Change | Commit |
|---|---|---|
| 8.0 | pytest test infrastructure (`pytest.ini`, `tests/conftest.py`, pinned `pytest==9.1.1`/`pytest-cov==7.1.0`) | `6a490b5` |
| 8.1 | `src/config.py` -- single source of truth for RANDOM_STATE, price/age segment bins, anomaly z-threshold, interval alpha | `236e4ba` |
| 8.2 | Parametrized every hardcoded bin/list/threshold in `src/evaluation/plots.py` | `87d322a` |
| 8.3 | Removed 2 confirmed-unused dead constants (`audit.py::DROP_COLS`, `dataset.py::DROP_BEFORE_MODEL`) | `b0a84b1` |
| 8.4 | Consolidated 5 independent `RANDOM_STATE = 42` definitions into one import | `850f17d` |
| 8.5 | Unified the duplicated LGBM-fit-with-early-stopping helper (`train.py`/`intervals.py`) | `e2a4dc2` |
| 8.6 | Relocated `coverage()`/`coverage_by_segment()` from `models/intervals.py` to `evaluation/metrics.py` | `ef8a854` |
| 8.7 | Parametrized `src/preprocess/cleaner.py` (DROP_COLS/CORE_COLS/cast-cols as constructor args, fixed mutable-default bug) | `9e76d70` |
| 8.8 | Parametrized `src/models/dataset.py::select_features` | `fab8758` |
| 8.9 | Parametrized `src/features/engineer.py` (REFERENCE_YEAR/AGE_MIN -- the reference-date-trap constant) | `24e274b` |
| 8.10 | Cleaned up `src/models/encoders.py` (KFold seed as param, type hints, extracted duplicated fillna block) | `52cdbeb` |
| 8.11 | Google-style docstrings on every remaining public function/class in src/ | `6aa1116` |
| 8.12 | Thinned all 6 in-scope pipeline scripts; extracted business logic into `src/evaluation/insights.py`, `src/evaluation/reporting.py`, `src/anomaly/detector.py` additions | `0d9dce4` |
| 8.13 | Synced CLAUDE.md / docs/plan.md project-structure sections with reality | `95cefab` |

## Success criteria (from docs/phase8_plan.md) -- verified

- **`pytest` passes, 0 failures:** 122 passed (`python -m pytest -q`).
- **Zero references remain to the two dead constants, the 5 independent
  `RANDOM_STATE` definitions, or the duplicated LGBM-fit scaffolding:**
  grep-verified -- `DROP_COLS`/`DROP_BEFORE_MODEL` gone; `RANDOM_STATE` defined
  exactly once (`src/config.py`); `_fit_lgbm_quantile` now delegates to
  `fit_lgbm_with_early_stopping`, not a second implementation.
- **Every src/ public function/class has a Google-style docstring:**
  AST sweep of the entire src/ tree returns 0 missing.
- **Every parametrized function's default value is unchanged:** enforced by
  the regression test suites written alongside each parametrization
  (`test_cleaner.py`, `test_dataset.py`, `test_engineer.py`, `test_encoders.py`,
  `test_config.py`'s signature-default-matches-config guard).
- **All 6 in-scope scripts import cleanly and contain no business logic
  beyond instantiate/call/save + CLI concerns:** `tests/scripts/test_imports.py`
  (6/6 pass) plus a `write_results` smoke test per script exercising the new
  src/ wiring with synthetic data.
- **CLAUDE.md and docs/plan.md project-structure sections match the repo:**
  done in 8.13.

## Accepted risk (restated)

Per the user's explicit choice (2026-07-07), verification was unit-tests-only
-- **no full 426,880-row pipeline rerun was performed.** The currently
committed `reports/suspicious_listings.csv`, `docs/phase3_results.md`,
`docs/phase4_results.md`, `docs/phase6_results.md`, `docs/phase7_results.md`,
and the README metrics table are unchanged from their pre-Phase-8 state.
Unit tests pin default values and behavior at the function/class level (price
filter bounds, age computation against the 2021 reference year, split
proportions, dedup rules, etc.), which catches an accidental default change,
but does not re-confirm the actual end-to-end dollar metrics still reproduce
bit-for-bit.

**Recommendation (not a Phase 8 blocker):** the user should manually run
`clean_data.py -> run_eda.py -> train.py -> detect_anomalies.py ->
predict_intervals.py -> ablation_description_features.py` once against the
real data at their convenience, as a real-world sanity check that the
refactor is truly behavior-preserving end to end.

## Out of scope (deferred, unchanged from the plan)

`check_consistency.py`, `probe_attrition.py`, `probe_description_signal.py`,
`probe_junk_rules.py`, `probe_mondrian_conditional_coverage.py`,
`probe_split_leakage.py` were left exactly as-is, per the locked-in scope
decision. `probe_mondrian_conditional_coverage.py` still has its own inline
copy of the price-segment bin literal -- this is expected and untouched
(confirmed via the final grep sweep); it is one of the explicitly deferred
scripts, not a missed cleanup.

## Test suite inventory

| File | Tests |
|---|---|
| `tests/test_config.py` | 5 |
| `tests/preprocess/test_cleaner.py` | 11 |
| `tests/features/test_engineer.py` | 11 |
| `tests/features/test_description.py` | 6 |
| `tests/models/test_dataset.py` | 6 |
| `tests/models/test_encoders.py` | 9 |
| `tests/models/test_train.py` | 7 |
| `tests/models/test_intervals.py` | 3 |
| `tests/evaluation/test_metrics.py` | 10 |
| `tests/evaluation/test_plots.py` | 14 |
| `tests/evaluation/test_insights.py` | 3 |
| `tests/evaluation/test_reporting.py` | 17 |
| `tests/anomaly/test_detector.py` | 10 |
| `tests/scripts/test_imports.py` | 6 |
| `tests/scripts/test_train_script.py` | 1 |
| `tests/scripts/test_detect_anomalies_script.py` | 1 |
| `tests/scripts/test_predict_intervals_script.py` | 1 |
| `tests/scripts/test_ablation_description_features_script.py` | 1 |
| **Total** | **122** |
