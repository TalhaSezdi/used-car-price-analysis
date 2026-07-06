"""Guards for src/config.py -- the single source of truth for shared constants.

These pin the exact values that were previously scattered across 5+ files as
independent literals (see docs/phase8_plan.md). If one of these ever changes,
every reported metric in docs/phase3_results.md, phase4_results.md, and
phase6_results.md is potentially stale -- so a value change here must be a
deliberate, documented decision, not an accident.
"""

from src import config


def test_random_state_is_42():
    assert config.RANDOM_STATE == 42


def test_price_segment_bins_and_labels_match_current_defaults():
    assert config.PRICE_SEGMENT_BINS == [0, 5_000, 10_000, 20_000, 50_000, 150_000]
    assert config.PRICE_SEGMENT_LABELS == ["<5k", "5-10k", "10-20k", "20-50k", "50-150k"]
    assert len(config.PRICE_SEGMENT_LABELS) == len(config.PRICE_SEGMENT_BINS) - 1


def test_age_bucket_bins_and_labels_match_current_defaults():
    assert config.AGE_BUCKET_BINS == [0, 3, 6, 10, 15, 60]
    assert config.AGE_BUCKET_LABELS == ["0-3", "4-6", "7-10", "11-15", "16+"]
    assert len(config.AGE_BUCKET_LABELS) == len(config.AGE_BUCKET_BINS) - 1


def test_anomaly_and_interval_defaults():
    assert config.ANOMALY_Z_THRESHOLD == 3.5
    assert config.INTERVAL_ALPHA == 0.10
    assert config.LGBM_QUANTILE_N_ESTIMATORS == 3000


def _default_of(func, param_name):
    import inspect

    return inspect.signature(func).parameters[param_name].default


def test_random_state_consumers_default_to_config_value():
    """Guard against the 5-independent-definitions drift the Phase 8 audit
    found: every module that previously hardcoded its own `RANDOM_STATE = 42`
    now imports it from here. If a consumer's own default ever drifts from
    this value, that's exactly the kind of silent divergence this test
    exists to catch."""
    from src.models.dataset import split_calibration
    from src.models.intervals import _fit_lgbm_quantile

    assert _default_of(split_calibration, "random_state") == config.RANDOM_STATE
    assert _default_of(_fit_lgbm_quantile, "random_state") == config.RANDOM_STATE

    from src.models import dataset, encoders, intervals, train
    from src.anomaly import detector

    for module in (dataset, encoders, intervals, train, detector):
        assert not hasattr(module, "RANDOM_STATE") or module.RANDOM_STATE is config.RANDOM_STATE
