"""Import-smoke tests for the 6 in-scope pipeline scripts (Phase 8.12).

Confirms each script's wiring to src/ resolves without error after the
Phase 8 extraction work, without running the actual 426k-row pipeline
(module-level code only defines constants/functions; `main()` is guarded by
`if __name__ == "__main__":` in every script).
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

SCRIPT_MODULES = [
    "scripts.clean_data",
    "scripts.run_eda",
    "scripts.train",
    "scripts.detect_anomalies",
    "scripts.predict_intervals",
    "scripts.ablation_description_features",
]


@pytest.mark.parametrize("module_name", SCRIPT_MODULES)
def test_script_imports_cleanly(module_name):
    importlib.import_module(module_name)
