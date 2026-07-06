"""Cross-check headline metrics across README.md, docs/, and reports/.

Usage:
    python scripts/check_consistency.py

No model training -- reads markdown files only, parses metric tables,
and flags any number that appears in one place but differs in another.
Exit code 0 = all consistent, 1 = mismatches found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

# ── helpers ────────────────────────────────────────────────────────────

def _extract_md_tables(text: str) -> list[list[list[str]]]:
    """Return every markdown table as a list of rows (each row = list of cells)."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r"-+|:?-+:?", c) for c in cells):
                continue  # separator row
            current.append(cells)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    return tables


def _clean_num(s: str) -> Optional[float]:
    """Parse a number from markdown cell text (strip $, %, **, commas)."""
    s = s.replace("**", "").replace("$", "").replace("%", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _find_model_row(table: list[list[str]], model_name: str) -> Optional[list[str]]:
    """Find a row whose first cell contains model_name (case-insensitive)."""
    target = model_name.lower().replace("**", "").strip()
    for row in table:
        cell = row[0].replace("**", "").strip().lower()
        if target in cell:
            return row
    return None


# ── source of truth: docs/phase3_results.md ────────────────────────────
#
# phase3_results.md now has two comparable tables:
#   Table 1: model comparison on VALIDATION set (used for the selection decision)
#   Table 2: final LightGBM on TEST set (single-row "unbiased headline" table)
#
# Ablation rows (A1 log1p, A3 target_encoding) are all evaluated on VAL, so they
# must match Table 1's LightGBM row. Inline README dollar amounts, the phase6
# snapshot note, and the phase7 "with desc_*" row all refer to the final test
# metric and must match Table 2's LightGBM row.

def _parse_table(table: list[list[str]]) -> dict[str, dict[str, float]]:
    header = table[0]
    out: dict[str, dict[str, float]] = {}
    for row in table[1:]:
        name = row[0].replace("**", "").strip()
        metrics: dict[str, float] = {}
        for i, col in enumerate(header[1:], start=1):
            col_clean = col.replace("**", "").strip()
            val = _clean_num(row[i]) if i < len(row) else None
            if val is not None:
                metrics[col_clean] = val
        out[name] = metrics
    return out


def load_phase3_truth(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Return (val_truth, test_lgbm_truth) parsed from phase3_results.md.

    val_truth: {model_name: metrics_dict} from the validation-set comparison.
    test_lgbm_truth: metrics_dict for LightGBM on the test set (final headline).
    """
    text = path.read_text(encoding="utf-8")
    tables = _extract_md_tables(text)
    if not tables:
        print(f"  WARN: no tables found in {path.name}")
        return {}, {}

    val_truth = _parse_table(tables[0])
    # Table 2 is the single-row final headline table.
    test_lgbm: dict[str, float] = {}
    if len(tables) >= 2:
        parsed = _parse_table(tables[1])
        # first (only) model row -- allow "LightGBM (final)", "LightGBM", etc.
        for name, metrics in parsed.items():
            if "lightgbm" in name.lower():
                test_lgbm = metrics
                break
    return val_truth, test_lgbm


# ── checkers ──────────────────────────────────────────────────────────

MISMATCHES: list[str] = []


def _check_value(label: str, expected: float, actual: float, tol: float = 0.5) -> None:
    if abs(expected - actual) > tol:
        msg = f"  MISMATCH  {label}: expected {expected}, found {actual}"
        MISMATCHES.append(msg)
        print(msg)


def check_readme(val_truth: dict[str, dict[str, float]],
                 test_lgbm: dict[str, float]) -> None:
    """README has two tables: val comparison (matches val_truth) and final
    headline (matches test_lgbm's LightGBM row)."""
    readme = ROOT / "README.md"
    if not readme.exists():
        print("  WARN: README.md not found")
        return

    text = readme.read_text(encoding="utf-8")
    tables = _extract_md_tables(text)

    # Table with all three named models -> val comparison
    for model_name, expected in val_truth.items():
        found = False
        for table in tables:
            row = _find_model_row(table, model_name)
            if row is None:
                continue
            # Skip tables that only have LightGBM (that's the final headline).
            row_names = [r[0].replace("**", "").strip().lower() for r in table[1:]]
            has_multiple = sum(1 for n in row_names if any(
                m in n for m in ("linear", "forest", "lightgbm"))) > 1
            if not has_multiple:
                continue
            found = True
            header = table[0]
            for i, col in enumerate(header[1:], start=1):
                col_clean = col.replace("**", "").strip()
                if col_clean in expected and i < len(row):
                    val = _clean_num(row[i])
                    if val is not None:
                        _check_value(
                            f"README val / {model_name} / {col_clean}",
                            expected[col_clean], val,
                        )
            break
        if not found:
            print(f"  WARN: {model_name} not found in README val tables")

    # Single-row LightGBM table -> final test headline
    for table in tables:
        row_names = [r[0].replace("**", "").strip().lower() for r in table[1:]]
        has_only_lgbm = (
            len(row_names) == 1 and "lightgbm" in row_names[0]
        )
        if not has_only_lgbm:
            continue
        row = table[1]
        header = table[0]
        for i, col in enumerate(header[1:], start=1):
            col_clean = col.replace("**", "").strip()
            if col_clean in test_lgbm and i < len(row):
                val = _clean_num(row[i])
                if val is not None:
                    _check_value(
                        f"README test / LightGBM / {col_clean}",
                        test_lgbm[col_clean], val,
                    )
        break


def check_phase7(test_lgbm: dict[str, float]) -> None:
    """phase7 'with desc_*' refits on 80% (train+val) and evaluates on test,
    which is exactly what the final LightGBM does. So it must match test_lgbm."""
    path = ROOT / "docs" / "phase7_results.md"
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    tables = _extract_md_tables(text)
    if not test_lgbm:
        return

    for table in tables:
        row = _find_model_row(table, "with desc")
        if row is None:
            continue
        header = table[0]
        for i, col in enumerate(header[1:], start=1):
            col_clean = col.replace("**", "").strip()
            if col_clean in test_lgbm and i < len(row):
                val = _clean_num(row[i])
                if val is not None:
                    _check_value(
                        f"phase7 'with desc_*' / {col_clean}",
                        test_lgbm[col_clean], val,
                    )
        break


def check_inline_numbers(test_lgbm: dict[str, float]) -> None:
    """Scan prose in README for dollar amounts that should match the test-set
    LightGBM RMSE (the "headline number" a reader will parse from prose)."""
    if not test_lgbm:
        return
    rmse = test_lgbm.get("RMSE ($)")
    readme = ROOT / "README.md"
    if not readme.exists():
        return

    text = readme.read_text(encoding="utf-8")

    # find all $X,XXX patterns near "RMSE" or "LightGBM"
    # skip comparative sentences ("$X vs $Y") -- those reference two models
    for m in re.finditer(r"\$(\d{1,3}(?:,\d{3})*)", text):
        val = float(m.group(1).replace(",", ""))
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        context = text[start:end].replace("\n", " ")
        if rmse and abs(val - rmse) > 1 and 5000 < val < 10000:
            # skip if this is a "vs" comparison (e.g. "$5,765 vs $6,261")
            if " vs " in context[m.start() - start:]:
                continue
            if any(kw in context.lower() for kw in ["rmse", "lightgbm"]):
                _check_value(
                    f"README inline ('{context.strip()[:60]}...')",
                    rmse, val, tol=1.0,
                )


def check_phase6_snapshot_note(test_lgbm: dict[str, float]) -> None:
    """Verify phase6_results.md snapshot note references the current test RMSE."""
    path = ROOT / "docs" / "phase6_results.md"
    if not path.exists():
        return
    rmse = test_lgbm.get("RMSE ($)")
    if rmse is None:
        return

    text = path.read_text(encoding="utf-8")
    m = re.search(r"test RMSE is \$(\d{1,3}(?:,\d{3})*)", text)
    if m:
        val = float(m.group(1).replace(",", ""))
        _check_value("phase6 snapshot note RMSE", rmse, val, tol=1.0)


def check_ablation_row(row_key: str, label: str,
                       val_truth: dict[str, dict[str, float]]) -> None:
    """A1 log1p and A3 target_encoding are both computed on the VAL set with
    the same LightGBM pipeline as the val comparison table -- so they must
    equal the val LightGBM row."""
    phase3 = ROOT / "docs" / "phase3_results.md"
    if not phase3.exists():
        return
    lgbm_val = val_truth.get("LightGBM", {})
    if not lgbm_val:
        return

    text = phase3.read_text(encoding="utf-8")
    tables = _extract_md_tables(text)

    for table in tables:
        row = _find_model_row(table, row_key)
        if row is None:
            continue
        header = table[0]
        for i, col in enumerate(header[1:], start=1):
            col_clean = col.replace("**", "").strip()
            if col_clean in lgbm_val and i < len(row):
                val = _clean_num(row[i])
                if val is not None:
                    _check_value(
                        f"{label} / {col_clean} vs val LightGBM",
                        lgbm_val[col_clean], val,
                    )
        break


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Consistency Check ===\n")
    print("Source of truth: docs/phase3_results.md (model comparison table)\n")

    phase3 = ROOT / "docs" / "phase3_results.md"
    if not phase3.exists():
        print("ERROR: docs/phase3_results.md not found. Run train.py first.")
        sys.exit(1)

    val_truth, test_lgbm = load_phase3_truth(phase3)
    if not val_truth or not test_lgbm:
        print("ERROR: could not parse val comparison and/or test headline tables.")
        sys.exit(1)

    print("Parsed val comparison:")
    for name, metrics in val_truth.items():
        vals = ", ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"  {name}: {vals}")
    print("Parsed final test LightGBM:")
    print("  " + ", ".join(f"{k}={v}" for k, v in test_lgbm.items()))
    print()

    print("[1] README.md tables vs phase3_results.md")
    check_readme(val_truth, test_lgbm)

    print("[2] phase7_results.md 'with desc_*' row vs test LightGBM")
    check_phase7(test_lgbm)

    print("[3] Inline dollar amounts in README vs test RMSE")
    check_inline_numbers(test_lgbm)

    print("[4] phase6_results.md snapshot note vs test RMSE")
    check_phase6_snapshot_note(test_lgbm)

    print("[5] A1 ablation log1p row == val LightGBM")
    check_ablation_row("log1p", "A1 log1p row", val_truth)

    print("[6] A3 ablation target_encoding row == val LightGBM")
    check_ablation_row("target_encoding", "A3 target_encoding", val_truth)

    print()
    if MISMATCHES:
        print(f"FAILED: {len(MISMATCHES)} mismatch(es) found.")
        sys.exit(1)
    else:
        print("PASSED: all cross-checked numbers are consistent.")
        sys.exit(0)


if __name__ == "__main__":
    main()
