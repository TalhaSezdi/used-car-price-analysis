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

def load_phase3_truth(path: Path) -> dict[str, dict[str, float]]:
    """Parse the model comparison table from phase3_results.md."""
    text = path.read_text(encoding="utf-8")
    tables = _extract_md_tables(text)
    if not tables:
        print(f"  WARN: no tables found in {path.name}")
        return {}

    header = tables[0][0]
    truth: dict[str, dict[str, float]] = {}
    for row in tables[0][1:]:
        name = row[0].replace("**", "").strip()
        metrics: dict[str, float] = {}
        for i, col in enumerate(header[1:], start=1):
            col_clean = col.replace("**", "").strip()
            val = _clean_num(row[i]) if i < len(row) else None
            if val is not None:
                metrics[col_clean] = val
        truth[name] = metrics
    return truth


# ── checkers ──────────────────────────────────────────────────────────

MISMATCHES: list[str] = []


def _check_value(label: str, expected: float, actual: float, tol: float = 0.5) -> None:
    if abs(expected - actual) > tol:
        msg = f"  MISMATCH  {label}: expected {expected}, found {actual}"
        MISMATCHES.append(msg)
        print(msg)


def check_readme(truth: dict[str, dict[str, float]]) -> None:
    """Compare README.md metric table against phase3 truth."""
    readme = ROOT / "README.md"
    if not readme.exists():
        print("  WARN: README.md not found")
        return

    text = readme.read_text(encoding="utf-8")
    tables = _extract_md_tables(text)

    for model_name, expected in truth.items():
        found = False
        for table in tables:
            row = _find_model_row(table, model_name)
            if row is None:
                continue
            found = True
            header = tables[tables.index(table)][0]
            for i, col in enumerate(header[1:], start=1):
                col_clean = col.replace("**", "").strip()
                if col_clean in expected and i < len(row):
                    val = _clean_num(row[i])
                    if val is not None:
                        _check_value(
                            f"README / {model_name} / {col_clean}",
                            expected[col_clean], val,
                        )
            break
        if not found:
            print(f"  WARN: {model_name} not found in README tables")


def check_phase7(truth: dict[str, dict[str, float]]) -> None:
    """Check phase7_results.md references against phase3 truth."""
    path = ROOT / "docs" / "phase7_results.md"
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    tables = _extract_md_tables(text)
    lgbm = truth.get("LightGBM", {})
    if not lgbm:
        return

    for table in tables:
        row = _find_model_row(table, "with desc")
        if row is None:
            continue
        header = table[0]
        for i, col in enumerate(header[1:], start=1):
            col_clean = col.replace("**", "").strip()
            if col_clean in lgbm and i < len(row):
                val = _clean_num(row[i])
                if val is not None:
                    _check_value(
                        f"phase7 'with desc_*' / {col_clean}",
                        lgbm[col_clean], val,
                    )
        break


def check_inline_numbers() -> None:
    """Scan prose in README and docs for specific dollar/percentage references
    that should match phase3 truth."""
    phase3 = ROOT / "docs" / "phase3_results.md"
    if not phase3.exists():
        return
    truth = load_phase3_truth(phase3)
    lgbm = truth.get("LightGBM", {})
    if not lgbm:
        return

    rmse = lgbm.get("RMSE ($)")
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


def check_phase6_snapshot_note() -> None:
    """Verify phase6_results.md snapshot note references the current RMSE."""
    path = ROOT / "docs" / "phase6_results.md"
    if not path.exists():
        return
    phase3 = ROOT / "docs" / "phase3_results.md"
    truth = load_phase3_truth(phase3)
    lgbm = truth.get("LightGBM", {})
    rmse = lgbm.get("RMSE ($)")
    if rmse is None:
        return

    text = path.read_text(encoding="utf-8")
    # look for "test RMSE is $X,XXX (post-7B)"
    m = re.search(r"test RMSE is \$(\d{1,3}(?:,\d{3})*)", text)
    if m:
        val = float(m.group(1).replace(",", ""))
        _check_value("phase6 snapshot note RMSE", rmse, val, tol=1.0)


def check_ablation_a1() -> None:
    """Cross-check A1 log target row matches the main LightGBM row."""
    phase3 = ROOT / "docs" / "phase3_results.md"
    if not phase3.exists():
        return
    text = phase3.read_text(encoding="utf-8")
    tables = _extract_md_tables(text)

    main_truth = load_phase3_truth(phase3)
    lgbm = main_truth.get("LightGBM", {})
    if not lgbm:
        return

    for table in tables:
        row = _find_model_row(table, "log1p")
        if row is None:
            continue
        header = table[0]
        for i, col in enumerate(header[1:], start=1):
            col_clean = col.replace("**", "").strip()
            if col_clean in lgbm and i < len(row):
                val = _clean_num(row[i])
                if val is not None:
                    _check_value(
                        f"A1 log1p row / {col_clean} vs main LightGBM",
                        lgbm[col_clean], val,
                    )
        break


def check_a3_target_row() -> None:
    """Cross-check A3 target_encoding row matches the main LightGBM row."""
    phase3 = ROOT / "docs" / "phase3_results.md"
    if not phase3.exists():
        return
    text = phase3.read_text(encoding="utf-8")
    tables = _extract_md_tables(text)

    main_truth = load_phase3_truth(phase3)
    lgbm = main_truth.get("LightGBM", {})
    if not lgbm:
        return

    for table in tables:
        row = _find_model_row(table, "target_encoding")
        if row is None:
            continue
        header = table[0]
        for i, col in enumerate(header[1:], start=1):
            col_clean = col.replace("**", "").strip()
            if col_clean in lgbm and i < len(row):
                val = _clean_num(row[i])
                if val is not None:
                    _check_value(
                        f"A3 target_encoding / {col_clean} vs main LightGBM",
                        lgbm[col_clean], val,
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

    truth = load_phase3_truth(phase3)
    if not truth:
        print("ERROR: could not parse model comparison table.")
        sys.exit(1)

    print("Parsed truth:")
    for name, metrics in truth.items():
        vals = ", ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"  {name}: {vals}")
    print()

    print("[1] README.md vs phase3_results.md")
    check_readme(truth)

    print("[2] phase7_results.md 'with desc_*' row vs phase3 LightGBM")
    check_phase7(truth)

    print("[3] Inline dollar amounts in README")
    check_inline_numbers()

    print("[4] phase6_results.md snapshot note")
    check_phase6_snapshot_note()

    print("[5] A1 ablation log1p row == main LightGBM")
    check_ablation_a1()

    print("[6] A3 ablation target_encoding row == main LightGBM")
    check_a3_target_row()

    print()
    if MISMATCHES:
        print(f"FAILED: {len(MISMATCHES)} mismatch(es) found.")
        sys.exit(1)
    else:
        print("PASSED: all cross-checked numbers are consistent.")
        sys.exit(0)


if __name__ == "__main__":
    main()
