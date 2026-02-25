#!/usr/bin/env python3
"""Mypy baseline enforcement: block new type errors, celebrate fixes.

Usage:
    python scripts/mypy_baseline_check.py          # Check against baseline
    python scripts/mypy_baseline_check.py --update  # Regenerate baseline after fixes

Exit codes:
    0 - No new errors (baseline matches or errors were fixed)
    1 - New type errors introduced
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = REPO_ROOT / ".mypy_baseline.txt"
MYPY_CONFIG = REPO_ROOT / "mypy.ini"


def run_mypy() -> list[str]:
    """Run mypy and return sorted error lines."""
    result = subprocess.run(
        ["mypy", "packages/", "--config-file", str(MYPY_CONFIG)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    # Extract only error lines starting with "packages/"
    errors = sorted(line for line in result.stdout.splitlines() if line.startswith("packages/"))
    return errors


def load_baseline() -> list[str]:
    """Load the baseline file."""
    if not BASELINE_FILE.exists():
        print(f"ERROR: Baseline file not found: {BASELINE_FILE}")
        print("Run: python scripts/mypy_baseline_check.py --update")
        sys.exit(1)
    return sorted(BASELINE_FILE.read_text().strip().splitlines())


def update_baseline(errors: list[str]) -> None:
    """Write current errors as the new baseline."""
    BASELINE_FILE.write_text("\n".join(errors) + "\n" if errors else "")
    print(f"Baseline updated: {len(errors)} errors written to {BASELINE_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mypy baseline enforcement")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Regenerate baseline from current mypy output",
    )
    args = parser.parse_args()

    print("Running mypy...")
    current_errors = run_mypy()

    if args.update:
        update_baseline(current_errors)
        return

    baseline_errors = load_baseline()

    # Compare sets
    baseline_set = set(baseline_errors)
    current_set = set(current_errors)

    new_errors = current_set - baseline_set
    fixed_errors = baseline_set - current_set

    # Report
    if fixed_errors:
        print(f"\n  {len(fixed_errors)} error(s) FIXED (nice work!):")
        for err in sorted(fixed_errors):
            print(f"    - {err}")
        print("\n  Run 'python scripts/mypy_baseline_check.py --update' to shrink the baseline.")

    if new_errors:
        print(f"\n  FAIL: {len(new_errors)} NEW type error(s) introduced:")
        for err in sorted(new_errors):
            print(f"    + {err}")
        print(f"\n  Baseline: {len(baseline_errors)} | Current: {len(current_errors)}")
        sys.exit(1)

    print(f"\n  OK: {len(current_errors)} known errors, 0 new errors.")


if __name__ == "__main__":
    main()
