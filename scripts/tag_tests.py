#!/usr/bin/env python3
"""Automated test marker tagging script.

Analyzes test files and suggests/applies appropriate pytest markers
based on directory structure, file content, and naming patterns.

Usage:
    python scripts/tag_tests.py              # Report only
    python scripts/tag_tests.py --apply      # Apply changes
    python scripts/tag_tests.py --verbose    # Detailed output
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Configuration
# ============================================================================

# Markers from pytest.ini
VALID_MARKERS = {
    "unit",
    "integration",
    "e2e",
    "smoke",
    "quality",
    "scripts",
    "slow",
    "requires_ollama",
    "requires_grobid",
    "requires_embedding",
    "requires_reranker",
}

# Directory-based rules (highest precedence)
DIRECTORY_RULES: dict[str, str] = {
    "tests/e2e/": "e2e",
    "tests/integration/": "integration",
    "tests/smoke/": "smoke",
    "tests/quality/": "quality",
    "tests/scripts/": "scripts",
}

# Filename pattern rules
FILENAME_RULES: dict[str, str] = {
    r"test_.*_integration\.py$": "integration",
    r"test_.*_e2e\.py$": "e2e",
    r"test_.*_smoke\.py$": "smoke",
}

# Content patterns for service requirements
# NOTE: Patterns must be specific enough to avoid matching string literals,
# config field names, error messages, and docstrings.
SERVICE_PATTERNS: dict[str, list[str]] = {
    "requires_ollama": [
        r"OllamaClient\(",  # actual instantiation
        r"ollama\.generate",  # API call
    ],
    "requires_embedding": [
        r"EmbeddingClient\(",  # actual instantiation
        r"from sentence_transformers import",  # actual model loading
    ],
    "requires_grobid": [
        r"grobid_client",  # module reference
        r"GrobidClient\(",  # actual instantiation
    ],
    "requires_reranker": [
        r"RerankerClient\(",  # actual instantiation
        r"BAAI/bge-reranker",  # model name
    ],
}

# Patterns that suggest unit tests (extensive mocking)
UNIT_PATTERNS = [
    r"@patch\(",
    r"with patch\(",
    r"MagicMock\(\)",
    r"AsyncMock\(\)",
    r"mock_",
    r"Mock\(",
]

# Patterns that suggest integration tests (actual DB usage, not string literals)
INTEGRATION_PATTERNS = [
    r"db_pool",
    r"asyncpg\.create_pool",
    r"async with pool",
]


@dataclass
class TaggingResult:
    """Result of analyzing a test file."""

    file_path: Path
    current_markers: set[str]
    suggested_markers: set[str]
    reason: str
    needs_update: bool


# ============================================================================
# Analysis Functions
# ============================================================================


def get_file_content(path: Path) -> str:
    """Read file content."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def extract_current_markers(content: str) -> set[str]:
    """Extract existing pytest markers from file."""
    markers = set()

    # Class-level markers: @pytest.mark.X
    class_markers = re.findall(r"@pytest\.mark\.(\w+)", content)
    markers.update(m for m in class_markers if m in VALID_MARKERS)

    # pytestmark = pytest.mark.X
    pytestmark_single = re.findall(r"pytestmark\s*=\s*pytest\.mark\.(\w+)", content)
    markers.update(m for m in pytestmark_single if m in VALID_MARKERS)

    # pytestmark = [pytest.mark.X, ...]
    pytestmark_list = re.findall(r"pytestmark\s*=\s*\[(.*?)\]", content, re.DOTALL)
    for match in pytestmark_list:
        list_markers = re.findall(r"pytest\.mark\.(\w+)", match)
        markers.update(m for m in list_markers if m in VALID_MARKERS)

    return markers


def check_directory_rules(path: Path) -> Optional[str]:
    """Check if file matches directory-based rules."""
    path_str = str(path)
    for pattern, marker in DIRECTORY_RULES.items():
        if pattern in path_str:
            return marker
    return None


def check_filename_rules(path: Path) -> Optional[str]:
    """Check if filename matches naming patterns."""
    filename = path.name
    for pattern, marker in FILENAME_RULES.items():
        if re.match(pattern, filename):
            return marker
    return None


def check_content_patterns(content: str) -> set[str]:
    """Analyze content for service requirements."""
    markers = set()

    for marker, patterns in SERVICE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                markers.add(marker)
                break

    return markers


def is_likely_unit_test(content: str) -> bool:
    """Check if file appears to be unit tests (extensive mocking)."""
    mock_count = sum(len(re.findall(pattern, content)) for pattern in UNIT_PATTERNS)
    return mock_count >= 3


def is_likely_integration_test(content: str) -> bool:
    """Check if file appears to be integration tests."""
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in INTEGRATION_PATTERNS)


def analyze_file(path: Path) -> TaggingResult:
    """Analyze a test file and suggest markers."""
    content = get_file_content(path)
    current = extract_current_markers(content)
    suggested = set()
    reasons = []

    # 1. Directory rules (highest precedence)
    dir_marker = check_directory_rules(path)
    if dir_marker:
        suggested.add(dir_marker)
        reasons.append(f"directory rule: {dir_marker}")

    # 2. Filename rules
    file_marker = check_filename_rules(path)
    if file_marker and file_marker not in suggested:
        suggested.add(file_marker)
        reasons.append(f"filename pattern: {file_marker}")

    # 3. Service requirements (skip for unit tests that mock the service)
    service_markers = check_content_patterns(content)
    if service_markers and is_likely_unit_test(content):
        # Heavy mocking means tests don't actually need the external service
        service_markers = set()
        reasons.append("service patterns found but mocked (skipped requires_*)")
    suggested.update(service_markers)
    if service_markers:
        reasons.append(f"service requirements: {', '.join(sorted(service_markers))}")

    # 4. Default to unit if in packages/*/tests/ and no other marker
    is_package_test = "packages/" in str(path) and "/tests/" in str(path)

    if is_package_test and not (
        suggested
        - {
            "requires_ollama",
            "requires_embedding",
            "requires_grobid",
            "requires_reranker",
        }
    ):
        # No primary marker assigned yet
        if is_likely_integration_test(content):
            suggested.add("integration")
            reasons.append("content analysis: database patterns")
        elif is_likely_unit_test(content) or not current:
            suggested.add("unit")
            reasons.append("default: package test with mocking")

    # 5. Tests in tests/ directory (repo root)
    if str(path).startswith("tests/"):
        if not (
            suggested
            - {
                "requires_ollama",
                "requires_embedding",
                "requires_grobid",
                "requires_reranker",
            }
        ):
            # Check subdirectory
            if "e2e" in str(path):
                suggested.add("e2e")
                reasons.append("path contains e2e")
            elif "integration" in str(path):
                suggested.add("integration")
                reasons.append("path contains integration")
            else:
                suggested.add("integration")
                reasons.append("default: repo-level test")

    # Determine if update needed
    primary_markers = {"unit", "integration", "e2e", "smoke", "quality", "scripts"}
    has_primary_current = bool(current & primary_markers)
    has_primary_suggested = bool(suggested & primary_markers)

    needs_update = (has_primary_suggested and not has_primary_current) or (
        service_markers and not service_markers.issubset(current)
    )

    return TaggingResult(
        file_path=path,
        current_markers=current,
        suggested_markers=suggested,
        reason="; ".join(reasons) if reasons else "no changes",
        needs_update=needs_update,
    )


# ============================================================================
# File Modification
# ============================================================================


def add_markers_to_file(path: Path, markers: set[str]) -> bool:
    """Add markers to a test file.

    Adds a pytestmark line at the top of the file (after imports).
    """
    content = get_file_content(path)
    if not content:
        return False

    # Check if pytestmark already exists
    if "pytestmark" in content:
        # Would need to merge - skip for now
        return False

    # Find insertion point (after module docstring and imports)
    lines = content.split("\n")
    insert_line = 0

    # Track state
    in_docstring = False
    in_multiline_import = False
    docstring_type = None
    has_pytest_import = "import pytest" in content

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track docstrings
        if not in_docstring and not in_multiline_import:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_type = stripped[:3]
                if stripped.count(docstring_type) >= 2 and len(stripped) > 3:
                    # Single-line docstring
                    insert_line = i + 1
                else:
                    in_docstring = True
        elif in_docstring:
            if docstring_type in stripped:
                in_docstring = False
                insert_line = i + 1
            continue

        # Track multi-line imports (parenthesized)
        if in_multiline_import:
            if ")" in stripped:
                in_multiline_import = False
                insert_line = i + 1
            continue

        # Track imports
        if not in_docstring and (stripped.startswith("import ") or stripped.startswith("from ")):
            if "(" in stripped and ")" not in stripped:
                # Start of multi-line import
                in_multiline_import = True
            else:
                insert_line = i + 1

        # Stop at first class/function/decorator
        if not in_docstring and (
            stripped.startswith("class ") or stripped.startswith("def ") or stripped.startswith("@")
        ):
            break

    # Build marker line
    sorted_markers = sorted(markers)
    if len(sorted_markers) == 1:
        marker_line = f"pytestmark = pytest.mark.{sorted_markers[0]}"
    else:
        marker_items = ", ".join(f"pytest.mark.{m}" for m in sorted_markers)
        marker_line = f"pytestmark = [{marker_items}]"

    # Ensure pytest is imported
    if not has_pytest_import:
        lines.insert(insert_line, "import pytest")
        insert_line += 1

    # Insert marker with blank line separation
    lines.insert(insert_line, "")
    lines.insert(insert_line + 1, marker_line)
    lines.insert(insert_line + 2, "")

    # Write back
    new_content = "\n".join(lines)
    path.write_text(new_content, encoding="utf-8")
    return True


# ============================================================================
# Main
# ============================================================================


def find_test_files(root: Path) -> list[Path]:
    """Find all test files."""
    test_files = []

    # packages/*/tests/
    for pkg_tests in root.glob("packages/*/tests/test_*.py"):
        test_files.append(pkg_tests)

    # tests/
    for repo_tests in root.glob("tests/**/test_*.py"):
        test_files.append(repo_tests)

    return sorted(set(test_files))


def main():
    parser = argparse.ArgumentParser(description="Tag tests with pytest markers")
    parser.add_argument("--apply", action="store_true", help="Apply changes to files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Find repo root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    print(f"Scanning test files in {repo_root}...")
    test_files = find_test_files(repo_root)
    print(f"Found {len(test_files)} test files\n")

    # Analyze all files
    results: list[TaggingResult] = []
    for path in test_files:
        rel_path = path.relative_to(repo_root)
        result = analyze_file(path)
        result.file_path = rel_path  # Use relative path for display
        results.append(result)

    # Categorize
    already_marked = [r for r in results if not r.needs_update]
    needs_tagging = [r for r in results if r.needs_update]

    # Report
    print("=" * 60)
    print("TEST MARKER TAGGING REPORT")
    print("=" * 60)
    print(f"\nFiles analyzed:       {len(results)}")
    print(f"Already marked:       {len(already_marked)}")
    print(f"Needs tagging:        {len(needs_tagging)}")

    # Summary by marker
    print("\n--- Suggested Markers Summary ---")
    marker_counts: dict[str, int] = {}
    for r in needs_tagging:
        for m in r.suggested_markers:
            marker_counts[m] = marker_counts.get(m, 0) + 1

    for marker, count in sorted(marker_counts.items(), key=lambda x: -x[1]):
        print(f"  {marker}: {count} files")

    # Verbose output
    if args.verbose:
        print("\n--- Files Needing Tags ---")
        for r in needs_tagging:
            print(f"\n  {r.file_path}")
            print(f"    Current:   {r.current_markers or 'none'}")
            print(f"    Suggested: {r.suggested_markers}")
            print(f"    Reason:    {r.reason}")

    # Apply changes
    if args.apply and needs_tagging:
        print("\n--- Applying Changes ---")
        applied = 0
        skipped = 0

        for r in needs_tagging:
            full_path = repo_root / r.file_path
            # Only add markers that aren't already present
            new_markers = r.suggested_markers - r.current_markers
            if new_markers:
                if add_markers_to_file(full_path, new_markers):
                    print(f"  [OK] {r.file_path}")
                    applied += 1
                else:
                    print(f"  [SKIP] {r.file_path} (has pytestmark)")
                    skipped += 1

        print(f"\nApplied: {applied}, Skipped: {skipped}")
    elif needs_tagging:
        print("\nRun with --apply to modify files")

    # Exit code
    return 0


if __name__ == "__main__":
    sys.exit(main())
