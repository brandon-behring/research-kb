#!/usr/bin/env python3
"""Import reviewed domain CSV back into R2 catalog JSONs.

Reads the edited domain_review.csv (after Brandon's offline review),
updates catalog_books_r2.json + catalog_other_r2.json with corrected domains,
and regenerates catalog_summary_r2.json.

Usage:
    python scripts/catalog_library/import_reviewed_csv.py
    python scripts/catalog_library/import_reviewed_csv.py --dry-run  # Preview changes
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Canonical 33-domain taxonomy (for validation)
CANONICAL_DOMAINS = {
    "algebra", "analysis", "topology_geometry", "linear_algebra",
    "probability_theory", "mathematics",
    "dynamical_systems", "numerical_methods", "optimization",
    "statistics", "causal_inference", "econometrics", "time_series",
    "algorithms", "software_engineering", "functional_programming", "sql",
    "machine_learning", "deep_learning", "reinforcement_learning",
    "rag_llm", "ml_engineering", "data_science", "recommender_systems",
    "physics", "biology_neuroscience", "signal_processing",
    "finance", "portfolio_management", "actuarial_insurance", "adtech",
    "interview_prep", "fitness",
}


def load_csv(csv_path: Path) -> dict[str, dict[str, str]]:
    """Load reviewed CSV, indexed by sha256."""
    rows: dict[str, dict[str, str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sha = row["sha256"]
            rows[sha] = row
    return rows


def apply_updates(
    entries: list[dict[str, Any]],
    csv_rows: dict[str, dict[str, str]],
    source_name: str,
    dry_run: bool = False,
) -> tuple[int, int, int, list[str]]:
    """Apply CSV corrections to catalog entries.

    Returns: (changed, skipped, unchanged, warnings)
    """
    changed = 0
    skipped = 0
    unchanged = 0
    warnings: list[str] = []

    for entry in entries:
        sha = entry["sha256"]
        row = csv_rows.get(sha)
        if not row:
            continue

        suggested = row.get("suggested_domain", "").strip()
        skip_flag = row.get("skip", "FALSE").strip().upper()
        secondary_str = row.get("secondary_domains", "").strip()

        # Handle skip
        if skip_flag == "TRUE":
            if entry.get("domain") != "skip":
                entry["domain"] = "skip"
                entry["confidence"] = "manual_skip"
                entry.setdefault("r2_classification", {})["method"] = "manual_review"
                changed += 1
                skipped += 1
            else:
                unchanged += 1
            continue

        # Validate domain
        if not suggested:
            warnings.append(f"Empty suggested_domain for {entry.get('filename', sha)}")
            unchanged += 1
            continue

        if suggested not in CANONICAL_DOMAINS:
            warnings.append(
                f"Unknown domain '{suggested}' for {entry.get('filename', sha)}"
            )
            # Still apply it, but warn

        # Parse secondary domains
        secondaries = [
            d.strip() for d in secondary_str.split(",") if d.strip()
        ] if secondary_str else []

        # Validate secondaries
        for sd in secondaries:
            if sd not in CANONICAL_DOMAINS:
                warnings.append(
                    f"Unknown secondary domain '{sd}' for {entry.get('filename', sha)}"
                )

        # Check if anything changed
        old_domain = entry.get("domain", "unclassified")
        old_secondaries = entry.get("secondary_domains", [])

        if suggested == old_domain and secondaries == old_secondaries:
            unchanged += 1
            continue

        # Apply changes
        if not dry_run:
            entry["domain"] = suggested
            entry["confidence"] = "medium"  # Human-confirmed
            entry["secondary_domains"] = secondaries
            entry.setdefault("r2_classification", {})
            entry["r2_classification"]["method"] = "manual_review"
            entry["r2_classification"]["domain"] = suggested

            # Preserve LLM classification metadata if present
            llm_domain = row.get("llm_domain", "").strip()
            if llm_domain:
                entry["r2_classification"]["llm_domain"] = llm_domain
                entry["r2_classification"]["llm_confidence"] = row.get("llm_confidence", "")

            # Preserve already_ingested flag
            if row.get("already_ingested", "").strip().upper() == "TRUE":
                entry["r2_classification"]["already_ingested"] = True

        changed += 1

    return changed, skipped, unchanged, warnings


def generate_summary(
    books: list[dict], other: list[dict], output_path: Path
) -> dict[str, Any]:
    """Regenerate catalog summary statistics."""
    all_entries = books + other
    domain_dist = Counter(e["domain"] for e in all_entries)
    confidence_dist = Counter(e.get("confidence", "unknown") for e in all_entries)

    # Count entries with secondary domains
    multi_domain = sum(1 for e in all_entries if e.get("secondary_domains"))

    summary = {
        "total_files": len(all_entries),
        "total_books": len(books),
        "total_other": len(other),
        "domain_distribution": dict(sorted(domain_dist.items(), key=lambda x: -x[1])),
        "confidence_distribution": dict(sorted(confidence_dist.items(), key=lambda x: -x[1])),
        "multi_domain_entries": multi_domain,
        "unique_domains": len([d for d in domain_dist if d != "skip"]),
        "skipped_entries": domain_dist.get("skip", 0),
    }

    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Import reviewed domain CSV into catalog")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to reviewed CSV (default: fixtures/library_catalog/domain_review.csv)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    catalog_dir = base / "fixtures" / "library_catalog"
    csv_path = args.csv or catalog_dir / "domain_review.csv"

    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    print("Loading data...")
    csv_rows = load_csv(csv_path)
    books = json.loads((catalog_dir / "catalog_books_r2.json").read_text())
    other = json.loads((catalog_dir / "catalog_other_r2.json").read_text())

    print(f"  CSV rows: {len(csv_rows)}")
    print(f"  Books: {len(books)}, Other: {len(other)}")

    mode = "DRY RUN" if args.dry_run else "APPLYING"
    print(f"\n--- {mode} ---\n")

    # Apply updates
    b_changed, b_skipped, b_unchanged, b_warnings = apply_updates(
        books, csv_rows, "books", args.dry_run
    )
    o_changed, o_skipped, o_unchanged, o_warnings = apply_updates(
        other, csv_rows, "other", args.dry_run
    )

    total_changed = b_changed + o_changed
    total_skipped = b_skipped + o_skipped
    total_unchanged = b_unchanged + o_unchanged
    all_warnings = b_warnings + o_warnings

    print(f"  Changed: {total_changed}")
    print(f"  Skipped (marked skip=TRUE): {total_skipped}")
    print(f"  Unchanged: {total_unchanged}")

    if all_warnings:
        print(f"\n  Warnings ({len(all_warnings)}):")
        for w in all_warnings[:20]:
            print(f"    - {w}")
        if len(all_warnings) > 20:
            print(f"    ... and {len(all_warnings) - 20} more")

    # Show domain distribution diff
    old_dist = Counter(
        e["domain"]
        for e in json.loads((catalog_dir / "catalog_books_r2.json").read_text())
        + json.loads((catalog_dir / "catalog_other_r2.json").read_text())
    )
    if not args.dry_run:
        new_dist = Counter(e["domain"] for e in books + other)
    else:
        # Simulate
        new_dist = Counter(
            csv_rows[e["sha256"]].get("suggested_domain", e["domain"])
            if e["sha256"] in csv_rows
            and csv_rows[e["sha256"]].get("skip", "FALSE").upper() != "TRUE"
            else ("skip" if e["sha256"] in csv_rows and csv_rows[e["sha256"]].get("skip", "FALSE").upper() == "TRUE" else e["domain"])
            for e in books + other
        )

    print("\n  Domain distribution changes:")
    all_domains = sorted(set(old_dist) | set(new_dist))
    for d in all_domains:
        old_c = old_dist.get(d, 0)
        new_c = new_dist.get(d, 0)
        if old_c != new_c:
            delta = new_c - old_c
            sign = "+" if delta > 0 else ""
            print(f"    {d}: {old_c} → {new_c} ({sign}{delta})")

    # Write if not dry-run
    if not args.dry_run:
        print("\nWriting updated catalogs...")
        (catalog_dir / "catalog_books_r2.json").write_text(
            json.dumps(books, indent=2, ensure_ascii=False) + "\n"
        )
        (catalog_dir / "catalog_other_r2.json").write_text(
            json.dumps(other, indent=2, ensure_ascii=False) + "\n"
        )

        summary = generate_summary(books, other, catalog_dir / "catalog_summary_r2.json")
        print(f"  Summary: {summary['unique_domains']} domains, {summary['skipped_entries']} skipped")
        print("\nDone. Catalogs updated.")
    else:
        print("\nDry run complete. No files modified.")


if __name__ == "__main__":
    main()
