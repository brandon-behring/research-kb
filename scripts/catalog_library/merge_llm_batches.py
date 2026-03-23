#!/usr/bin/env python3
"""Merge LLM classification batch outputs into a single checkpoint file.

Reads .llm_output_batch_*.json files, validates entries, and writes
the merged result to .llm_classification_checkpoint.json.

Usage:
    python scripts/catalog_library/merge_llm_batches.py
    python scripts/catalog_library/merge_llm_batches.py --check  # Check completion status only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

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
    "skip",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LLM batch outputs")
    parser.add_argument("--check", action="store_true", help="Check status only")
    args = parser.parse_args()

    catalog_dir = Path(__file__).resolve().parents[2] / "fixtures" / "library_catalog"

    # Find all batch output files
    batch_files = sorted(catalog_dir.glob(".llm_output_batch_*.json"))
    expected = sorted(catalog_dir.glob(".llm_input_batch_*.json"))

    print(f"Found {len(batch_files)}/{len(expected)} batch output files")

    if args.check:
        # Show which batches are missing
        done_ids = {int(f.stem.split("_")[-1]) for f in batch_files}
        expected_ids = {int(f.stem.split("_")[-1]) for f in expected}
        missing = sorted(expected_ids - done_ids)
        if missing:
            print(f"Missing batches: {missing}")
        else:
            print("All batches complete!")
        return

    if not batch_files:
        print("No batch output files found.", file=sys.stderr)
        sys.exit(1)

    # Merge all batches
    merged: dict[str, dict] = {}
    warnings: list[str] = []
    domain_counts: Counter[str] = Counter()

    for bf in batch_files:
        try:
            data = json.loads(bf.read_text())
        except json.JSONDecodeError as e:
            warnings.append(f"Invalid JSON in {bf.name}: {e}")
            continue

        if not isinstance(data, dict):
            warnings.append(f"{bf.name}: expected dict, got {type(data).__name__}")
            continue

        batch_count = 0
        for sha, classification in data.items():
            if not isinstance(classification, dict):
                warnings.append(f"{bf.name}: invalid entry for {sha[:12]}...")
                continue

            domain = classification.get("domain", "")
            confidence = classification.get("confidence", 0)
            reasoning = classification.get("reasoning", "")

            # Validate domain
            if domain not in CANONICAL_DOMAINS:
                warnings.append(f"{bf.name}: unknown domain '{domain}' for {sha[:12]}...")
                # Try to fix common issues
                domain_lower = domain.lower().strip()
                if domain_lower in CANONICAL_DOMAINS:
                    domain = domain_lower
                else:
                    continue

            # Normalize confidence to float
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = 0.5

            merged[sha] = {
                "domain": domain,
                "confidence": round(confidence, 2),
                "reasoning": str(reasoning)[:200],
            }
            domain_counts[domain] += 1
            batch_count += 1

        print(f"  {bf.name}: {batch_count} entries")

    # Write checkpoint
    checkpoint_path = catalog_dir / ".llm_classification_checkpoint.json"
    checkpoint_path.write_text(json.dumps(merged, indent=2) + "\n")

    print(f"\nMerged: {len(merged)} classifications → {checkpoint_path.name}")

    # Stats
    print(f"\nDomain distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")

    skip_count = domain_counts.get("skip", 0)
    high_conf = sum(1 for v in merged.values() if v["confidence"] >= 0.85)
    med_conf = sum(1 for v in merged.values() if 0.70 <= v["confidence"] < 0.85)
    low_conf = sum(1 for v in merged.values() if v["confidence"] < 0.70)

    print(f"\nConfidence tiers:")
    print(f"  High (≥0.85): {high_conf}")
    print(f"  Medium (0.70-0.85): {med_conf}")
    print(f"  Low (<0.70): {low_conf}")
    print(f"  Skip: {skip_count}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  - {w}")

    print("\nDone.")


if __name__ == "__main__":
    main()
