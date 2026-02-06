#!/usr/bin/env python3
"""Pre-populate method_assumption_cache for top interview methods.

Uses Anthropic Haiku (or Ollama) to extract structured assumptions,
then caches them for instant retrieval by MCP tools and CLI.

Usage:
    python scripts/populate_assumption_cache.py
    python scripts/populate_assumption_cache.py --backend ollama
    python scripts/populate_assumption_cache.py --dry-run
    python scripts/populate_assumption_cache.py --method "double machine learning"
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_storage import get_connection_pool
from research_kb_storage.assumption_audit import (
    MIN_ASSUMPTIONS_THRESHOLD,
    MethodAssumptionAuditor,
)

# Top 10 causal inference methods (matches eval_interview_readiness.py)
TOP_METHODS = [
    "instrumental variables",
    "double machine learning",
    "difference-in-differences",
    "propensity score matching",
    "inverse probability weighting",
    "doubly robust estimation",
    "regression discontinuity design",
    "causal forests",
    "CUPED",
    "LATE",
]


async def populate_method(
    method_name: str,
    backend: str = "anthropic",
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Populate cache for a single method.

    Args:
        method_name: Method name to look up and populate
        backend: "anthropic" or "ollama"
        dry_run: If True, show what would happen without caching
        force: If True, re-populate even if cache already has enough assumptions

    Returns:
        Dict with method, status, assumption_count, etc.
    """
    result = {
        "method": method_name,
        "status": "unknown",
        "graph_count": 0,
        "cache_count": 0,
        "extracted_count": 0,
        "cached_count": 0,
    }

    # Step 1: Find the method concept
    method = await MethodAssumptionAuditor.find_method(method_name)
    if method is None:
        result["status"] = "not_found"
        return result

    result["method_id"] = str(method.id)
    result["canonical_name"] = method.canonical_name or method.name

    # Step 2: Check existing graph assumptions
    graph_assumptions = await MethodAssumptionAuditor.get_assumptions_from_graph(method.id)
    result["graph_count"] = len(graph_assumptions)

    # Step 3: Check existing cache
    cached_assumptions = await MethodAssumptionAuditor.get_cached_assumptions(method.id)
    result["cache_count"] = len(cached_assumptions)

    total_existing = len(graph_assumptions) + len(cached_assumptions)

    # Step 4: Skip if already populated (unless force)
    if not force and total_existing >= MIN_ASSUMPTIONS_THRESHOLD:
        result["status"] = "already_populated"
        return result

    if dry_run:
        result["status"] = "dry_run"
        return result

    # Step 5: Extract via chosen backend
    if backend == "anthropic":
        extracted = await MethodAssumptionAuditor.extract_assumptions_with_anthropic(
            method_name=method.canonical_name or method.name,
            definition=method.definition,
        )
        extraction_method = "anthropic:claude-haiku-4-5-20251001"
    else:
        extracted = await MethodAssumptionAuditor.extract_assumptions_with_ollama(
            method_name=method.canonical_name or method.name,
            definition=method.definition,
        )
        extraction_method = "ollama:llama3.1:8b"

    result["extracted_count"] = len(extracted)

    if not extracted:
        result["status"] = "extraction_failed"
        return result

    # Step 6: Cache the results
    cached_count = await MethodAssumptionAuditor.cache_assumptions(
        method_id=method.id,
        assumptions=extracted,
        extraction_method=extraction_method,
    )
    result["cached_count"] = cached_count
    result["status"] = "populated"
    result["assumptions"] = [
        {"name": a.name, "importance": a.importance} for a in extracted
    ]

    return result


async def main(args: argparse.Namespace) -> int:
    """Run population for all methods or a single method.

    Returns:
        Exit code: 0 = success, 1 = partial failure, 2 = total failure
    """
    start_time = time.monotonic()

    # Initialize connection pool
    await get_connection_pool()

    methods = [args.method] if args.method else TOP_METHODS

    print(f"Populating assumption cache for {len(methods)} method(s)")
    print(f"Backend: {args.backend}")
    print(f"Dry run: {args.dry_run}")
    print(f"Force: {args.force}")
    print("-" * 60)

    results = []
    populated = 0
    skipped = 0
    failed = 0

    for i, method_name in enumerate(methods, 1):
        print(f"\n[{i}/{len(methods)}] {method_name}...", end=" ", flush=True)

        result = await populate_method(
            method_name,
            backend=args.backend,
            dry_run=args.dry_run,
            force=args.force,
        )

        status = result["status"]

        if status == "populated":
            populated += 1
            assumptions = result.get("assumptions", [])
            names = [a["name"] for a in assumptions]
            print(f"OK ({result['cached_count']} assumptions cached)")
            for a in assumptions:
                imp = a["importance"]
                marker = "[CRITICAL]" if imp == "critical" else f"[{imp}]"
                print(f"    {marker} {a['name']}")
        elif status == "already_populated":
            skipped += 1
            print(f"SKIP (already has {result['graph_count']}g + {result['cache_count']}c)")
        elif status == "not_found":
            failed += 1
            print("NOT FOUND in knowledge base")
        elif status == "extraction_failed":
            failed += 1
            print(f"EXTRACTION FAILED ({args.backend})")
        elif status == "dry_run":
            print(f"DRY RUN (would extract, has {result['graph_count']}g + {result['cache_count']}c)")
        else:
            failed += 1
            print(f"UNKNOWN STATUS: {status}")

        results.append(result)

    elapsed = time.monotonic() - start_time

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY ({elapsed:.1f}s)")
    print(f"  Populated: {populated}/{len(methods)}")
    print(f"  Skipped (already cached): {skipped}/{len(methods)}")
    print(f"  Failed: {failed}/{len(methods)}")
    print("=" * 60)

    # JSON output if requested
    if args.json:
        output = {
            "backend": args.backend,
            "dry_run": args.dry_run,
            "elapsed_seconds": round(elapsed, 2),
            "populated": populated,
            "skipped": skipped,
            "failed": failed,
            "results": results,
        }
        print(json.dumps(output, indent=2, default=str))

    if failed == len(methods):
        return 2
    elif failed > 0:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-populate method_assumption_cache for top interview methods.",
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "ollama"],
        default="anthropic",
        help="LLM backend for assumption extraction (default: anthropic)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Single method to populate (default: all TOP_METHODS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without actually populating",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-populate even if cache already has enough assumptions",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
