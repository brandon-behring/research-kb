#!/usr/bin/env python3
"""Mass ingest books from library catalog into research-kb.

Reads catalog_books_r2.json, filters by domain/tier/priority, and ingests
using Docling extraction. Supports checkpointing for resume after interruption.

Usage:
    # Tier 1: Top 10 per domain (~310 books)
    python scripts/mass_ingest_catalog.py --tier 1 --no-embed --quiet \
        --base-dir ~/Claude/research-kb/fixtures/library_books/

    # Single domain
    python scripts/mass_ingest_catalog.py --domain mathematics --no-embed --quiet \
        --base-dir ~/Claude/research-kb/fixtures/library_books/

    # Dry run to preview
    python scripts/mass_ingest_catalog.py --tier 1 --dry-run \
        --base-dir ~/Claude/research-kb/fixtures/library_books/

    # Retry previously failed books
    python scripts/mass_ingest_catalog.py --tier 1 --retry-failed --no-embed --quiet \
        --base-dir ~/Claude/research-kb/fixtures/library_books/
"""

import argparse
import asyncio
import gc
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Reuse ingest_textbook() from existing script (avoids duplication)
sys.path.insert(0, str(Path(__file__).parent))
from ingest_missing_textbooks import compute_file_hash, ingest_textbook

# Add packages to path (also done by ingest_missing_textbooks import,
# but explicit is better than implicit)
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from uuid import UUID

from research_kb_common import EmbeddingError, StorageError, configure_logging, get_logger
from research_kb_pdf.post_ingest import run_post_ingest_hooks
from research_kb_storage import DatabaseConfig, SourceStore, get_connection_pool

logger = get_logger(__name__)

DEFAULT_CATALOG = (
    Path(__file__).parent.parent / "fixtures" / "library_catalog" / "catalog_books_r2.json"
)
CHECKPOINT_PATH = (
    Path(__file__).parent.parent / "fixtures" / "library_catalog" / ".mass_ingest_checkpoint.json"
)

# All 35 valid domains (original 23 + 12 from catalog expansion)
VALID_DOMAINS = {
    "actuarial_insurance",
    "adtech",
    "algebra",
    "algorithms",
    "analysis",
    "biology_neuroscience",
    "causal_inference",
    "data_science",
    "deep_learning",
    "dynamical_systems",
    "econometrics",
    "economics",
    "finance",
    "fitness",
    "forecasting",
    "functional_programming",
    "healthcare",
    "interview_prep",
    "linear_algebra",
    "machine_learning",
    "mathematics",
    "ml_engineering",
    "numerical_methods",
    "optimization",
    "physics",
    "portfolio_management",
    "probability_theory",
    "rag_llm",
    "recommender_systems",
    "reinforcement_learning",
    "signal_processing",
    "software_engineering",
    "sql",
    "statistics",
    "time_series",
    "topology_geometry",
}

BOOKS_PER_TIER = 10


# --- Checkpoint ---


def load_checkpoint() -> dict:
    """Load checkpoint file for resume support."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"completed": [], "failed": {}, "last_updated": None}


def save_checkpoint(checkpoint: dict) -> None:
    """Persist checkpoint atomically (write-then-rename)."""
    checkpoint["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2)
    tmp.rename(CHECKPOINT_PATH)


# --- Catalog loading and filtering ---


def load_catalog(catalog_path: Path) -> list[dict]:
    """Load catalog entries: books only, domain != skip."""
    with open(catalog_path) as f:
        entries = json.load(f)
    return [e for e in entries if e.get("domain") != "skip" and e.get("is_book", False)]


def deduplicate_meap(entries: list[dict]) -> list[dict]:
    """Keep only the highest version of each MEAP title.

    Groups by base title (stripped of _vN_MEAP suffix), keeps the entry
    with the highest version number. Removes ~182 redundant versions.
    """
    meap_groups: dict[str, list[dict]] = defaultdict(list)
    non_meap = []

    for e in entries:
        name = e.get("title", e.get("filename", ""))
        if "_MEAP" in name or "MEAP" in e.get("filename", ""):
            base = re.sub(r"_v\d+_MEAP.*", "", name).strip()
            base = re.sub(r"\s*\(\d+\)$", "", base)
            meap_groups[base].append(e)
        else:
            non_meap.append(e)

    # Keep only highest version per group
    kept = []
    removed = 0
    for base, versions in meap_groups.items():

        def _version_key(entry: dict) -> int:
            name = entry.get("title", entry.get("filename", ""))
            m = re.search(r"_v(\d+)_MEAP", name)
            return int(m.group(1)) if m else 0

        versions.sort(key=_version_key, reverse=True)
        kept.append(versions[0])  # Keep highest version
        removed += len(versions) - 1

    if removed > 0:
        logger.info("meap_dedup", kept=len(kept), removed=removed)

    return non_meap + kept


def apply_filters(entries: list[dict], args: argparse.Namespace) -> list[dict]:
    """Apply CLI filters and return sorted entries.

    Sorting: priority_score DESC within domain, file_size ASC for ties.
    Tier slicing: T1=top 10/domain, T2=next 10, T3=rest.
    """
    # Deduplicate MEAP versions before other filters
    entries = deduplicate_meap(entries)

    if args.domain:
        entries = [e for e in entries if e["domain"] == args.domain]

    if args.min_priority is not None:
        entries = [e for e in entries if e.get("priority_score", 0) >= args.min_priority]

    if args.max_size_mb is not None:
        entries = [e for e in entries if e.get("file_size_mb", 0) <= args.max_size_mb]

    # Group by domain, sort within each group
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_domain[e["domain"]].append(e)

    for d in by_domain:
        by_domain[d].sort(key=lambda e: (-e.get("priority_score", 0), e.get("file_size_mb", 0)))

    # Slice per tier/max-per-domain
    result: list[dict] = []
    if args.tier == 1:
        for domain_entries in by_domain.values():
            result.extend(domain_entries[:BOOKS_PER_TIER])
    elif args.tier == 2:
        for domain_entries in by_domain.values():
            result.extend(domain_entries[BOOKS_PER_TIER : BOOKS_PER_TIER * 2])
    elif args.tier == 3:
        for domain_entries in by_domain.values():
            result.extend(domain_entries[BOOKS_PER_TIER * 2 :])
    elif args.max_per_domain:
        for domain_entries in by_domain.values():
            result.extend(domain_entries[: args.max_per_domain])
    else:
        for domain_entries in by_domain.values():
            result.extend(domain_entries)

    # Final ordering: smallest files first (fastest throughput), priority DESC as tiebreaker
    result.sort(key=lambda e: (e.get("file_size_mb", 0), -e.get("priority_score", 0)))
    return result


_CASE_INSENSITIVE_CACHE: dict[str, Path] | None = None


def _build_case_cache(base_dir: Path) -> dict[str, Path]:
    """Build lowercase filename → Path cache for case-insensitive lookup."""
    global _CASE_INSENSITIVE_CACHE
    if _CASE_INSENSITIVE_CACHE is None:
        _CASE_INSENSITIVE_CACHE = {}
        for f in base_dir.iterdir():
            if f.is_file():
                _CASE_INSENSITIVE_CACHE[f.name.lower()] = f
    return _CASE_INSENSITIVE_CACHE


def remap_path(full_path: str, base_dir: Path) -> Path:
    """Remap catalog path to local copy using filename only.

    Catalog paths: /media/.../books/SomeBook.pdf
    Local paths:   base_dir/SomeBook.pdf

    Falls back to case-insensitive match for .PDF vs .pdf mismatches.
    """
    target = base_dir / Path(full_path).name
    if target.exists():
        return target
    # Case-insensitive fallback (handles .PDF vs .pdf)
    cache = _build_case_cache(base_dir)
    name_lower = Path(full_path).name.lower()
    if name_lower in cache:
        return cache[name_lower]
    return target  # Return original (will fail at open time with clear error)


# --- CLI ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mass ingest books from library catalog.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Local directory containing copied PDFs",
    )
    parser.add_argument(
        "--no-embed", action="store_true", default=True, help="Skip embeddings (default: True)"
    )
    parser.add_argument(
        "--with-embed", action="store_true", help="Enable embedding (unsafe on shared GPU)"
    )
    parser.add_argument("--domain", type=str, help="Ingest only this domain")
    parser.add_argument("--min-priority", type=int, help="Min priority_score filter")
    parser.add_argument("--max-per-domain", type=int, help="Cap N books per domain")
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        help="Tier shortcut: 1=top 10/domain, 2=next 10, 3=rest",
    )
    parser.add_argument("--max-size-mb", type=float, help="Skip PDFs larger than N MB")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be ingested")
    parser.add_argument("--retry-failed", action="store_true", help="Retry previously failed")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--json", action="store_true", help="JSON summary output")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Catalog JSON path")
    parser.add_argument(
        "--no-build-citations",
        action="store_true",
        help="Skip the post-ingest citation graph build (default: on)",
    )
    parser.add_argument(
        "--auto-stop-services",
        action="store_true",
        help=(
            "Stop research_kb_rerank.service if VRAM is below the Docling floor, "
            "then restart on exit. Default: abort with a systemctl hint."
        ),
    )
    return parser.parse_args()


# --- Dry run ---


def print_dry_run(entries: list[dict], base_dir: Path) -> None:
    """Show what would be ingested without touching the DB."""
    domain_counts = Counter(e["domain"] for e in entries)
    total_size_gb = sum(e.get("file_size_mb", 0) for e in entries) / 1024

    print(f"Would ingest {len(entries)} books ({total_size_gb:.1f} GB)")
    print(f"\nPer domain ({len(domain_counts)} domains):")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")

    missing = sum(1 for e in entries if not remap_path(e["full_path"], base_dir).exists())
    if missing:
        print(f"\nWARNING: {missing}/{len(entries)} files not found in {base_dir}")


# --- Main loop ---


async def main() -> None:
    args = parse_args()

    if args.quiet:
        configure_logging(level="ERROR")

    # Load catalog
    entries = load_catalog(args.catalog)
    if not entries:
        print("No non-skip book entries in catalog.")
        return

    # Handle --retry-failed before filtering
    checkpoint = load_checkpoint()
    if args.retry_failed:
        checkpoint["failed"] = {}
        save_checkpoint(checkpoint)

    # Filter
    entries = apply_filters(entries, args)
    if not entries:
        print("No entries match filters.")
        return

    # Dry run exits early
    if args.dry_run:
        print_dry_run(entries, args.base_dir)
        return

    # Validate base-dir exists
    if not args.base_dir.exists():
        print(f"Error: --base-dir does not exist: {args.base_dir}")
        sys.exit(1)

    # Docling VRAM preflight — applies to any real (non-dry-run) pass.
    # Mirrors ingest_missing_textbooks / ingest_missing_papers wiring.
    from research_kb_common.gpu_guard import (
        DEFAULT_DOCLING_VRAM_MIN_MB,
        MINERU_MANAGED_SERVICES,
        abort_if_vram_insufficient,
        restart_services,
    )

    _stopped_services = abort_if_vram_insufficient(
        min_mib=DEFAULT_DOCLING_VRAM_MIN_MB,
        managed_services=MINERU_MANAGED_SERVICES,
        auto_stop_services=args.auto_stop_services,
        caller_label="docling preflight (mass_ingest_catalog)",
    )

    # Initialize DB
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Build lookup sets from checkpoint
    completed_set = set(checkpoint["completed"])
    failed_set = set(checkpoint["failed"].keys())

    results: dict = {
        "success": [],
        "failed": [],
        "skipped_checkpoint": 0,
        "skipped_db": 0,
        "skipped_missing": 0,
    }

    total = len(entries)
    attempt = 0  # Counter for entries we actually attempt

    for i, entry in enumerate(entries):
        sha = entry["sha256"]
        title = entry.get("title") or entry["filename"]
        domain = entry["domain"]
        pos = i + 1

        # Skip if in checkpoint (completed or failed)
        if sha in completed_set:
            results["skipped_checkpoint"] += 1
            continue
        if sha in failed_set:
            results["skipped_checkpoint"] += 1
            continue

        # Remap path and verify file exists
        local_path = remap_path(entry["full_path"], args.base_dir)
        if not local_path.exists():
            results["skipped_missing"] += 1
            if not args.quiet and not args.json:
                print(f"  [{pos}/{total}] MISSING: {entry['filename']}")
            continue

        # Check DB by file hash (dedup)
        existing = await SourceStore.get_by_file_hash(sha)
        if existing:
            results["skipped_db"] += 1
            completed_set.add(sha)
            checkpoint["completed"].append(sha)
            save_checkpoint(checkpoint)
            continue

        # Validate domain
        if domain not in VALID_DOMAINS:
            logger.warning("unknown_domain", domain=domain, file=entry["filename"])
            domain = "machine_learning"

        # Build metadata from catalog fields
        metadata: dict = {
            "auto_ingested": True,
            "source": "mass_ingest_catalog",
            "catalog_priority": entry.get("priority_score"),
            "catalog_confidence": entry.get("confidence"),
        }
        if entry.get("secondary_domains"):
            metadata["secondary_domains"] = entry["secondary_domains"]
        if entry.get("publisher"):
            metadata["publisher"] = entry["publisher"]
        if entry.get("edition"):
            metadata["edition"] = entry["edition"]

        attempt += 1
        if not args.quiet and not args.json:
            size_mb = entry.get("file_size_mb", 0)
            print(f"\n[{pos}/{total}] {title} ({domain}, {size_mb:.0f}MB)")

        start = time.time()
        try:
            source_id, num_chunks, num_headings = await ingest_textbook(
                pdf_path=str(local_path),
                title=title,
                authors=entry.get("authors", []),
                year=entry.get("year"),
                domain_id=domain,
                metadata=metadata,
                quiet=args.quiet,
                no_embed=args.no_embed,
            )
            elapsed = time.time() - start

            results["success"].append(
                {
                    "file": entry["filename"],
                    "title": title,
                    "domain": domain,
                    "chunks": num_chunks,
                    "elapsed_seconds": round(elapsed, 1),
                    "source_id": source_id,
                }
            )

            # Update checkpoint
            completed_set.add(sha)
            checkpoint["completed"].append(sha)
            save_checkpoint(checkpoint)

            if args.quiet and not args.json:
                print(
                    f"OK [{pos}/{total}] {entry['filename']}: {num_chunks} chunks ({elapsed:.0f}s)"
                )
            elif not args.json:
                print(f"  OK {num_chunks} chunks ({elapsed:.0f}s)")

            # Docling leaks ~1GB/book; force GC to reclaim
            gc.collect()

        except (MemoryError, OSError) as e:
            error_type = "memory_exhausted" if isinstance(e, MemoryError) else "file_io_error"
            results["failed"].append(
                {
                    "file": entry["filename"],
                    "error": f"{error_type}: {str(e)[:100]}",
                    "recoverable": False,
                }
            )
            checkpoint["failed"][sha] = f"{error_type}: {str(e)[:100]}"
            save_checkpoint(checkpoint)
            if args.quiet and not args.json:
                print(f"FAIL [{pos}/{total}] {entry['filename']}: {error_type}")
            elif not args.json:
                print(f"  FAIL {error_type}: {str(e)[:80]}")

        except (EmbeddingError, ConnectionError) as e:
            results["failed"].append(
                {
                    "file": entry["filename"],
                    "error": "Embedding service failure",
                    "recoverable": True,
                }
            )
            checkpoint["failed"][sha] = f"embedding: {str(e)[:100]}"
            save_checkpoint(checkpoint)
            if args.quiet and not args.json:
                print(f"FAIL [{pos}/{total}] {entry['filename']}: embedding failure (recoverable)")

        except StorageError as e:
            results["failed"].append(
                {
                    "file": entry["filename"],
                    "error": f"Database error: {str(e)[:100]}",
                    "recoverable": True,
                }
            )
            checkpoint["failed"][sha] = f"storage: {str(e)[:100]}"
            save_checkpoint(checkpoint)
            if args.quiet and not args.json:
                print(f"FAIL [{pos}/{total}] {entry['filename']}: DB error (recoverable)")

        except Exception as e:
            results["failed"].append(
                {
                    "file": entry["filename"],
                    "error": str(e)[:100],
                    "recoverable": False,
                }
            )
            checkpoint["failed"][sha] = str(e)[:100]
            save_checkpoint(checkpoint)
            if args.quiet and not args.json:
                print(f"FAIL [{pos}/{total}] {entry['filename']}: {str(e)[:60]}")

    # --- Summary ---
    total_chunks = sum(r["chunks"] for r in results["success"])
    domain_counts = Counter(r["domain"] for r in results["success"])

    if args.json:
        summary = {
            "success_count": len(results["success"]),
            "failed_count": len(results["failed"]),
            "total_chunks": total_chunks,
            "skipped_checkpoint": results["skipped_checkpoint"],
            "skipped_db": results["skipped_db"],
            "skipped_missing": results["skipped_missing"],
            "per_domain": dict(sorted(domain_counts.items())),
            "failed_files": [
                {"file": r["file"], "error": r["error"], "recoverable": r.get("recoverable", False)}
                for r in results["failed"]
            ],
        }
        print(json.dumps(summary, indent=2))
    elif args.quiet:
        print(f"\nIngested: {len(results['success'])} books | {total_chunks} chunks")
        if results["skipped_checkpoint"]:
            print(f"Skipped (checkpoint): {results['skipped_checkpoint']}")
        if results["skipped_db"]:
            print(f"Skipped (in DB): {results['skipped_db']}")
        if results["skipped_missing"]:
            print(f"Skipped (missing): {results['skipped_missing']}")
        if results["failed"]:
            recoverable = sum(1 for r in results["failed"] if r.get("recoverable"))
            print(f"Failed: {len(results['failed'])} ({recoverable} recoverable)")
    else:
        print("\n" + "=" * 70)
        print("MASS INGESTION SUMMARY")
        print("=" * 70)
        print(f"Ingested:  {len(results['success'])} books ({total_chunks} chunks)")
        print(f"Failed:    {len(results['failed'])}")
        print(f"Checkpoint:{results['skipped_checkpoint']}")
        print(f"In DB:     {results['skipped_db']}")
        print(f"Missing:   {results['skipped_missing']}")

        if domain_counts:
            print(f"\nPer domain ({len(domain_counts)} domains):")
            for domain, count in sorted(domain_counts.items()):
                domain_chunks = sum(
                    r["chunks"] for r in results["success"] if r["domain"] == domain
                )
                print(f"  {domain}: {count} books, {domain_chunks} chunks")

        if results["failed"]:
            print("\nFailed files:")
            for r in results["failed"]:
                tag = "recoverable" if r.get("recoverable") else "permanent"
                print(f"  - {r['file']}: {r['error'][:60]} ({tag})")

    # Post-ingest hook: build citation graph for just the new sources (Issue #5).
    # Mass ingests can be huge; users can skip with --no-build-citations.
    new_ids: list[UUID] = []
    for r in results["success"]:
        if r.get("source_id") is None:
            continue
        try:
            new_ids.append(UUID(str(r["source_id"])))
        except (TypeError, ValueError):
            logger.warning("invalid_source_id", source_id=r.get("source_id"))

    if new_ids and not getattr(args, "no_build_citations", False):
        try:
            if not args.json:
                print(f"\nBuilding citation graph for {len(new_ids)} new sources...")
            summary = await run_post_ingest_hooks(new_ids)
            citation_stats = summary.get("citations", {})
            if not args.json:
                print(
                    f"  matched={citation_stats.get('matched', 0)} "
                    f"unmatched={citation_stats.get('unmatched', 0)}"
                )
        except Exception as e:
            logger.error("post_ingest_hook_failed", error=str(e))
            if not args.json:
                print(f"  ⚠ citation graph build failed: {e}")

    # Restart any services the VRAM preflight stopped at startup.
    if _stopped_services:
        if not args.quiet:
            print(f"  [preflight] Restarting: {', '.join(_stopped_services)}")
        restart_services(_stopped_services)


if __name__ == "__main__":
    asyncio.run(main())
