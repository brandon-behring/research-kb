#!/usr/bin/env python3
"""Manifest-driven batch ingestion for reinforcement learning books.

All books use MinerU extraction for consistent formula handling.
Designed for the RTX 2070 three-phase GPU workflow:

  Phase 1: python scripts/ingest_rl_batch.py              # MinerU extraction (--no-embed default)
  Phase 2: python scripts/backfill_embeddings.py --batch-size 8  # Embed (embed_server on GPU)
  Phase 3: python scripts/build_citation_graph.py          # Citation graph (CPU only)

Usage:
    python scripts/ingest_rl_batch.py                      # Extract all, no embed (default)
    python scripts/ingest_rl_batch.py --dry-run            # Preview manifest
    python scripts/ingest_rl_batch.py --resume             # Resume from checkpoint
    python scripts/ingest_rl_batch.py --only 0,1,2,3      # Extract P1 books only
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from uuid import UUID

from research_kb_common import get_logger
from research_kb_contracts import SourceType
from research_kb_pdf.mineru_extractor import extract_and_chunk
from research_kb_pdf.post_ingest import run_post_ingest_hooks
from research_kb_storage import (
    ChunkStore,
    SourceStore,
    close_connection_pool,
    get_connection_pool,
)

logger = get_logger(__name__)

# ── Checkpoint ────────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path(__file__).parent.parent / "data" / "rl_batch_checkpoint.json"


def load_checkpoint() -> set[str]:
    """Load completed book keys from checkpoint file."""
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        return set(data.get("completed", []))
    return set()


def save_checkpoint(completed: set[str]) -> None:
    """Atomically save checkpoint."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"completed": sorted(completed)}, indent=2))
    tmp.rename(CHECKPOINT_PATH)


# ── Manifest (priority-ordered: P1 → P2 → P3) ──────────────────────────

RL = Path("fixtures/library_books/reinforcement_learning")

MANIFEST: list[dict[str, Any]] = [
    # ── P1: Core RL Textbooks ───────────────────────────────────────────
    {
        "slug": "sutton_barto_rl_2e",
        "title": "Reinforcement Learning: An Introduction, 2nd Edition",
        "authors": ["Richard S. Sutton", "Andrew G. Barto"],
        "year": 2018,
        "domain_id": "reinforcement_learning",
        "pdf": RL / "sutton_barto_rl_intro_2e_2018.pdf",
        "action": "new",
    },
    {
        "slug": "bertsekas_rl_optimal_control",
        "title": "Reinforcement Learning and Optimal Control",
        "authors": ["Dimitri P. Bertsekas"],
        "year": 2019,
        "domain_id": "reinforcement_learning",
        "pdf": RL / "bertsekas_rl_optimal_control_2019.pdf",
        "action": "new",
    },
    {
        "slug": "bertsekas_course_rl_2e",
        "title": "A Course in Reinforcement Learning, 2nd Edition",
        "authors": ["Dimitri P. Bertsekas"],
        "year": 2025,
        "domain_id": "reinforcement_learning",
        "pdf": RL / "bertsekas_course_in_rl_2e_2025.pdf",
        "action": "new",
    },
    {
        "slug": "szepesvari_algorithms_rl",
        "title": "Algorithms for Reinforcement Learning",
        "authors": ["Csaba Szepesvári"],
        "year": 2010,
        "domain_id": "reinforcement_learning",
        "pdf": RL / "szepesvari_algorithms_for_rl.pdf",
        "action": "new",
    },
    # ── P2: Supporting Texts ────────────────────────────────────────────
    {
        "slug": "dudik_doubly_robust",
        "title": "Doubly Robust Policy Evaluation and Optimization",
        "authors": ["Miroslav Dudík", "Dumitru Erhan", "John Langford", "Lihong Li"],
        "year": 2014,
        "domain_id": "reinforcement_learning",
        "source_type": "paper",
        "pdf": RL / "dudik_doubly_robust_policy_eval.pdf",
        "action": "new",
    },
    {
        "slug": "bertsekas_dp_vol1",
        "title": "Dynamic Programming and Optimal Control, Vol. 1",
        "authors": ["Dimitri P. Bertsekas"],
        "year": 2005,
        "domain_id": "reinforcement_learning",
        "pdf": RL / "bertsekas_dynamic_programming_vol1_2005.pdf",
        "action": "new",
        "metadata_extra": {"cross_domain": ["optimization"]},
    },
    {
        "slug": "bertsekas_abstract_dp",
        "title": "Abstract Dynamic Programming, 3rd Edition",
        "authors": ["Dimitri P. Bertsekas"],
        "year": 2022,
        "domain_id": "reinforcement_learning",
        "pdf": RL / "bertsekas_abstract_dp_3e.pdf",
        "action": "new",
        "metadata_extra": {"cross_domain": ["optimization"]},
    },
    {
        "slug": "bertsekas_alphazero",
        "title": "Lessons from AlphaZero for Optimal, Model Predictive, and Adaptive Control",
        "authors": ["Dimitri P. Bertsekas"],
        "year": 2022,
        "domain_id": "reinforcement_learning",
        "pdf": RL / "bertsekas_lessons_from_alphazero.pdf",
        "action": "new",
    },
    {
        "slug": "grokking_deep_rl",
        "title": "Grokking Deep Reinforcement Learning",
        "authors": ["Miguel Morales"],
        "year": 2020,
        "domain_id": "reinforcement_learning",
        "pdf": RL / "grokking_deep_rl.pdf",
        "action": "new",
    },
    # ── P3: Practical Guides ────────────────────────────────────────────
    {
        "slug": "deep_rl_in_action",
        "title": "Deep Reinforcement Learning in Action",
        "authors": ["Alexander Zai", "Brandon Brown"],
        "year": 2020,
        "domain_id": "reinforcement_learning",
        "pdf": RL / "deep_rl_in_action.pdf",
        "action": "new",
    },
]


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


async def ingest_one(entry: dict, quiet: bool = False) -> tuple[str, int]:
    """Extract one PDF with MinerU and store in DB.

    Returns
    -------
    tuple[str, int]
        (source_id, chunk_count)
    """
    pdf_path = str(entry["pdf"])
    title = entry["title"]
    authors = entry["authors"]
    year = entry["year"]
    domain_id = entry["domain_id"]
    source_type = entry.get("source_type", "textbook")

    if not quiet:
        logger.info("extracting_pdf", path=pdf_path, title=title)

    t0 = time.time()
    extraction_result, chunks = extract_and_chunk(pdf_path, max_tokens=300)
    elapsed = time.time() - t0

    metadata = {
        "extraction_method": "mineru",
        "total_pages": extraction_result.total_pages,
        "total_chars": extraction_result.total_chars,
        "total_headings": extraction_result.heading_count,
        "total_chunks": len(chunks),
        "has_equations": extraction_result.has_equations,
        "extraction_time_s": round(elapsed, 1),
        "domain": domain_id,
        "batch": "rl_batch",
    }

    # Merge extra metadata (e.g., cross_domain)
    if entry.get("metadata_extra"):
        metadata.update(entry["metadata_extra"])

    if not quiet:
        logger.info(
            "extraction_complete",
            title=title,
            chunks=len(chunks),
            pages=extraction_result.total_pages,
            time_s=round(elapsed, 1),
        )

    file_hash = compute_file_hash(pdf_path)

    st = SourceType.PAPER if source_type == "paper" else SourceType.TEXTBOOK
    source = await SourceStore.create(
        source_type=st,
        title=title,
        file_hash=file_hash,
        domain_id=domain_id,
        authors=authors,
        year=year,
        file_path=pdf_path,
        metadata=metadata,
    )

    # Batch insert chunks
    chunks_data = []
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        chunks_data.append(
            {
                "source_id": source.id,
                "content": chunk.content,
                "content_hash": content_hash,
                "page_start": chunk.start_page,
                "page_end": chunk.end_page,
                "embedding": None,  # Phase 2
                "domain_id": domain_id,
                "metadata": {
                    "section_header": chunk.metadata.get("section", ""),
                    "chunk_index": i,
                },
            }
        )

    BATCH_SIZE = 100
    chunks_created = 0
    for i in range(0, len(chunks_data), BATCH_SIZE):
        batch = chunks_data[i : i + BATCH_SIZE]
        await ChunkStore.batch_create(batch)
        chunks_created += len(batch)

    if not quiet:
        logger.info(
            "ingestion_complete",
            source_id=str(source.id),
            title=title,
            chunks=chunks_created,
        )

    return str(source.id), chunks_created


async def run(args: argparse.Namespace) -> None:
    """Main entry point."""
    # Filter manifest
    manifest = MANIFEST
    if args.only:
        indices = {int(x.strip()) for x in args.only.split(",")}
        manifest = [m for i, m in enumerate(manifest) if i in indices]

    # Dry run
    if args.dry_run:
        print(f"\n{'#':>3s}  {'ACTION':10s}  {'DOMAIN':25s}  {'KEY':40s}  PDF EXISTS?")
        print("-" * 110)
        for i, entry in enumerate(manifest):
            exists = entry["pdf"].exists()
            flag = "OK" if exists else "MISSING"
            print(
                f"{i:3d}  {entry['action']:10s}  {entry['domain_id']:25s}  {entry['key']:40s}  {flag}"
            )
            if not exists:
                print(f"     -> {entry['pdf']}")
        missing = [e for e in manifest if not e["pdf"].exists()]
        print(f"\nTotal: {len(manifest)} books, {len(missing)} PDFs missing")
        return

    # Verify all PDFs exist before starting
    missing = [e for e in manifest if not e["pdf"].exists()]
    if missing:
        print("ERROR: Missing PDFs:")
        for e in missing:
            print(f"  {e['key']}: {e['pdf']}")
        sys.exit(1)

    # Load checkpoint
    completed = load_checkpoint() if args.resume else set()

    # Initialize DB pool
    await get_connection_pool()

    total_chunks = 0
    total_ingested = 0
    failed: list[tuple[str, str]] = []
    new_source_ids: list[UUID] = []

    for i, entry in enumerate(manifest):
        key = entry["slug"]

        if key in completed:
            print(f"[{i+1}/{len(manifest)}] SKIP (checkpoint): {key}")
            continue

        print(f"\n[{i+1}/{len(manifest)}] {entry['action'].upper()}: {entry['title']}")

        try:
            source_id, chunk_count = await ingest_one(entry, quiet=args.quiet)
            total_chunks += chunk_count
            total_ingested += 1
            try:
                new_source_ids.append(UUID(source_id))
            except (TypeError, ValueError):
                logger.warning("invalid_source_id", source_id=str(source_id))

            print(f"  OK: {chunk_count} chunks, source_id={source_id[:12]}")

            # Checkpoint
            completed.add(key)
            save_checkpoint(completed)

        except Exception as e:
            print(f"  FAILED: {e}")
            logger.error("book_failed", key=key, error=str(e))
            failed.append((key, str(e)))

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"  Ingested: {total_ingested}/{len(manifest)} books")
    print(f"  Chunks:   {total_chunks}")
    print(f"  Failed:   {len(failed)}")
    if failed:
        for key, err in failed:
            print(f"    {key}: {err[:80]}")
    print(f"{'=' * 60}")

    # Post-ingest hook: build citation graph for just the new sources (Issue #5)
    if new_source_ids and not args.no_build_citations:
        try:
            print(f"\nBuilding citation graph for {len(new_source_ids)} new sources...")
            summary = await run_post_ingest_hooks(new_source_ids)
            citation_stats = summary.get("citations", {})
            print(
                f"  matched={citation_stats.get('matched', 0)} "
                f"unmatched={citation_stats.get('unmatched', 0)}"
            )
        except Exception as e:
            print(f"  ⚠ citation graph build failed: {e}")
            logger.error("post_ingest_hook_failed", error=str(e))

    await close_connection_pool()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL batch ingestion with MinerU extraction (10 books, priority-ordered)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview manifest without executing")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated manifest indices to process (e.g., '0,1,2,3' for P1 only)",
    )
    parser.add_argument(
        "--no-build-citations",
        action="store_true",
        help="Skip the post-ingest citation graph build (default: on)",
    )
    args = parser.parse_args()

    import asyncio

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
