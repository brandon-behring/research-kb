#!/usr/bin/env python3
"""Manifest-driven batch ingestion using MinerU extraction.

Handles both new ingestion and re-extraction (delete old + re-ingest) in one pass.
Designed for the RTX 2070 three-phase GPU workflow:

  Phase 1: python scripts/ingest_bayesian_batch.py          # MinerU extraction (--no-embed default)
  Phase 2: python scripts/backfill_embeddings.py --batch-size 8  # Embed (embed_server on GPU)
  Phase 3: python scripts/build_citation_graph.py            # Citation graph (CPU only)

Usage:
    python scripts/ingest_bayesian_batch.py                  # Extract all, no embed (default)
    python scripts/ingest_bayesian_batch.py --dry-run        # Preview manifest
    python scripts/ingest_bayesian_batch.py --resume         # Resume from checkpoint
    python scripts/ingest_bayesian_batch.py --only 3,5,7     # Extract specific manifest entries
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any
from uuid import UUID

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

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

CHECKPOINT_PATH = Path(__file__).parent.parent / "data" / "bayesian_batch_checkpoint.json"


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


# ── Manifest ──────────────────────────────────────────────────────────────

LIB = Path("fixtures/library_books")
TXT = Path("fixtures/textbooks")

MANIFEST: list[dict[str, Any]] = [
    # ── New ingestion (8 books) ──────────────────────────────────────────
    {
        "slug": "martin_bayesian_python_3e",
        "title": "Bayesian Analysis with Python",
        "authors": ["Osvaldo Martin", "Christopher Fonnesbeck", "Thomas Wiecki"],
        "year": 2024,
        "domain_id": "statistics",
        "pdf": LIB / "martin_bayesian_analysis_python_3e_2024.pdf",
        "action": "new",
    },
    {
        "slug": "kruschke_doing_bayesian_2e",
        "title": "Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan",
        "authors": ["John K. Kruschke"],
        "year": 2014,
        "domain_id": "statistics",
        "pdf": LIB / "kruschke_doing_bayesian_data_analysis_2e_2014.pdf",
        "action": "new",
    },
    {
        "slug": "mcelreath_statistical_rethinking_2e",
        "title": "Statistical Rethinking: A Bayesian Course with Examples in R and Stan",
        "authors": ["Richard McElreath"],
        "year": 2020,
        "domain_id": "statistics",
        "pdf": LIB / "mcelreath_statistical_rethinking_2e_2020.pdf",
        "action": "new",
    },
    {
        "slug": "murphy_pml_advanced_topics",
        "title": "Probabilistic Machine Learning: Advanced Topics",
        "authors": ["Kevin P. Murphy"],
        "year": 2023,
        "domain_id": "machine_learning",
        "pdf": LIB / "murphy_pml_advanced_topics_2025.pdf",
        "action": "new",
    },
    {
        "slug": "pole_west_applied_bayesian_forecasting",
        "title": "Applied Bayesian Forecasting and Time Series Analysis",
        "authors": ["Andy Pole", "Mike West", "Jeff Harrison"],
        "year": 1994,
        "domain_id": "time_series",
        "pdf": LIB
        / "Andy Pole, Mike West, Jeff Harrison (auth.) - Applied Bayesian Forecasting and Time Series Analysis-Springer US (1994).pdf",
        "action": "new",
    },
    {
        "slug": "west_prado_time_series",
        "title": "Time Series: Modeling, Computation, and Inference",
        "authors": ["Mike West", "Raquel Prado"],
        "year": 2010,
        "domain_id": "time_series",
        "pdf": LIB
        / "(Chapman & Hall_CRC Texts in Statistical Science) West, Mike_ Prado, Raquel - Time Series_ Modeling, Computation, and Inference-Chapman and Hall_CRC (2010).pdf",
        "action": "new",
    },
    {
        "slug": "hogg_tanis_probability_statistical_inference",
        "title": "Probability and Statistical Inference",
        "authors": ["Robert V. Hogg", "Elliot Tanis", "Dale Zimmerman"],
        "year": 2019,
        "domain_id": "statistics",
        "pdf": LIB
        / "(10) by Robert V. Hogg, Elliot Tanis, Dale Zimmerman - Probability and Statistical Inference (10th Edition) (2019).pdf",
        "action": "new",
    },
    # BDA fresh re-ingest (both old copies deleted below)
    {
        "slug": "gelman_bda_3e",
        "title": "Bayesian Data Analysis",
        "authors": [
            "Andrew Gelman",
            "John B. Carlin",
            "Hal S. Stern",
            "David B. Dunson",
            "Aki Vehtari",
            "Donald B. Rubin",
        ],
        "year": 2014,
        "domain_id": "statistics",
        "pdf": LIB
        / "data_science/(Chapman & Hall_CRC Texts in Statistical Science) Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin - Bayesian Data Analysis-Chapman and Hall_CRC (2014).pdf",
        "action": "new",
        "delete_titles": [
            # Delete both BDA duplicates before ingesting fresh copy
            "Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donal",
            "Bayesian Data Analysis Third Edition 3rd Edition -- Gelman Andrew Carlin John B",
        ],
    },
    # ── Re-extraction (7 existing books) ─────────────────────────────────
    {
        "slug": "murphy_pml_introduction",
        "title": "Probabilistic Machine Learning: An Introduction",
        "authors": ["Kevin P. Murphy"],
        "year": 2021,
        "domain_id": "machine_learning",
        "pdf": LIB
        / "Kevin P. Murphy - Probabilistic Machine Learning_ An Introduction-The MIT Press (2021).pdf",
        "action": "reextract",
        "delete_title_pattern": "Probabilistic Machine Learning: An Introduction",
    },
    {
        "slug": "lee_bayesian_statistics_intro",
        "title": "Bayesian Statistics: An Introduction",
        "authors": ["Peter M. Lee"],
        "year": 2012,
        "domain_id": "statistics",
        "pdf": LIB
        / "Lee, Peter - Bayesian Statistics_ an Introduction-John Wiley & Sons (2012).pdf",
        "action": "reextract",
        "delete_title_pattern": "Lee, Peter - Bayesian Statistics%",
    },
    {
        "slug": "barber_bayesian_time_series",
        "title": "Bayesian Time Series Models",
        "authors": ["David Barber", "A. Taylan Cemgil", "Silvia Chiappa"],
        "year": 2011,
        "domain_id": "time_series",
        "pdf": LIB
        / "Barber D., Cemgil A.T., Chiappa S. (eds.) - Bayesian Time Series Models-Cambridge University Press (2011).pdf",
        "action": "reextract",
        "delete_title_pattern": "Barber D.%Bayesian Time Series%",
    },
    {
        "slug": "koller_friedman_pgm",
        "title": "Probabilistic Graphical Models: Principles and Techniques",
        "authors": ["Daphne Koller", "Nir Friedman"],
        "year": 2009,
        "domain_id": "statistics",
        "pdf": LIB / "KollerFriedman.pdf",
        "action": "reextract",
        "delete_title_pattern": "Koller Friedman%",
    },
    {
        "slug": "donovan_bayesian_beginners",
        "title": "Bayesian Statistics for Beginners: A Step-by-Step Approach",
        "authors": ["Therese M. Donovan", "Ruth M. Mickey"],
        "year": 2019,
        "domain_id": "statistics",
        "pdf": LIB
        / "Therese M Donovan_ Ruth M Mickey - Bayesian Statistics for Beginners_ A Step-By-Step Approach-Oxford University Press, USA (2019).pdf",
        "action": "reextract",
        "delete_title_pattern": "Therese M Donovan%Bayesian Statistics for Beginners%",
    },
    {
        "slug": "gelman_applied_bayesian_modeling_ci",
        "title": "Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives",
        "authors": ["Andrew Gelman", "Donald B. Rubin", "Xiao-Li Meng"],
        "year": 2004,
        "domain_id": "causal_inference",
        "pdf": TXT
        / "Applied Bayesian modeling and causal inference from -- Donald B Rubin; Andrew Gelman; Xiao-Li Meng -- Wiley Series in Probability and Statistics, -- 9780470090435 -- 6b4631b748ce7a0ce1b9e18edbfee08c -- Anna\u2019s Archive (1).pdf",
        "action": "reextract",
        "delete_title_pattern": "Applied Bayesian Modeling and Causal Inference%",
    },
    {
        "slug": "lambert_students_guide_bayesian",
        "title": "A Student's Guide to Bayesian Statistics",
        "authors": ["Ben Lambert"],
        "year": 2018,
        "domain_id": "statistics",
        "pdf": LIB / "lambert_students_guide_bayesian_statistics.pdf",
        "action": "reextract",
        "delete_title_pattern": "A Student%Guide to Bayesian Statistics",
    },
]


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


async def delete_sources_by_title_pattern(pattern: str, quiet: bool = False) -> int:
    """Delete sources matching a LIKE pattern. Returns count deleted."""
    pool = await get_connection_pool()
    async with pool.acquire() as conn:
        # Find matching sources
        rows = await conn.fetch(
            "SELECT id, title FROM sources WHERE title LIKE $1",
            pattern,
        )
        if not rows:
            if not quiet:
                logger.warning("no_sources_matched", pattern=pattern)
            return 0

        for row in rows:
            sid = row["id"]
            # Delete chunks first (in case CASCADE isn't set on all FKs)
            chunk_count = await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE source_id = $1", sid
            )
            await conn.execute("DELETE FROM chunks WHERE source_id = $1", sid)
            await conn.execute("DELETE FROM sources WHERE id = $1", sid)
            if not quiet:
                logger.info(
                    "deleted_source",
                    source_id=str(sid),
                    title=row["title"][:60],
                    chunks=chunk_count,
                )
        return len(rows)


async def delete_sources_by_title_prefix(prefixes: list[str], quiet: bool = False) -> int:
    """Delete sources whose title starts with any of the given prefixes."""
    total = 0
    pool = await get_connection_pool()
    async with pool.acquire() as conn:
        for prefix in prefixes:
            rows = await conn.fetch(
                "SELECT id, title FROM sources WHERE title LIKE $1",
                prefix + "%",
            )
            for row in rows:
                sid = row["id"]
                chunk_count = await conn.fetchval(
                    "SELECT count(*) FROM chunks WHERE source_id = $1", sid
                )
                await conn.execute("DELETE FROM chunks WHERE source_id = $1", sid)
                await conn.execute("DELETE FROM sources WHERE id = $1", sid)
                total += 1
                if not quiet:
                    logger.info(
                        "deleted_source",
                        source_id=str(sid),
                        title=row["title"][:60],
                        chunks=chunk_count,
                    )
    return total


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
    }

    if not quiet:
        logger.info(
            "extraction_complete",
            title=title,
            chunks=len(chunks),
            pages=extraction_result.total_pages,
            time_s=round(elapsed, 1),
        )

    file_hash = compute_file_hash(pdf_path)

    source = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,
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
    import asyncio

    # Filter manifest
    manifest = MANIFEST
    if args.only:
        indices = {int(x.strip()) for x in args.only.split(",")}
        manifest = [m for i, m in enumerate(manifest) if i in indices]

    # Dry run
    if args.dry_run:
        print(f"\n{'#':>3s}  {'ACTION':10s}  {'DOMAIN':20s}  {'KEY':45s}  PDF EXISTS?")
        print("-" * 110)
        for i, entry in enumerate(manifest):
            exists = entry["pdf"].exists()
            flag = "OK" if exists else "MISSING"
            print(
                f"{i:3d}  {entry['action']:10s}  {entry['domain_id']:20s}  {entry['key']:45s}  {flag}"
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
    total_deleted = 0
    total_ingested = 0
    failed = []
    new_source_ids: list[UUID] = []

    for i, entry in enumerate(manifest):
        key = entry["slug"]

        if key in completed:
            print(f"[{i+1}/{len(manifest)}] SKIP (checkpoint): {key}")
            continue

        print(f"\n[{i+1}/{len(manifest)}] {entry['action'].upper()}: {entry['title']}")

        try:
            # Step 1: Delete old sources if re-extraction or BDA cleanup
            if entry.get("delete_titles"):
                deleted = await delete_sources_by_title_prefix(
                    entry["delete_titles"], quiet=args.quiet
                )
                total_deleted += deleted
                print(f"  Deleted {deleted} old source(s)")

            if entry.get("delete_title_pattern"):
                deleted = await delete_sources_by_title_pattern(
                    entry["delete_title_pattern"], quiet=args.quiet
                )
                total_deleted += deleted
                print(f"  Deleted {deleted} old source(s)")

            # Step 2: Extract and ingest
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
    print(f"SUMMARY")
    print(f"  Ingested: {total_ingested}/{len(manifest)} books")
    print(f"  Chunks:   {total_chunks}")
    print(f"  Deleted:  {total_deleted} old sources")
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
        description="Manifest-driven batch ingestion with MinerU extraction"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview manifest without executing")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated manifest indices to process (e.g., '0,3,5')",
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
