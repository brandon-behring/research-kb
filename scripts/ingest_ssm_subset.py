#!/usr/bin/env python3
"""Ingest a targeted SSM/post-transformer paper subset.

This is a one-off wrapper around ingest_missing_papers.ingest_pdf() that only
processes a specific list of arXiv IDs relevant to the post_transformers
project (W1-W20 curriculum).

Written 2026-04-10 to support research-kb B2 phase after verification showed
13 SSM-relevant primaries missing from KB.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

# Add packages to path (same as ingest_missing_papers.py)
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from ingest_missing_papers import (  # noqa: E402
    compute_file_hash,
    get_metadata_for_pdf,
    ingest_pdf,
)
from research_kb_common import get_logger  # noqa: E402
from research_kb_pdf.post_ingest import run_post_ingest_hooks  # noqa: E402
from research_kb_storage import (  # noqa: E402
    DatabaseConfig,
    SourceStore,
    close_connection_pool,
    get_connection_pool,
)

logger = get_logger(__name__)

# Target arXiv IDs (B2 phase — verified missing from KB as of 2026-04-10).
TARGET_ARXIV_IDS = [
    "2212.14052",  # H3 Hungry Hungry Hippos
    "2307.08621",  # RetNet
    "2405.04517",  # xLSTM original
    "2501.12352",  # Test-Time Regression
    "2406.00209",  # Mamba Lyapunov (TMLR)
    "2504.19561",  # ESS Effective State Size
    "2511.17970",  # Controllability Influence Score
    "2512.16723",  # KOSS Kalman-Optimal SSMs
    "2312.04927",  # Zoology / MQAR benchmark
    "2404.06654",  # RULER benchmark
    "2110.13985",  # LSSL (S4 predecessor)
    "2402.18668",  # Based (Arora hybrid)
    "2303.06349",  # LRU (Linear Recurrent Units, Orvieto)
]

DOMAIN_ID = "deep_learning"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SSM/post-transformer paper subset")
    parser.add_argument(
        "--no-build-citations",
        action="store_true",
        help="Skip the post-ingest citation graph build (default: on)",
    )
    args = parser.parse_args()

    papers_dir = Path(__file__).parent.parent / "fixtures" / "papers" / "arxiv"

    config = DatabaseConfig()
    await get_connection_pool(config)

    results = {"success": [], "skipped": [], "failed": []}

    for arxiv_id in TARGET_ARXIV_IDS:
        pdf_path = papers_dir / f"{arxiv_id}.pdf"
        if not pdf_path.exists():
            print(f"✗ {arxiv_id}: PDF not found at {pdf_path}")
            results["failed"].append({"arxiv_id": arxiv_id, "error": "pdf missing"})
            continue

        file_hash = compute_file_hash(str(pdf_path))
        existing = await SourceStore.get_by_file_hash(file_hash)
        if existing:
            print(f"⊘ {arxiv_id}: already ingested as '{existing.title[:60]}'")
            results["skipped"].append({"arxiv_id": arxiv_id, "title": existing.title})
            continue

        title, authors, year, extra_metadata = get_metadata_for_pdf(pdf_path)
        extra_metadata["auto_ingested"] = True
        extra_metadata["ingest_batch"] = "ssm_subset_2026_04_10"
        paper_domain = extra_metadata.pop("sidecar_domain_id", None) or DOMAIN_ID

        print(f"→ {arxiv_id}: '{title[:60]}' (domain={paper_domain})")
        try:
            source_id, num_chunks, num_headings = await ingest_pdf(
                pdf_path=str(pdf_path),
                title=title,
                authors=authors,
                year=year,
                metadata=extra_metadata,
                domain_id=paper_domain,
                no_embed=True,  # two-phase workflow; backfill embeddings after
            )
            print(f"  ✓ source_id={source_id[:8]}, {num_chunks} chunks, {num_headings} headings")
            results["success"].append(
                {"arxiv_id": arxiv_id, "source_id": source_id, "chunks": num_chunks}
            )
        except Exception as e:
            print(f"  ✗ failed: {e}")
            results["failed"].append({"arxiv_id": arxiv_id, "error": str(e)[:200]})

    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Success: {len(results['success'])}")
    print(f"Skipped (already in KB): {len(results['skipped'])}")
    print(f"Failed: {len(results['failed'])}")

    if results["success"]:
        total_chunks = sum(r["chunks"] for r in results["success"])
        print(f"Total new chunks: {total_chunks}")

    if results["failed"]:
        print("\nFailed:")
        for r in results["failed"]:
            print(f"  - {r['arxiv_id']}: {r['error'][:80]}")

    # Post-ingest hook: build citation graph for just the new sources (Issue #5)
    new_ids: list[UUID] = []
    for r in results["success"]:
        try:
            new_ids.append(UUID(r["source_id"]))
        except (TypeError, ValueError):
            logger.warning("invalid_source_id", source_id=r.get("source_id"))

    if new_ids and not args.no_build_citations:
        try:
            print(f"\nBuilding citation graph for {len(new_ids)} new sources...")
            summary = await run_post_ingest_hooks(new_ids)
            citation_stats = summary.get("citations", {})
            print(
                f"  matched={citation_stats.get('matched', 0)} "
                f"unmatched={citation_stats.get('unmatched', 0)}"
            )
        except Exception as e:
            print(f"  ⚠ citation graph build failed: {e}")
            logger.error("post_ingest_hook_failed", error=str(e))

    await close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())
