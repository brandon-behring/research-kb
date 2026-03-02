#!/usr/bin/env python3
"""Re-chunk existing corpus using structure-driven chunking.

Replaces old fill-the-bucket chunks with heading-aware chunks for each source.
Per-source transactions: one failure doesn't block others.

Usage:
    python scripts/rechunk_corpus.py --dry-run              # Preview what would change
    python scripts/rechunk_corpus.py --limit 5              # Process first 5 sources
    python scripts/rechunk_corpus.py --domain causal_inference  # One domain only
    python scripts/rechunk_corpus.py --source-id <uuid>     # Single source
    python scripts/rechunk_corpus.py --quiet                # Minimal output
    python scripts/rechunk_corpus.py --json                 # JSON output for monitoring
    python scripts/rechunk_corpus.py --skip-backup          # Skip backup reminder

After rechunking, you MUST re-run:
    1. python scripts/regenerate_golden_candidates.py  (update eval chunk UUIDs)
    2. python scripts/extract_concepts.py              (concept extraction)
    3. python scripts/sync_kuzu.py                     (KuzuDB sync)
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from uuid import UUID

import numpy as np

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_pdf import EmbeddingClient, extract_with_headings
from research_kb_pdf.chunker import chunk_by_structure
from research_kb_storage import DatabaseConfig, get_connection_pool

logger = get_logger(__name__)


async def get_sources_to_rechunk(
    pool, domain: str | None, source_id: str | None, limit: int | None
):
    """Fetch sources that need rechunking.

    Skips sources where ALL chunks already have metadata.chunking_method = 'structure'.

    Args:
        pool: asyncpg connection pool
        domain: Optional domain_id filter
        source_id: Optional single source UUID
        limit: Optional max number of sources

    Returns:
        List of dicts with id, title, file_path, domain_id, old_chunk_count
    """
    async with pool.acquire() as conn:
        conditions = ["s.file_path IS NOT NULL"]
        params: list = []

        if source_id:
            params.append(UUID(source_id))
            conditions.append(f"s.id = ${len(params)}")

        if domain:
            params.append(domain)
            conditions.append(f"s.domain_id = ${len(params)}")

        where = " AND ".join(conditions)

        # Skip sources already fully rechunked
        query = f"""
            SELECT s.id, s.title, s.file_path, s.domain_id,
                   COUNT(c.id) as chunk_count,
                   COUNT(c.id) FILTER (
                       WHERE c.metadata->>'chunking_method' = 'structure'
                   ) as structure_count
            FROM sources s
            LEFT JOIN chunks c ON c.source_id = s.id
            WHERE {where}
            GROUP BY s.id, s.title, s.file_path, s.domain_id
            HAVING COUNT(c.id) = 0
                OR COUNT(c.id) FILTER (
                    WHERE c.metadata->>'chunking_method' = 'structure'
                ) < COUNT(c.id)
            ORDER BY s.title
        """

        if limit:
            params.append(limit)
            query += f" LIMIT ${len(params)}"

        rows = await conn.fetch(query, *params)

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "file_path": row["file_path"],
            "domain_id": row["domain_id"],
            "old_chunk_count": row["chunk_count"],
        }
        for row in rows
    ]


async def rechunk_source(
    pool, source: dict, embed_client: EmbeddingClient, dry_run: bool, quiet: bool
):
    """Re-chunk a single source: extract -> chunk -> delete old -> embed -> insert.

    Runs in a single transaction per source. Uses set_type_codec for JSONB
    and register_vector for pgvector column compatibility.

    Args:
        pool: asyncpg connection pool
        source: Source dict with id, title, file_path, domain_id, old_chunk_count
        embed_client: Embedding client for generating vectors
        dry_run: If True, only report what would change
        quiet: Minimal output

    Returns:
        Dict with results: old_count, new_count, delta, status, elapsed_s
    """
    from pgvector.asyncpg import register_vector

    source_id = source["id"]
    file_path = source["file_path"]
    title = source["title"]
    old_count = source["old_chunk_count"]
    t0 = time.time()

    # Verify file exists
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return {
            "title": title,
            "old_count": old_count,
            "new_count": 0,
            "delta": -old_count,
            "status": "skipped",
            "reason": f"PDF not found: {file_path}",
            "elapsed_s": 0.0,
        }

    try:
        # Extract and chunk
        doc, headings = extract_with_headings(str(pdf_path))
        new_chunks = chunk_by_structure(doc, headings, target_tokens=300)
        new_count = len(new_chunks)

        if dry_run:
            return {
                "title": title,
                "old_count": old_count,
                "new_count": new_count,
                "delta": new_count - old_count,
                "status": "dry_run",
                "reason": "",
                "elapsed_s": time.time() - t0,
            }

        # Generate embeddings in batches
        texts = [chunk.content for chunk in new_chunks]
        embeddings = embed_client.embed_batch(texts, batch_size=32)

        # Transaction: delete old chunks -> insert new
        async with pool.acquire() as conn:
            # Register codecs for JSONB and pgvector
            await conn.set_type_codec(
                "jsonb",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
            )
            await register_vector(conn)

            async with conn.transaction():
                # Delete old chunks (cascades to chunk_concepts)
                await conn.execute(
                    "DELETE FROM chunks WHERE source_id = $1", source_id
                )

                # Insert new chunks with proper type handling
                for chunk, embedding in zip(new_chunks, embeddings):
                    content_hash = hashlib.sha256(
                        chunk.content.encode("utf-8")
                    ).hexdigest()

                    # Convert embedding list to numpy array for pgvector
                    embedding_vec = np.array(embedding, dtype=np.float32)

                    await conn.execute(
                        """
                        INSERT INTO chunks (
                            source_id, content, content_hash,
                            page_start, page_end, embedding, metadata, domain_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        source_id,
                        chunk.content,
                        content_hash,
                        chunk.start_page,
                        chunk.end_page,
                        embedding_vec,
                        chunk.metadata,  # dict -> jsonb via set_type_codec
                        source["domain_id"],
                    )

        elapsed = time.time() - t0
        return {
            "title": title,
            "old_count": old_count,
            "new_count": new_count,
            "delta": new_count - old_count,
            "status": "success",
            "reason": "",
            "elapsed_s": elapsed,
        }

    except Exception as e:
        logger.error(
            "rechunk_failed",
            source_id=str(source_id),
            title=title,
            error=str(e),
        )
        return {
            "title": title,
            "old_count": old_count,
            "new_count": 0,
            "delta": -old_count,
            "status": "failed",
            "reason": str(e)[:200],
            "elapsed_s": time.time() - t0,
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Re-chunk corpus with structure-driven chunking."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without modifying data",
    )
    parser.add_argument(
        "--source-id",
        type=str,
        default=None,
        help="Re-chunk a single source by UUID",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Only re-chunk sources in this domain",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of sources to process",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="JSON output for programmatic monitoring",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip the backup reminder at start",
    )
    return parser.parse_args()


def format_eta(elapsed: float, completed: int, total: int) -> str:
    """Calculate and format ETA string.

    Args:
        elapsed: Seconds elapsed so far
        completed: Number of items completed
        total: Total number of items

    Returns:
        Formatted ETA string like "~2h 15m remaining"
    """
    if completed == 0:
        return "calculating..."
    rate = elapsed / completed
    remaining_s = rate * (total - completed)
    if remaining_s < 60:
        return f"~{remaining_s:.0f}s remaining"
    elif remaining_s < 3600:
        return f"~{remaining_s / 60:.0f}m remaining"
    else:
        hours = int(remaining_s // 3600)
        minutes = int((remaining_s % 3600) // 60)
        return f"~{hours}h {minutes}m remaining"


async def main():
    """Re-chunk the corpus using structure-driven chunking."""
    from research_kb_common import configure_logging

    args = parse_args()

    if args.quiet or args.json_output:
        configure_logging(level="ERROR")

    # Backup reminder
    if not args.skip_backup and not args.dry_run and not args.json_output:
        print("=" * 70)
        print("BACKUP REMINDER")
        print("=" * 70)
        print("Before rechunking, ensure you have a recent backup:")
        print("  ./scripts/backup_db.sh")
        print()
        print("This operation will DELETE and REPLACE all chunks for selected sources.")
        print("Press Ctrl+C to abort, or wait 5 seconds to continue...")
        print("=" * 70)
        try:
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    # Connect to database
    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    # Get sources to rechunk
    sources = await get_sources_to_rechunk(
        pool, domain=args.domain, source_id=args.source_id, limit=args.limit
    )

    if not sources:
        msg = "No sources need rechunking (all already have chunking_method='structure')."
        if args.json_output:
            print(json.dumps({"status": "no_work", "message": msg}))
        else:
            print(msg)
        return

    mode_label = "DRY RUN" if args.dry_run else "RECHUNKING"
    if not args.quiet and not args.json_output:
        print(f"\n{mode_label}: {len(sources)} sources")
        if args.domain:
            print(f"  Domain filter: {args.domain}")
        total_old_chunks = sum(s["old_chunk_count"] for s in sources)
        print(f"  Total existing chunks: {total_old_chunks:,}")
        print()

    # Initialize embedding client (not needed for dry run)
    embed_client = None
    if not args.dry_run:
        embed_client = EmbeddingClient()

    # Process each source
    results = []
    start_time = time.time()

    for i, source in enumerate(sources):
        if not args.quiet and not args.json_output:
            eta = format_eta(time.time() - start_time, i, len(sources)) if i > 0 else ""
            eta_suffix = f"  ({eta})" if eta else ""
            print(
                f"[{i + 1}/{len(sources)}] {source['title'][:55]}...{eta_suffix}",
                end=" ",
                flush=True,
            )

        result = await rechunk_source(
            pool, source, embed_client, dry_run=args.dry_run, quiet=args.quiet
        )
        results.append(result)

        if args.json_output:
            # Emit per-source JSON line for monitoring
            print(
                json.dumps(
                    {
                        "event": "source_complete",
                        "index": i + 1,
                        "total": len(sources),
                        **result,
                    }
                )
            )
        elif not args.quiet:
            if result["status"] == "success":
                print(
                    f"{result['old_count']} -> {result['new_count']} chunks "
                    f"(delta {result['delta']:+d}, {result['elapsed_s']:.1f}s)"
                )
            elif result["status"] == "dry_run":
                print(
                    f"{result['old_count']} -> {result['new_count']} "
                    f"(delta {result['delta']:+d}) [dry run]"
                )
            elif result["status"] == "skipped":
                print(f"SKIPPED: {result['reason']}")
            else:
                print(f"FAILED: {result['reason']}")
        elif args.quiet and result["status"] == "failed":
            print(f"FAIL {source['title'][:40]}: {result['reason']}")

    elapsed = time.time() - start_time

    # Summary
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    dry_run_results = [r for r in results if r["status"] == "dry_run"]

    total_old = sum(r["old_count"] for r in results)
    total_new = sum(r["new_count"] for r in results)

    summary = {
        "mode": "dry_run" if args.dry_run else "rechunk",
        "success_count": len(success) if not args.dry_run else len(dry_run_results),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "chunks_old": total_old,
        "chunks_new": total_new,
        "chunks_delta": total_new - total_old,
        "elapsed_s": round(elapsed, 1),
        "failed_sources": [
            {"title": r["title"][:50], "reason": r["reason"]} for r in failed
        ],
    }

    if args.json_output:
        print(json.dumps({"event": "summary", **summary}))
    else:
        print()
        print("=" * 70)
        print(f"{'DRY RUN ' if args.dry_run else ''}SUMMARY")
        print("=" * 70)

        if args.dry_run:
            print(f"Would rechunk: {len(dry_run_results)} sources")
        else:
            print(f"Success: {len(success)} sources")

        print(f"Skipped: {len(skipped)} sources")
        print(f"Failed: {len(failed)} sources")
        print(
            f"Chunks: {total_old:,} old -> {total_new:,} new "
            f"(delta {total_new - total_old:+,})"
        )
        print(f"Elapsed: {elapsed:.1f}s")

        if failed:
            print("\nFailed sources:")
            for r in failed:
                print(f"  - {r['title'][:50]}: {r['reason']}")

        if not args.dry_run and success:
            print()
            print("NEXT STEPS:")
            print(
                "  1. Regenerate golden candidates: "
                "python scripts/regenerate_golden_candidates.py"
            )
            print(
                "  2. Re-run concept extraction: "
                "python scripts/extract_concepts.py"
            )
            print("  3. Sync KuzuDB: python scripts/sync_kuzu.py")
            print(
                "  4. Validate retrieval: "
                "python scripts/eval_retrieval.py --per-domain"
            )


if __name__ == "__main__":
    asyncio.run(main())
