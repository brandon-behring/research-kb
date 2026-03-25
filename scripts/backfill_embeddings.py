#!/usr/bin/env python3
"""Backfill NULL embeddings for chunks after --no-embed rechunking.

Phase 2 of the two-phase GPU workflow:
    1. rechunk_corpus.py --no-embed  (Docling on GPU, no embed_server needed)
    2. backfill_embeddings.py        (embed_server on GPU, Docling not loaded)

Processes chunks in batches per source. Skips chunks that already have embeddings.

Usage:
    python scripts/backfill_embeddings.py                    # Backfill all
    python scripts/backfill_embeddings.py --domain finance   # One domain
    python scripts/backfill_embeddings.py --limit 10         # First 10 sources
    python scripts/backfill_embeddings.py --batch-size 64    # Larger batches
    python scripts/backfill_embeddings.py --json             # JSON output
    python scripts/backfill_embeddings.py --dry-run          # Preview scope
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_pdf import EmbeddingClient
from research_kb_storage import DatabaseConfig, get_connection_pool

logger = get_logger(__name__)


async def get_sources_needing_embeddings(pool, domain: str | None, limit: int | None) -> list[dict]:
    """Find sources with chunks that have NULL embeddings.

    Args:
        pool: asyncpg connection pool
        domain: Optional domain_id filter
        limit: Optional max number of sources

    Returns:
        List of dicts with id, title, domain_id, null_count
    """
    async with pool.acquire() as conn:
        conditions = ["c.embedding IS NULL"]
        params: list = []

        if domain:
            params.append(domain)
            conditions.append(f"s.domain_id = ${len(params)}")

        where = " AND ".join(conditions)

        query = f"""
            SELECT s.id, s.title, s.domain_id,
                   COUNT(c.id) as null_count,
                   COUNT(c.id) FILTER (WHERE c.embedding IS NOT NULL) as has_count
            FROM sources s
            JOIN chunks c ON c.source_id = s.id
            WHERE {where}
            GROUP BY s.id, s.title, s.domain_id
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
            "domain_id": row["domain_id"],
            "null_count": row["null_count"],
            "has_count": row["has_count"],
        }
        for row in rows
    ]


async def backfill_source(
    pool, source: dict, embed_client: EmbeddingClient, batch_size: int
) -> dict:
    """Backfill embeddings for one source's NULL-embedding chunks.

    Args:
        pool: asyncpg connection pool
        source: Source dict with id, title, null_count
        embed_client: Embedding client
        batch_size: Number of chunks per embedding batch

    Returns:
        Dict with results: title, filled, elapsed_s, status
    """
    from pgvector.asyncpg import register_vector

    source_id = source["id"]
    t0 = time.time()

    try:
        async with pool.acquire() as conn:
            await conn.set_type_codec(
                "jsonb",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
            )
            await register_vector(conn)

            # Fetch chunks needing embeddings
            rows = await conn.fetch(
                "SELECT id, content FROM chunks WHERE source_id = $1 AND embedding IS NULL ORDER BY page_start",
                source_id,
            )

            if not rows:
                return {
                    "title": source["title"],
                    "filled": 0,
                    "status": "skipped",
                    "reason": "no NULL embeddings",
                    "elapsed_s": 0.0,
                }

            # Process in batches
            filled = 0
            for batch_start in range(0, len(rows), batch_size):
                batch = rows[batch_start : batch_start + batch_size]
                texts = [r["content"] for r in batch]
                chunk_ids = [r["id"] for r in batch]

                embeddings = embed_client.embed_batch(texts, batch_size=batch_size)

                for chunk_id, embedding in zip(chunk_ids, embeddings):
                    embedding_vec = np.array(embedding, dtype=np.float32)
                    await conn.execute(
                        "UPDATE chunks SET embedding = $1 WHERE id = $2",
                        embedding_vec,
                        chunk_id,
                    )
                    filled += 1

        elapsed = time.time() - t0
        return {
            "title": source["title"],
            "filled": filled,
            "status": "success",
            "reason": "",
            "elapsed_s": elapsed,
        }

    except Exception as e:
        logger.error(
            "backfill_failed",
            source_id=str(source_id),
            title=source["title"],
            error=str(e),
        )
        return {
            "title": source["title"],
            "filled": 0,
            "status": "failed",
            "reason": str(e)[:200],
            "elapsed_s": time.time() - t0,
        }


def format_eta(elapsed: float, completed: int, total: int) -> str:
    """Calculate and format ETA string."""
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill NULL embeddings after --no-embed rechunking."
    )
    parser.add_argument(
        "--domain", type=str, default=None, help="Only backfill sources in this domain"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of sources to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Chunks per embedding batch (default: 32)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show scope without backfilling",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="JSON output for programmatic monitoring",
    )
    return parser.parse_args()


async def main():
    """Backfill NULL embeddings for rechunked sources."""
    from research_kb_common import configure_logging

    args = parse_args()

    if args.quiet or args.json_output:
        configure_logging(level="ERROR")

    pool = await get_connection_pool(DatabaseConfig())

    sources = await get_sources_needing_embeddings(pool, domain=args.domain, limit=args.limit)

    if not sources:
        msg = "No chunks with NULL embeddings found."
        if args.json_output:
            print(json.dumps({"status": "no_work", "message": msg}))
        else:
            print(msg)
        return

    total_null = sum(s["null_count"] for s in sources)

    if args.dry_run:
        if args.json_output:
            print(
                json.dumps(
                    {
                        "mode": "dry_run",
                        "sources": len(sources),
                        "chunks_to_embed": total_null,
                    }
                )
            )
        else:
            print(f"DRY RUN: {len(sources)} sources, {total_null:,} chunks need embeddings")
            for s in sources[:20]:
                print(f"  [{s['domain_id']}] {s['title'][:55]} — {s['null_count']} chunks")
            if len(sources) > 20:
                print(f"  ... and {len(sources) - 20} more")
        return

    if not args.quiet and not args.json_output:
        print(f"\nBACKFILL EMBEDDINGS: {len(sources)} sources, {total_null:,} chunks")
        print(f"  Batch size: {args.batch_size}")
        print()

    embed_client = EmbeddingClient()

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

        result = await backfill_source(pool, source, embed_client, batch_size=args.batch_size)
        results.append(result)

        if args.json_output:
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
                print(f"{result['filled']} embedded ({result['elapsed_s']:.1f}s)")
            elif result["status"] == "skipped":
                print(f"SKIPPED: {result['reason']}")
            else:
                print(f"FAILED: {result['reason']}")

    elapsed = time.time() - start_time

    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    total_filled = sum(r["filled"] for r in results)

    summary = {
        "success_count": len(success),
        "failed_count": len(failed),
        "chunks_filled": total_filled,
        "elapsed_s": round(elapsed, 1),
        "failed_sources": [{"title": r["title"][:50], "reason": r["reason"]} for r in failed],
    }

    if args.json_output:
        print(json.dumps({"event": "summary", **summary}))
    else:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Success: {len(success)} sources")
        print(f"Failed: {len(failed)} sources")
        print(f"Chunks embedded: {total_filled:,}")
        print(f"Elapsed: {elapsed:.1f}s")

        if failed:
            print("\nFailed sources:")
            for r in failed:
                print(f"  - {r['title'][:50]}: {r['reason']}")


if __name__ == "__main__":
    asyncio.run(main())
