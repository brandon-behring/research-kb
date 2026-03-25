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
    python scripts/rechunk_corpus.py --no-embed             # Extract only, NULL embeddings
    python scripts/rechunk_corpus.py --force                # Re-rechunk ALL sources (incl. already Docling)

Two-phase GPU workflow (when Docling + embed_server can't share VRAM):
    1. Kill embed_server
    2. python scripts/rechunk_corpus.py --no-embed --skip-backup --json > rechunk_report.json
    3. Restart embed_server
    4. python scripts/backfill_embeddings.py --json > backfill_report.json

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
from research_kb_pdf import EmbeddingClient, extract_and_chunk, reset_converter
from research_kb_storage import DatabaseConfig, get_connection_pool

logger = get_logger(__name__)


async def get_sources_to_rechunk(
    pool,
    domain: str | None,
    source_id: str | None,
    limit: int | None,
    force: bool = False,
):
    """Fetch sources that need rechunking.

    By default, skips sources where ALL chunks already have
    metadata.chunking_method = 'docling'. Use force=True to rechunk everything.

    Args:
        pool: asyncpg connection pool
        domain: Optional domain_id filter
        source_id: Optional single source UUID
        limit: Optional max number of sources
        force: If True, include already-rechunked sources

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

        # Base query with optional HAVING filter
        query = f"""
            SELECT s.id, s.title, s.file_path, s.domain_id,
                   COUNT(c.id) as chunk_count,
                   COUNT(c.id) FILTER (
                       WHERE c.metadata->>'chunking_method' = 'docling'
                   ) as docling_count
            FROM sources s
            LEFT JOIN chunks c ON c.source_id = s.id
            WHERE {where}
            GROUP BY s.id, s.title, s.file_path, s.domain_id
        """

        if not force:
            # Skip sources already fully rechunked
            query += """
            HAVING COUNT(c.id) = 0
                OR COUNT(c.id) FILTER (
                    WHERE c.metadata->>'chunking_method' = 'docling'
                ) < COUNT(c.id)
            """

        query += " ORDER BY s.title"

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
    pool,
    source: dict,
    embed_client: EmbeddingClient | None,
    dry_run: bool,
    quiet: bool,
    no_embed: bool = False,
):
    """Re-chunk a single source: extract -> chunk -> delete old -> embed -> insert.

    Runs in a single transaction per source. Uses set_type_codec for JSONB
    and register_vector for pgvector column compatibility.

    Args:
        pool: asyncpg connection pool
        source: Source dict with id, title, file_path, domain_id, old_chunk_count
        embed_client: Embedding client for generating vectors (None if no_embed)
        dry_run: If True, only report what would change
        quiet: Minimal output
        no_embed: If True, insert chunks with NULL embeddings (two-phase workflow)

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
        # Extract and chunk with Docling (singleton converter — no VRAM leak)
        try:
            extraction_result, new_chunks = extract_and_chunk(str(pdf_path), max_tokens=300)
        except (RuntimeError, ValueError) as e:
            if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                logger.warning("gpu_oom_fallback_cpu", title=title)
                import gc
                import os

                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                # Reset singleton so it reinitializes on CPU
                reset_converter()
                old_val = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                try:
                    extraction_result, new_chunks = extract_and_chunk(str(pdf_path), max_tokens=300)
                finally:
                    # Restore GPU visibility for next source
                    if old_val is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = old_val
                    reset_converter()  # Force GPU converter reload for next source
            else:
                raise

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

        # Sanitize null bytes defensively (PostgreSQL rejects \x00)
        def _sanitize_strings(obj):
            """Recursively remove null bytes from all string values."""
            if isinstance(obj, dict):
                return {k: _sanitize_strings(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_sanitize_strings(v) for v in obj]
            elif isinstance(obj, str):
                return obj.replace("\x00", "").replace("\ufffd", "")
            return obj

        for chunk in new_chunks:
            chunk.content = chunk.content.replace("\x00", "").replace("\ufffd", "")
            chunk.metadata = _sanitize_strings(chunk.metadata)

        # Deduplicate chunks by content hash (HybridChunker can produce dupes)
        seen_hashes: set[str] = set()
        deduped_chunks = []
        for chunk in new_chunks:
            h = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped_chunks.append(chunk)
        if len(deduped_chunks) < len(new_chunks):
            logger.info(
                "deduped_chunks",
                source_id=str(source_id),
                original=len(new_chunks),
                deduped=len(deduped_chunks),
            )
        new_chunks = deduped_chunks
        new_count = len(new_chunks)

        # Generate embeddings (unless --no-embed for two-phase workflow)
        embeddings = None
        if not no_embed and embed_client is not None:
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
                await conn.execute("DELETE FROM chunks WHERE source_id = $1", source_id)

                # Insert new chunks with proper type handling
                for i, chunk in enumerate(new_chunks):
                    content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()

                    # Convert embedding to numpy array for pgvector, or None
                    embedding_vec = None
                    if embeddings is not None:
                        embedding_vec = np.array(embeddings[i], dtype=np.float32)

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

        # GPU memory cleanup after each source
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        elapsed = time.time() - t0
        status = "success" if not no_embed else "extracted"
        return {
            "title": title,
            "old_count": old_count,
            "new_count": new_count,
            "delta": new_count - old_count,
            "status": status,
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
    parser = argparse.ArgumentParser(description="Re-chunk corpus with structure-driven chunking.")
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
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Extract and chunk only, store with NULL embeddings. "
        "Use backfill_embeddings.py afterward to fill embeddings.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-rechunk ALL sources, including those already chunked with Docling. "
        "Use for full corpus re-processing after extractor fixes.",
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
        pool,
        domain=args.domain,
        source_id=args.source_id,
        limit=args.limit,
        force=args.force,
    )

    if not sources:
        msg = "No sources need rechunking (all already have chunking_method='docling')."
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

    # Initialize embedding client (not needed for dry run or --no-embed)
    embed_client = None
    if not args.dry_run and not args.no_embed:
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
            pool,
            source,
            embed_client,
            dry_run=args.dry_run,
            quiet=args.quiet,
            no_embed=args.no_embed,
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
            if result["status"] in ("success", "extracted"):
                embed_note = " [no-embed]" if result["status"] == "extracted" else ""
                print(
                    f"{result['old_count']} -> {result['new_count']} chunks "
                    f"(delta {result['delta']:+d}, {result['elapsed_s']:.1f}s){embed_note}"
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
    success = [r for r in results if r["status"] in ("success", "extracted")]
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
        "failed_sources": [{"title": r["title"][:50], "reason": r["reason"]} for r in failed],
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
            f"Chunks: {total_old:,} old -> {total_new:,} new " f"(delta {total_new - total_old:+,})"
        )
        print(f"Elapsed: {elapsed:.1f}s")

        if failed:
            print("\nFailed sources:")
            for r in failed:
                print(f"  - {r['title'][:50]}: {r['reason']}")

        if not args.dry_run and success:
            print()
            print("NEXT STEPS:")
            if args.no_embed:
                print("  1. Restart embed_server: " "python -m research_kb_pdf.embed_server")
                print("  2. Backfill embeddings: " "python scripts/backfill_embeddings.py --json")
            print(
                f"  {'3' if args.no_embed else '1'}. Regenerate golden candidates: "
                "python scripts/regenerate_golden_candidates.py"
            )
            print(
                f"  {'4' if args.no_embed else '2'}. Re-run concept extraction: "
                "python scripts/extract_concepts.py"
            )
            print(f"  {'5' if args.no_embed else '3'}. Sync KuzuDB: " "python scripts/sync_kuzu.py")
            print(
                f"  {'6' if args.no_embed else '4'}. Validate retrieval: "
                "python scripts/eval_retrieval.py --per-domain"
            )


if __name__ == "__main__":
    asyncio.run(main())
