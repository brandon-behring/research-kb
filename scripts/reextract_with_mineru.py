#!/usr/bin/env python3
"""Re-extract sources with MinerU for LaTeX-quality formula extraction.

Replaces Docling-extracted chunks with MinerU-extracted chunks for specified
domains. MinerU produces publication-quality LaTeX formulas ($$...$$ blocks)
instead of Docling's `<!-- formula-not-decoded -->` gaps.

Per-source transactions: one failure doesn't block others. Failed PDFs
logged to DLQ for retry.

Usage:
    # Dry-run on 1 source
    python scripts/reextract_with_mineru.py --domains linear_algebra --limit 1 --dry-run

    # Single source by UUID
    python scripts/reextract_with_mineru.py --source-id <uuid> --no-embed

    # Full Tier A extraction (4 domains, 102 sources)
    python scripts/reextract_with_mineru.py \
      --domains linear_algebra,topology_geometry,probability_theory,analysis \
      --no-embed

    # Two-phase GPU workflow:
    #   1. Kill embed_server + rerank_server + Ollama
    #   2. python scripts/reextract_with_mineru.py --domains ... --no-embed
    #   3. Restart services
    #   4. python scripts/backfill_embeddings.py --batch-size 8
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_pdf.mineru_extractor import (
    MinerUExtractionResult,
    check_vram_available,
    extract_and_chunk,
)
from research_kb_storage import DatabaseConfig, get_connection_pool

logger = get_logger(__name__)

DLQ_PATH = Path(__file__).parent.parent / "data" / "dlq" / "mineru_failed.jsonl"


# ── Database Queries ─────────────────────────────────────────────────────────


async def get_sources_for_domains(
    pool,
    domains: list[str],
    limit: Optional[int] = None,
    source_id: Optional[str] = None,
) -> list[dict]:
    """Query sources to re-extract.

    Parameters
    ----------
    pool : asyncpg.Pool
        Connection pool.
    domains : list[str]
        Domain IDs to filter by.
    limit : int or None
        Max sources to return.
    source_id : str or None
        Specific source UUID (overrides domain filter).

    Returns
    -------
    list[dict]
        Source dicts with id, title, file_path, domain_id, chunk_count.
    """
    async with pool.acquire() as conn:
        if source_id:
            rows = await conn.fetch(
                """
                SELECT s.id, s.title, s.file_path, s.domain_id,
                       count(c.id) as chunk_count,
                       count(c.id) FILTER (
                           WHERE c.content LIKE '%formula-not-decoded%'
                       ) as gap_count
                FROM sources s
                LEFT JOIN chunks c ON c.source_id = s.id
                WHERE s.id = $1
                GROUP BY s.id, s.title, s.file_path, s.domain_id
                """,
                UUID(source_id),
            )
        else:
            query = """
                SELECT s.id, s.title, s.file_path, s.domain_id,
                       count(c.id) as chunk_count,
                       count(c.id) FILTER (
                           WHERE c.content LIKE '%formula-not-decoded%'
                       ) as gap_count
                FROM sources s
                LEFT JOIN chunks c ON c.source_id = s.id
                WHERE s.domain_id = ANY($1)
                GROUP BY s.id, s.title, s.file_path, s.domain_id
                ORDER BY gap_count DESC, s.title
            """
            params: list = [domains]
            if limit:
                query += f" LIMIT $2"
                params.append(limit)
            rows = await conn.fetch(query, *params)

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "file_path": row["file_path"],
            "domain_id": row["domain_id"],
            "chunk_count": row["chunk_count"],
            "gap_count": row["gap_count"],
        }
        for row in rows
    ]


async def reextract_source(
    pool,
    source: dict,
    dry_run: bool,
    no_embed: bool = True,
    venv_path: str = "~/.local/share/mineru-venv",
) -> dict:
    """Re-extract a single source with MinerU.

    Flow: extract → chunk → dedup → transaction(delete old → insert new) → update source metadata.

    Parameters
    ----------
    pool : asyncpg.Pool
        Connection pool.
    source : dict
        Source with id, title, file_path, domain_id, chunk_count, gap_count.
    dry_run : bool
        If True, extract and chunk but don't modify DB.
    no_embed : bool
        If True, insert with NULL embeddings (two-phase workflow).
    venv_path : str
        Path to MinerU Python 3.12 venv.

    Returns
    -------
    dict
        Result with title, old_count, new_count, old_gaps, new_gaps, status, elapsed_s.
    """
    from pgvector.asyncpg import register_vector

    source_id = source["id"]
    file_path = source["file_path"]
    title = source["title"]
    old_count = source["chunk_count"]
    old_gaps = source["gap_count"]
    t0 = time.time()

    # Verify file exists
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return {
            "title": title,
            "source_id": str(source_id),
            "old_count": old_count,
            "new_count": 0,
            "old_gaps": old_gaps,
            "new_gaps": 0,
            "status": "skipped",
            "reason": f"PDF not found: {file_path}",
            "elapsed_s": 0.0,
        }

    # VRAM check
    sufficient, free_mb = check_vram_available()
    if not sufficient:
        return {
            "title": title,
            "source_id": str(source_id),
            "old_count": old_count,
            "new_count": 0,
            "old_gaps": old_gaps,
            "new_gaps": 0,
            "status": "skipped",
            "reason": f"Insufficient VRAM: {free_mb} MB free",
            "elapsed_s": 0.0,
        }

    try:
        # Extract with MinerU
        extraction_result, new_chunks = extract_and_chunk(
            str(pdf_path),
            max_tokens=300,
            venv_path=venv_path,
            check_vram=False,  # Already checked above
        )
    except Exception as e:
        elapsed = time.time() - t0
        error_msg = str(e)[:300]
        logger.error(
            "mineru_extraction_failed",
            title=title,
            source_id=str(source_id),
            error=error_msg,
        )
        return {
            "title": title,
            "source_id": str(source_id),
            "old_count": old_count,
            "new_count": 0,
            "old_gaps": old_gaps,
            "new_gaps": 0,
            "status": "failed",
            "reason": error_msg,
            "elapsed_s": elapsed,
        }

    new_count = len(new_chunks)

    # Count formula gaps in new chunks
    new_gaps = sum(1 for c in new_chunks if "formula-not-decoded" in c.content)

    if dry_run:
        return {
            "title": title,
            "source_id": str(source_id),
            "old_count": old_count,
            "new_count": new_count,
            "old_gaps": old_gaps,
            "new_gaps": new_gaps,
            "status": "dry_run",
            "reason": "",
            "elapsed_s": time.time() - t0,
        }

    # Sanitize null bytes
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        elif isinstance(obj, str):
            return obj.replace("\x00", "").replace("\ufffd", "")
        return obj

    for chunk in new_chunks:
        chunk.content = chunk.content.replace("\x00", "").replace("\ufffd", "")
        chunk.metadata = _sanitize(chunk.metadata)

    # Deduplicate by content hash
    seen_hashes: set[str] = set()
    deduped: list = []
    for chunk in new_chunks:
        h = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(chunk)
    if len(deduped) < len(new_chunks):
        logger.info(
            "deduped_chunks",
            source_id=str(source_id),
            original=len(new_chunks),
            deduped=len(deduped),
        )
    new_chunks = deduped
    new_count = len(new_chunks)

    # Transaction: delete old → insert new
    async with pool.acquire() as conn:
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )
        await register_vector(conn)

        async with conn.transaction():
            # Delete old chunks
            await conn.execute("DELETE FROM chunks WHERE source_id = $1", source_id)

            # Insert new chunks
            for i, chunk in enumerate(new_chunks):
                content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()

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
                    None,  # Embeddings backfilled separately
                    chunk.metadata,
                    source["domain_id"],
                )

            # Update source metadata to record extraction method.
            # Pass the dict directly: the jsonb codec registered above
            # (encoder=json.dumps) runs json.dumps for us. Wrapping the
            # dict in json.dumps() here and casting to ::jsonb causes
            # double-encoding — the codec re-encodes the str, producing
            # a JSONB string scalar, and `metadata || <string_scalar>`
            # then coerces both sides to [dict, "str"] (issue #8 §C).
            # Same pattern as SourceStore.update_metadata.
            await conn.execute(
                "UPDATE sources SET metadata = metadata || $1 WHERE id = $2",
                {
                    "extraction_method": "mineru",
                    "mineru_extracted_at": datetime.now(timezone.utc).isoformat(),
                    "total_pages": extraction_result.total_pages,
                    "total_chars": extraction_result.total_chars,
                    "total_chunks": new_count,
                    "has_equations": extraction_result.has_equations,
                    "heading_count": extraction_result.heading_count,
                },
                source_id,
            )

    elapsed = time.time() - t0
    logger.info(
        "source_reextracted",
        title=title,
        source_id=str(source_id),
        old_count=old_count,
        new_count=new_count,
        old_gaps=old_gaps,
        new_gaps=new_gaps,
        elapsed_s=round(elapsed, 1),
    )

    return {
        "title": title,
        "source_id": str(source_id),
        "old_count": old_count,
        "new_count": new_count,
        "old_gaps": old_gaps,
        "new_gaps": new_gaps,
        "status": "ok",
        "reason": "",
        "elapsed_s": round(elapsed, 1),
    }


def log_to_dlq(source: dict, error: str) -> None:
    """Append failed source to dead letter queue.

    Parameters
    ----------
    source : dict
        Source info.
    error : str
        Error message.
    """
    DLQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "source_id": str(source["id"]),
        "title": source["title"],
        "file_path": source["file_path"],
        "domain_id": source["domain_id"],
        "error": error[:500],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(DLQ_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────


async def run(args: argparse.Namespace) -> None:
    """Main async entry point."""
    pool = await get_connection_pool()

    # Parse domains
    domains = [d.strip() for d in args.domains.split(",")] if args.domains else []

    if not domains and not args.source_id:
        print("ERROR: Must specify --domains or --source-id", file=sys.stderr)
        sys.exit(1)

    # Get sources
    sources = await get_sources_for_domains(
        pool, domains, limit=args.limit, source_id=args.source_id
    )

    if not sources:
        print("No sources found matching criteria.")
        return

    # Summary
    total_chunks = sum(s["chunk_count"] for s in sources)
    total_gaps = sum(s["gap_count"] for s in sources)
    print(f"MinerU Re-extraction")
    print(f"  Sources: {len(sources)}")
    print(f"  Domains: {', '.join(set(s['domain_id'] for s in sources))}")
    print(f"  Total chunks: {total_chunks:,}")
    print(f"  Total formula gaps: {total_gaps:,}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Embeddings: {'skip (--no-embed)' if args.no_embed else 'generate'}")
    print(f"  Venv: {args.venv_path}")
    print()

    if not args.dry_run and not args.skip_backup:
        print("WARNING: This will DELETE and REPLACE chunks for these sources.")
        print("Run ./scripts/backup_db.sh first!")
        print("Use --skip-backup to suppress this warning.")
        print()

    # VRAM check
    sufficient, free_mb = check_vram_available()
    if not sufficient:
        print(f"ERROR: Insufficient VRAM ({free_mb} MB free, need 4000 MB).")
        print("Stop embed_server, rerank_server, and Ollama first.")
        sys.exit(1)
    print(f"  VRAM: {free_mb} MB free (OK)")
    print()

    # Process sources
    results: list[dict] = []
    ok = 0
    failed = 0
    skipped = 0
    t_start = time.time()

    for i, source in enumerate(sources):
        title = source["title"][:60]
        gaps = source["gap_count"]
        print(f"  [{i+1}/{len(sources)}] {title} ({gaps} gaps)...", end=" ", flush=True)

        result = await reextract_source(
            pool=pool,
            source=source,
            dry_run=args.dry_run,
            no_embed=args.no_embed,
            venv_path=args.venv_path,
        )
        results.append(result)

        status = result["status"]
        if status == "ok" or status == "dry_run":
            ok += 1
            delta = result["new_count"] - result["old_count"]
            gap_delta = result["new_gaps"] - result["old_gaps"]
            print(
                f"{result['new_count']} chunks ({delta:+d}), "
                f"{result['new_gaps']} gaps ({gap_delta:+d}), "
                f"{result['elapsed_s']:.0f}s"
            )
        elif status == "skipped":
            skipped += 1
            print(f"SKIPPED: {result['reason']}")
        elif status == "failed":
            failed += 1
            print(f"FAILED: {result['reason'][:80]}")
            log_to_dlq(source, result["reason"])

    elapsed_total = time.time() - t_start

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Processed: {len(sources)}")
    print(f"  OK: {ok}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    if results:
        old_total = sum(r["old_count"] for r in results if r["status"] in ("ok", "dry_run"))
        new_total = sum(r["new_count"] for r in results if r["status"] in ("ok", "dry_run"))
        old_gaps_total = sum(r["old_gaps"] for r in results if r["status"] in ("ok", "dry_run"))
        new_gaps_total = sum(r["new_gaps"] for r in results if r["status"] in ("ok", "dry_run"))
        print(f"  Chunks: {old_total:,} → {new_total:,} ({new_total - old_total:+,})")
        print(
            f"  Gaps: {old_gaps_total:,} → {new_gaps_total:,} ({new_gaps_total - old_gaps_total:+,})"
        )

    if failed > 0:
        print(f"\n  Failed sources logged to: {DLQ_PATH}")

    if not args.dry_run and ok > 0:
        print(f"\n  Next steps:")
        print(f"    1. Backfill embeddings: python scripts/backfill_embeddings.py --batch-size 8")
        print(
            f"    2. Re-pool eval: python scripts/eval_pool_candidates.py --queries fixtures/eval/v2_queries.yaml"
        )

    # Save report
    if args.json_output:
        report = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "mode": "dry_run" if args.dry_run else "live",
            "domains": domains,
            "sources_processed": len(sources),
            "ok": ok,
            "failed": failed,
            "skipped": skipped,
            "elapsed_seconds": round(elapsed_total, 1),
            "results": results,
        }
        report_path = Path(args.json_output)
        report_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  Report saved: {report_path}")

    await pool.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Re-extract sources with MinerU for LaTeX formula quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="",
        help="Comma-separated domain IDs (e.g., linear_algebra,probability_theory)",
    )
    parser.add_argument(
        "--source-id",
        type=str,
        default=None,
        help="Specific source UUID to re-extract",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max sources to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk but don't modify DB",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        default=True,
        help="Skip embedding generation (default: True, use backfill_embeddings.py)",
    )
    parser.add_argument(
        "--with-embed",
        action="store_true",
        help="Generate embeddings inline (requires embed_server)",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip backup reminder",
    )
    parser.add_argument(
        "--venv-path",
        type=str,
        default=str(Path("~/.local/share/mineru-venv").expanduser()),
        help="Path to MinerU Python 3.12 venv",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Save JSON report to path",
    )
    args = parser.parse_args()

    if args.with_embed:
        args.no_embed = False

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
