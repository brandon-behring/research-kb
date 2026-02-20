#!/usr/bin/env python3
"""Repair sources with 0 chunks by re-running embedding.

This script fixes sources that were created but didn't complete
chunking/embedding due to server errors.
"""

import asyncio
import hashlib
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_pdf import (
    EmbeddingClient,
    chunk_with_sections,
    extract_with_headings,
)
from research_kb_storage import ChunkStore, DatabaseConfig, get_connection_pool

logger = get_logger(__name__)


async def get_orphan_sources(pool):
    """Get sources that have 0 chunks."""
    return await pool.fetch(
        """
        SELECT s.id, s.title, s.file_path
        FROM sources s
        WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.source_id = s.id)
          AND s.file_path IS NOT NULL
        ORDER BY s.title
    """
    )


async def repair_source(source_id: str, file_path: str, title: str, pool):
    """Re-run chunking and embedding for a source."""
    logger.info("repairing_source", source_id=source_id, title=title[:50])

    if not Path(file_path).exists():
        logger.error("file_not_found", file_path=file_path)
        return 0

    # Extract text and headings
    try:
        doc, headings = extract_with_headings(file_path)
    except Exception as e:
        logger.error("extraction_failed", file_path=file_path, error=str(e))
        return 0

    # Chunk the document
    chunks = chunk_with_sections(doc, headings, target_tokens=300)

    if not chunks:
        logger.warning("no_chunks_created", file_path=file_path)
        return 0

    logger.info("chunking_complete", chunks=len(chunks))

    # Generate embeddings and store chunks
    embedding_client = EmbeddingClient()
    chunks_created = 0

    for i, chunk in enumerate(chunks):
        try:
            embedding = embedding_client.embed(chunk.content)

            # Compute content hash for deduplication
            content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()

            await ChunkStore.create(
                source_id=source_id,
                content=chunk.content,
                content_hash=content_hash,
                page_start=chunk.start_page,
                page_end=chunk.end_page,
                embedding=embedding,
                metadata={
                    "section_header": chunk.metadata.get("section", ""),
                    "chunk_index": i,
                },
            )
            chunks_created += 1

            if (i + 1) % 50 == 0:
                logger.info("chunks_progress", current=i + 1, total=len(chunks))

        except Exception as e:
            logger.error("chunk_failed", chunk_index=i, error=str(e))
            raise  # Fail fast on embedding errors

    logger.info("repair_complete", source_id=source_id, chunks=chunks_created)
    return chunks_created


async def main():
    """Repair all orphan sources."""
    import argparse

    parser = argparse.ArgumentParser(description="Repair sources with 0 chunks")
    parser.add_argument("--limit", type=int, default=100, help="Max sources to repair")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    orphans = await get_orphan_sources(pool)
    print(f"Found {len(orphans)} sources with 0 chunks")

    if args.dry_run:
        print("\nSources that would be repaired:")
        for s in orphans[: args.limit]:
            exists = "YES" if Path(s["file_path"] or "").exists() else "NO"
            print(f"  [{exists}] {s['title'][:60]}")
        return

    to_repair = [s for s in orphans if s["file_path"] and Path(s["file_path"]).exists()]
    print(f"Sources with existing files: {len(to_repair)}")
    print(f"Will repair up to: {args.limit}")

    results = {"success": 0, "failed": 0, "total_chunks": 0}

    for i, source in enumerate(to_repair[: args.limit]):
        print(f"\n[{i+1}/{min(len(to_repair), args.limit)}] {source['title'][:50]}...")

        try:
            chunks = await repair_source(
                str(source["id"]),
                source["file_path"],
                source["title"],
                pool,
            )
            results["success"] += 1
            results["total_chunks"] += chunks
            print(f"  ✓ {chunks} chunks created")

        except Exception as e:
            logger.error("repair_failed", title=source["title"], error=str(e))
            results["failed"] += 1
            print(f"  ✗ Failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("REPAIR SUMMARY")
    print("=" * 70)
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Total new chunks: {results['total_chunks']}")


if __name__ == "__main__":
    asyncio.run(main())
