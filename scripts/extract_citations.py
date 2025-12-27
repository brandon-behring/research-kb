#!/usr/bin/env python3
"""Extract citations from all sources using GROBID.

This script:
1. Lists all sources (papers + textbooks) in the database
2. For each source without citations, processes the PDF with GROBID
3. Stores extracted citations via CitationStore.batch_create()
4. Reports progress and statistics by source type

Usage:
    python scripts/extract_citations.py [--dry-run] [--limit N] [--type paper|textbook]
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_contracts import Source, SourceType
from research_kb_pdf import GrobidClient
from research_kb_storage import CitationStore, DatabaseConfig, get_connection_pool

logger = get_logger(__name__)


async def get_all_sources(
    source_type: Optional[SourceType] = None,
    limit: Optional[int] = None,
) -> list[Source]:
    """Get all sources from database, optionally filtered by type."""
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        import json
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

        if source_type:
            query = """
                SELECT * FROM sources
                WHERE source_type = $1
                ORDER BY title
            """
            if limit:
                query += f" LIMIT {limit}"
            rows = await conn.fetch(query, source_type.value)
        else:
            query = "SELECT * FROM sources ORDER BY title"
            if limit:
                query += f" LIMIT {limit}"
            rows = await conn.fetch(query)

        from datetime import datetime, timezone
        sources = []
        for row in rows:
            sources.append(Source(
                id=row["id"],
                source_type=SourceType(row["source_type"]),
                title=row["title"],
                authors=row["authors"] or [],
                year=row["year"],
                file_path=row["file_path"],
                file_hash=row["file_hash"],
                metadata=row["metadata"] or {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            ))

        return sources


async def extract_citations_for_source(
    source: Source,
    grobid_client: GrobidClient,
    dry_run: bool = False,
) -> dict:
    """Extract citations from a single source via GROBID.

    Returns:
        Dict with keys: source_id, source_type, title, citations_count, skipped, error
    """
    result = {
        "source_id": str(source.id),
        "source_type": source.source_type.value,
        "title": source.title,
        "citations_count": 0,
        "skipped": False,
        "error": None,
    }

    # Check if already extracted
    existing_count = await CitationStore.count_by_source(source.id)
    if existing_count > 0:
        result["skipped"] = True
        result["citations_count"] = existing_count
        logger.info(
            "citations_already_extracted",
            source_id=str(source.id),
            title=source.title,
            count=existing_count,
        )
        return result

    # Check for valid file path
    if not source.file_path:
        result["error"] = "No file_path"
        logger.warning("no_file_path", source_id=str(source.id), title=source.title)
        return result

    file_path = Path(source.file_path)
    if not file_path.exists():
        result["error"] = f"File not found: {file_path}"
        logger.warning("file_not_found", source_id=str(source.id), path=str(file_path))
        return result

    if dry_run:
        result["skipped"] = True
        logger.info("dry_run_skip", source_id=str(source.id), title=source.title)
        return result

    # Process with GROBID
    try:
        logger.info(
            "extracting_citations",
            source_id=str(source.id),
            source_type=source.source_type.value,
            title=source.title,
        )

        paper = grobid_client.process_pdf(str(file_path))

        if not paper.citations:
            logger.info(
                "no_citations_found",
                source_id=str(source.id),
                title=source.title,
            )
            return result

        # Prepare citation data for batch insert
        citations_data = []
        for citation in paper.citations:
            citations_data.append({
                "source_id": source.id,
                "raw_string": citation.raw_string,
                "authors": citation.authors,
                "title": citation.title,
                "year": citation.year,
                "venue": citation.venue,
                "doi": citation.doi,
                "arxiv_id": citation.arxiv_id,
                "context": citation.context,  # Citing sentence from body text
                "extraction_method": "grobid",
                "metadata": {
                    "source_type": source.source_type.value,
                },
            })

        # Store citations
        await CitationStore.batch_create(citations_data)
        result["citations_count"] = len(citations_data)

        logger.info(
            "citations_stored",
            source_id=str(source.id),
            title=source.title,
            count=len(citations_data),
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error(
            "citation_extraction_failed",
            source_id=str(source.id),
            title=source.title,
            error=str(e),
        )

    return result


async def main():
    """Extract citations from all sources."""
    parser = argparse.ArgumentParser(description="Extract citations via GROBID")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually extract")
    parser.add_argument("--limit", type=int, help="Limit number of sources")
    parser.add_argument("--type", choices=["paper", "textbook"], help="Filter by source type")
    args = parser.parse_args()

    # Initialize database
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Initialize GROBID client
    grobid_client = GrobidClient()

    if not grobid_client.is_alive():
        print("ERROR: GROBID service not available at http://localhost:8070")
        print("Start with: docker-compose up grobid")
        return

    print("GROBID service is alive")

    # Get sources
    source_type = None
    if args.type:
        source_type = SourceType.PAPER if args.type == "paper" else SourceType.TEXTBOOK

    sources = await get_all_sources(source_type=source_type, limit=args.limit)
    print(f"\nFound {len(sources)} sources to process")

    # Count by type
    papers = [s for s in sources if s.source_type == SourceType.PAPER]
    textbooks = [s for s in sources if s.source_type == SourceType.TEXTBOOK]
    print(f"  Papers: {len(papers)}")
    print(f"  Textbooks: {len(textbooks)}")

    if args.dry_run:
        print("\n[DRY RUN - no citations will be extracted]")

    # Process each source
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "total_citations": 0,
        "by_type": {
            "paper": {"processed": 0, "citations": 0},
            "textbook": {"processed": 0, "citations": 0},
        },
    }

    for i, source in enumerate(sources, 1):
        print(f"\n[{i}/{len(sources)}] {source.source_type.value}: {source.title[:60]}...")

        result = await extract_citations_for_source(
            source=source,
            grobid_client=grobid_client,
            dry_run=args.dry_run,
        )

        if result["error"]:
            stats["errors"] += 1
            print(f"  ERROR: {result['error'][:80]}")
        elif result["skipped"]:
            stats["skipped"] += 1
            if result["citations_count"] > 0:
                print(f"  Skipped (already has {result['citations_count']} citations)")
            else:
                print("  Skipped")
        else:
            stats["processed"] += 1
            stats["total_citations"] += result["citations_count"]
            stats["by_type"][result["source_type"]]["processed"] += 1
            stats["by_type"][result["source_type"]]["citations"] += result["citations_count"]
            print(f"  Extracted {result['citations_count']} citations")

    # Summary
    print("\n" + "=" * 70)
    print("CITATION EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total new citations: {stats['total_citations']}")
    print("\nBy source type:")
    print(f"  Papers: {stats['by_type']['paper']['processed']} processed, {stats['by_type']['paper']['citations']} citations")
    print(f"  Textbooks: {stats['by_type']['textbook']['processed']} processed, {stats['by_type']['textbook']['citations']} citations")

    # Get final count
    pool = await get_connection_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM citations")
        print(f"\nTotal citations in database: {total}")


if __name__ == "__main__":
    asyncio.run(main())
