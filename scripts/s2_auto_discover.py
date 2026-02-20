#!/usr/bin/env python3
"""S2 Auto-Discovery Pipeline - Main orchestrator.

Discovers new papers via Semantic Scholar API and queues them for ingestion.
Integrates with:
- s2-client package for S2 API access
- research_kb_storage for database operations
- QueueStore for async processing queue
- DiscoveryStore for audit logging

Usage:
    # Discover papers by keyword search
    python scripts/s2_auto_discover.py search "double machine learning" --domain causal_inference

    # Discover from pre-configured topics
    python scripts/s2_auto_discover.py topics --year-from 2022 --min-citations 50

    # Discover papers by author
    python scripts/s2_auto_discover.py author 26331346  # Chernozhukov

    # Discover via citation graph traversal (Phase 8.4)
    python scripts/s2_auto_discover.py traverse --seed-from-corpus --depth 1
    python scripts/s2_auto_discover.py traverse --seed-paper-id "649def34..." --direction both

    # Show queue status
    python scripts/s2_auto_discover.py queue-status

    # Show discovery statistics
    python scripts/s2_auto_discover.py stats

    # Dry run (show what would be queued)
    python scripts/s2_auto_discover.py search "DML" --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "s2-client" / "src"))

from research_kb_common import get_logger
from research_kb_storage import (
    DatabaseConfig,
    SourceStore,
    get_connection_pool,
)
from research_kb_storage.discovery_store import DiscoveryStore, DiscoveryMethod
from research_kb_storage.queue_store import QueueStore
from s2_client import (
    CitationTraversal,
    S2Client,
    S2Paper,
    TopicDiscovery,
    SearchFilters,
    DiscoveryTopic,
)

logger = get_logger(__name__)


def paper_to_queue_item(paper: S2Paper, domain_id: str, priority: int = 0) -> dict[str, Any]:
    """Convert S2Paper to queue item dict.

    Args:
        paper: S2Paper from discovery
        domain_id: Target domain
        priority: Queue priority (higher = first)

    Returns:
        Dict ready for QueueStore.add()
    """
    # Get PDF URL
    pdf_url = None
    if paper.open_access_pdf and paper.open_access_pdf.url:
        pdf_url = paper.open_access_pdf.url
    elif paper.arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"

    return {
        "s2_paper_id": paper.paper_id,
        "title": paper.title or "Unknown",
        "pdf_url": pdf_url,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "authors": [a.name for a in (paper.authors or []) if a.name],
        "year": paper.year,
        "venue": paper.venue,
        "domain_id": domain_id,
        "priority": priority,
        "metadata": {
            "citation_count": paper.citation_count,
            "influential_citation_count": paper.influential_citation_count,
            "is_open_access": paper.is_open_access,
            "fields_of_study": [f.get("category") for f in (paper.s2_fields_of_study or [])],
            "abstract": paper.abstract,
        },
    }


async def discover_by_search(
    query: str,
    domain_id: str = "causal_inference",
    year_from: int | None = None,
    min_citations: int = 50,
    open_access_only: bool = True,
    limit: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Discover papers by keyword search.

    Args:
        query: Search query
        domain_id: Target domain
        year_from: Minimum publication year
        min_citations: Minimum citation count
        open_access_only: Only include open access papers
        limit: Maximum papers to search
        dry_run: Don't actually queue papers

    Returns:
        Summary dict with results
    """
    start_time = time.time()

    # Initialize database
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Get existing identifiers for deduplication
    existing = await SourceStore.get_existing_identifiers(domain_id)
    existing_s2_ids = existing["s2_ids"]
    existing_dois = existing["dois"]
    existing_arxiv_ids = existing["arxiv_ids"]

    print(f"\n{'='*60}")
    print("S2 Discovery: Keyword Search")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"Domain: {domain_id}")
    print(f"Year from: {year_from or 'any'}")
    print(f"Min citations: {min_citations}")
    print(f"Open access only: {open_access_only}")
    print(f"Existing papers in corpus: {len(existing_s2_ids)} S2 IDs, {len(existing_dois)} DOIs")
    print(f"Dry run: {dry_run}\n")

    # Search S2
    filters = SearchFilters(
        year_from=year_from,
        min_citations=min_citations,
        open_access_only=open_access_only,
        exclude_paper_ids=existing_s2_ids,
    )

    papers_found = 0
    papers_queued = 0
    papers_skipped = 0

    async with S2Client() as client:
        discovery = TopicDiscovery(client)
        result = await discovery.discover(
            topics=[query],  # Custom query string
            filters=filters,
            limit_per_topic=limit,
        )

        papers_found = len(result.papers)
        print(f"Found {papers_found} papers matching criteria\n")

        if papers_found == 0:
            print("No new papers found.")
        else:
            # Deduplicate against existing corpus
            new_papers = []
            for paper in result.papers:
                if paper.paper_id and paper.paper_id in existing_s2_ids:
                    papers_skipped += 1
                    continue
                if paper.doi and paper.doi in existing_dois:
                    papers_skipped += 1
                    continue
                if paper.arxiv_id and paper.arxiv_id in existing_arxiv_ids:
                    papers_skipped += 1
                    continue
                new_papers.append(paper)

            print(f"After deduplication: {len(new_papers)} new papers")
            print(f"Skipped (duplicates): {papers_skipped}\n")

            if new_papers:
                # Show top papers
                print("Top papers to queue:")
                for i, paper in enumerate(new_papers[:10], 1):
                    citations = paper.citation_count or 0
                    oa = "OA" if paper.is_open_access else "closed"
                    print(f"  {i:2}. [{citations:4} cit] [{oa:6}] {paper.title[:60]}")
                    if paper.first_author_name:
                        print(f"       Author: {paper.first_author_name} ({paper.year})")
                print()

                if not dry_run:
                    # Queue papers for ingestion
                    print("Queueing papers for ingestion...")
                    queue_items = [
                        paper_to_queue_item(p, domain_id, priority=p.citation_count or 0)
                        for p in new_papers
                    ]
                    added, skipped = await QueueStore.add_batch(queue_items)
                    papers_queued = added
                    print(f"Queued: {added}, Already in queue: {skipped}")
                else:
                    papers_queued = len(new_papers)
                    print(f"Would queue {papers_queued} papers (dry run)")

    duration = time.time() - start_time

    # Log discovery
    if not dry_run:
        await DiscoveryStore.log_discovery(
            discovery_method=DiscoveryMethod.KEYWORD_SEARCH,
            query=query,
            domain_id=domain_id,
            papers_found=papers_found,
            papers_ingested=0,  # Will be updated by queue processor
            papers_skipped=papers_skipped,
            duration_seconds=duration,
            metadata={
                "year_from": year_from,
                "min_citations": min_citations,
                "open_access_only": open_access_only,
            },
        )

    print(f"\nCompleted in {duration:.1f}s")

    return {
        "query": query,
        "papers_found": papers_found,
        "papers_queued": papers_queued,
        "papers_skipped": papers_skipped,
        "duration_seconds": duration,
    }


async def discover_by_topics(
    domain_id: str = "causal_inference",
    year_from: int | None = None,
    min_citations: int = 50,
    open_access_only: bool = True,
    limit_per_topic: int = 30,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Discover papers from pre-configured topics.

    Args:
        domain_id: Target domain
        year_from: Minimum publication year
        min_citations: Minimum citation count
        open_access_only: Only include open access papers
        limit_per_topic: Max papers per topic
        dry_run: Don't actually queue papers

    Returns:
        Summary dict with results
    """
    start_time = time.time()

    # Initialize database
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Get existing identifiers
    existing = await SourceStore.get_existing_identifiers(domain_id)
    existing_s2_ids = existing["s2_ids"]
    existing_dois = existing["dois"]
    existing_arxiv_ids = existing["arxiv_ids"]

    # Select topics based on domain
    if domain_id == "causal_inference":
        topics = [
            DiscoveryTopic.DOUBLE_ML,
            DiscoveryTopic.CAUSAL_FOREST,
            DiscoveryTopic.SYNTHETIC_CONTROL,
            DiscoveryTopic.IV_METHODS,
            DiscoveryTopic.DIFF_IN_DIFF,
        ]
    else:
        # Default to all topics
        topics = list(DiscoveryTopic)

    print(f"\n{'='*60}")
    print("S2 Discovery: Topic Batch")
    print(f"{'='*60}")
    print(f"Domain: {domain_id}")
    print(f"Topics: {len(topics)}")
    print(f"Year from: {year_from or 'any'}")
    print(f"Min citations: {min_citations}")
    print(f"Limit per topic: {limit_per_topic}")
    print(f"Dry run: {dry_run}\n")

    filters = SearchFilters(
        year_from=year_from,
        min_citations=min_citations,
        open_access_only=open_access_only,
        exclude_paper_ids=existing_s2_ids,
    )

    total_found = 0
    total_queued = 0
    total_skipped = 0

    async with S2Client() as client:
        discovery = TopicDiscovery(client)
        result = await discovery.discover(
            topics=topics,
            filters=filters,
            limit_per_topic=limit_per_topic,
        )

        total_found = len(result.papers)
        print(f"Found {total_found} unique papers across {len(topics)} topics\n")

        # Deduplicate
        new_papers = []
        for paper in result.papers:
            if paper.paper_id and paper.paper_id in existing_s2_ids:
                total_skipped += 1
                continue
            if paper.doi and paper.doi in existing_dois:
                total_skipped += 1
                continue
            if paper.arxiv_id and paper.arxiv_id in existing_arxiv_ids:
                total_skipped += 1
                continue
            new_papers.append(paper)

        print(f"After deduplication: {len(new_papers)} new papers")
        print(f"Skipped (duplicates): {total_skipped}\n")

        if new_papers and not dry_run:
            queue_items = [
                paper_to_queue_item(p, domain_id, priority=p.citation_count or 0)
                for p in new_papers
            ]
            added, skipped = await QueueStore.add_batch(queue_items)
            total_queued = added
            print(f"Queued: {added}, Already in queue: {skipped}")
        elif new_papers:
            total_queued = len(new_papers)
            print(f"Would queue {total_queued} papers (dry run)")

    duration = time.time() - start_time

    if not dry_run:
        await DiscoveryStore.log_discovery(
            discovery_method=DiscoveryMethod.TOPIC_BATCH,
            query=f"{len(topics)} topics",
            domain_id=domain_id,
            papers_found=total_found,
            papers_skipped=total_skipped,
            duration_seconds=duration,
            metadata={
                "topics": [t.name for t in topics],
                "year_from": year_from,
                "min_citations": min_citations,
            },
        )

    print(f"\nCompleted in {duration:.1f}s")

    return {
        "topics": len(topics),
        "papers_found": total_found,
        "papers_queued": total_queued,
        "papers_skipped": total_skipped,
        "duration_seconds": duration,
    }


async def discover_by_author(
    author_id: str,
    domain_id: str = "causal_inference",
    min_citations: int = 20,
    open_access_only: bool = True,
    limit: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Discover papers by author.

    Args:
        author_id: S2 author ID (e.g., "26331346" for Chernozhukov)
        domain_id: Target domain
        min_citations: Minimum citation count
        open_access_only: Only include open access papers
        limit: Maximum papers to fetch
        dry_run: Don't actually queue papers

    Returns:
        Summary dict with results
    """
    start_time = time.time()

    # Initialize database
    config = DatabaseConfig()
    await get_connection_pool(config)

    existing = await SourceStore.get_existing_identifiers(domain_id)

    print(f"\n{'='*60}")
    print("S2 Discovery: Author Papers")
    print(f"{'='*60}")
    print(f"Author ID: {author_id}")
    print(f"Domain: {domain_id}")
    print(f"Min citations: {min_citations}")
    print(f"Dry run: {dry_run}\n")

    papers_found = 0
    papers_queued = 0
    papers_skipped = 0

    async with S2Client() as client:
        # Get author info
        author = await client.get_author(author_id)
        print(f"Author: {author.name}")
        print(f"h-index: {author.h_index}, Papers: {author.paper_count}\n")

        # Get author's papers
        result = await client.get_author_papers(author_id, limit=limit)
        papers = result.data

        papers_found = len(papers)
        print(f"Found {papers_found} papers\n")

        # Filter and deduplicate
        new_papers = []
        for paper in papers:
            # Skip low citation papers
            if (paper.citation_count or 0) < min_citations:
                papers_skipped += 1
                continue
            # Skip closed access if filter enabled
            if open_access_only and not paper.is_open_access:
                papers_skipped += 1
                continue
            # Skip duplicates
            if paper.paper_id and paper.paper_id in existing["s2_ids"]:
                papers_skipped += 1
                continue
            if paper.doi and paper.doi in existing["dois"]:
                papers_skipped += 1
                continue
            new_papers.append(paper)

        print(f"After filtering: {len(new_papers)} new papers")

        if new_papers:
            print("\nTop papers:")
            for i, paper in enumerate(
                sorted(new_papers, key=lambda p: p.citation_count or 0, reverse=True)[:10],
                1,
            ):
                print(f"  {i:2}. [{paper.citation_count or 0:4} cit] {paper.title[:55]}")

            if not dry_run:
                queue_items = [
                    paper_to_queue_item(p, domain_id, priority=p.citation_count or 0)
                    for p in new_papers
                ]
                added, skipped = await QueueStore.add_batch(queue_items)
                papers_queued = added
                print(f"\nQueued: {added}")
            else:
                papers_queued = len(new_papers)
                print(f"\nWould queue {papers_queued} papers (dry run)")

    duration = time.time() - start_time

    if not dry_run:
        await DiscoveryStore.log_discovery(
            discovery_method=DiscoveryMethod.AUTHOR_SEARCH,
            query=f"author:{author_id}",
            domain_id=domain_id,
            papers_found=papers_found,
            papers_skipped=papers_skipped,
            duration_seconds=duration,
            metadata={"author_name": author.name},
        )

    print(f"\nCompleted in {duration:.1f}s")

    return {
        "author_id": author_id,
        "author_name": author.name,
        "papers_found": papers_found,
        "papers_queued": papers_queued,
        "papers_skipped": papers_skipped,
    }


async def discover_by_traversal(
    seed_paper_ids: list[str] | None = None,
    seed_from_corpus: bool = False,
    domain_id: str = "causal_inference",
    depth: int = 1,
    direction: str = "both",
    min_citations: int = 30,
    open_access_only: bool = True,
    limit_per_paper: int = 50,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Discover papers by traversing citation graph from seed papers.

    Args:
        seed_paper_ids: Specific S2 paper IDs to start from
        seed_from_corpus: Use top-cited papers from corpus as seeds
        domain_id: Target domain
        depth: Traversal depth (1 = direct citations only)
        direction: "citations", "references", or "both"
        min_citations: Minimum citation count for discovered papers
        open_access_only: Only include open access papers
        limit_per_paper: Max citations/references to fetch per paper
        dry_run: Don't actually queue papers

    Returns:
        Summary dict with results
    """
    start_time = time.time()

    # Initialize database
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Get existing identifiers for deduplication
    existing = await SourceStore.get_existing_identifiers(domain_id)
    existing_s2_ids = existing["s2_ids"]
    existing_dois = existing["dois"]
    existing_arxiv_ids = existing["arxiv_ids"]

    print(f"\n{'='*60}")
    print("S2 Discovery: Citation Traversal")
    print(f"{'='*60}")
    print(f"Domain: {domain_id}")
    print(f"Direction: {direction}")
    print(f"Depth: {depth}")
    print(f"Min citations: {min_citations}")
    print(f"Open access only: {open_access_only}")
    print(f"Existing papers in corpus: {len(existing_s2_ids)} S2 IDs")
    print(f"Dry run: {dry_run}\n")

    # Determine seed papers
    seeds: list[str] = []

    if seed_paper_ids:
        seeds = list(seed_paper_ids)
        print(f"Using {len(seeds)} specified seed paper(s)")
    elif seed_from_corpus:
        # Get top-cited papers from corpus that have S2 IDs
        print("Finding top-cited papers in corpus with S2 IDs...")
        # Note: This queries the metadata JSONB for s2_paper_id
        pool = await get_connection_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT metadata->>'s2_paper_id' as s2_id
                FROM sources
                WHERE domain_id = $1
                  AND metadata->>'s2_paper_id' IS NOT NULL
                ORDER BY COALESCE((metadata->>'citation_count')::int, 0) DESC
                LIMIT 10
                """,
                domain_id,
            )
            seeds = [row["s2_id"] for row in rows if row["s2_id"]]
        print(f"Found {len(seeds)} seed papers from corpus")
    else:
        print("ERROR: Must specify --seed-paper-id or --seed-from-corpus")
        return {"error": "No seed papers specified"}

    if not seeds:
        print("No seed papers available. Cannot traverse.")
        return {"error": "No seed papers found"}

    print(f"\nSeed papers: {seeds[:5]}{'...' if len(seeds) > 5 else ''}\n")

    papers_found = 0
    papers_queued = 0
    papers_skipped = 0

    async with S2Client() as client:
        traversal = CitationTraversal(client)
        result = await traversal.discover(
            seed_paper_ids=seeds,
            depth=depth,
            direction=direction,
            min_citations=min_citations,
            open_access_only=open_access_only,
            exclude_ids=existing_s2_ids,
            limit_per_paper=limit_per_paper,
        )

        papers_found = len(result.papers)
        print("\nTraversal complete:")
        print(f"  Papers discovered: {papers_found}")
        print(f"  Total traversed: {result.total_traversed}")
        print(f"  Duplicates removed: {result.duplicates_removed}")
        print(f"  Filtered out: {result.filtered_out}")

        if papers_found == 0:
            print("\nNo new papers found.")
        else:
            # Additional deduplication against DOI/arXiv
            new_papers = []
            for paper in result.papers:
                if paper.doi and paper.doi in existing_dois:
                    papers_skipped += 1
                    continue
                if paper.arxiv_id and paper.arxiv_id in existing_arxiv_ids:
                    papers_skipped += 1
                    continue
                new_papers.append(paper)

            print(f"\nAfter DOI/arXiv dedup: {len(new_papers)} new papers")
            if papers_skipped > 0:
                print(f"Skipped (DOI/arXiv match): {papers_skipped}")

            if new_papers:
                # Show top papers
                print("\nTop discovered papers:")
                sorted_papers = sorted(
                    new_papers, key=lambda p: p.citation_count or 0, reverse=True
                )
                for i, paper in enumerate(sorted_papers[:10], 1):
                    citations = paper.citation_count or 0
                    print(f"  {i:2}. [{citations:4} cit] {(paper.title or 'Unknown')[:55]}")

                if not dry_run:
                    print("\nQueueing papers for ingestion...")
                    queue_items = [
                        paper_to_queue_item(p, domain_id, priority=p.citation_count or 0)
                        for p in new_papers
                    ]
                    added, skipped = await QueueStore.add_batch(queue_items)
                    papers_queued = added
                    print(f"Queued: {added}, Already in queue: {skipped}")
                else:
                    papers_queued = len(new_papers)
                    print(f"\nWould queue {papers_queued} papers (dry run)")

    duration = time.time() - start_time

    # Log discovery
    if not dry_run:
        await DiscoveryStore.log_discovery(
            discovery_method=DiscoveryMethod.CITATION_TRAVERSE,
            query=f"seeds:{len(seeds)},depth:{depth},dir:{direction}",
            domain_id=domain_id,
            papers_found=papers_found,
            papers_skipped=papers_skipped + result.filtered_out,
            duration_seconds=duration,
            metadata={
                "seed_count": len(seeds),
                "depth": depth,
                "direction": direction,
                "min_citations": min_citations,
                "total_traversed": result.total_traversed,
            },
        )

    print(f"\nCompleted in {duration:.1f}s")

    return {
        "seed_count": len(seeds),
        "papers_found": papers_found,
        "papers_queued": papers_queued,
        "papers_skipped": papers_skipped,
        "total_traversed": result.total_traversed,
        "duration_seconds": duration,
    }


async def show_queue_status() -> None:
    """Show current queue status."""
    config = DatabaseConfig()
    await get_connection_pool(config)

    stats = await QueueStore.get_stats()

    print(f"\n{'='*60}")
    print("Ingestion Queue Status")
    print(f"{'='*60}")
    print(f"Total items: {stats['total']}")
    print("\nBy status:")
    for status, count in stats["by_status"].items():
        print(f"  {status}: {count}")
    print("\nBy domain:")
    for domain, count in stats["by_domain"].items():
        print(f"  {domain}: {count}")
    if stats["with_retries"] > 0:
        print(f"\nItems with retries: {stats['with_retries']}")
        print(f"Max retry count: {stats['max_retries']}")


async def show_discovery_stats(days: int = 30) -> None:
    """Show discovery statistics."""
    config = DatabaseConfig()
    await get_connection_pool(config)

    stats = await DiscoveryStore.get_stats(days=days)

    print(f"\n{'='*60}")
    print(f"Discovery Statistics (last {days} days)")
    print(f"{'='*60}")
    print(f"Total discoveries: {stats['total_discoveries']}")
    print(f"Papers found: {stats['total_papers_found']}")
    print(f"Papers ingested: {stats['total_papers_ingested']}")
    print(f"Papers skipped: {stats['total_papers_skipped']}")
    print(f"Avg duration: {stats['avg_duration_seconds']:.1f}s")

    if stats["by_method"]:
        print("\nBy method:")
        for method, data in stats["by_method"].items():
            print(f"  {method}: {data['count']} runs, {data['papers_ingested']} ingested")

    # Recent discoveries
    recent = await DiscoveryStore.get_recent(limit=5)
    if recent:
        print("\nRecent discoveries:")
        for entry in recent:
            query = entry.get("query", "")[:30]
            found = entry.get("papers_found", 0)
            method = entry.get("discovery_method", "")
            print(f"  - [{method}] '{query}' → {found} found")


def main():
    parser = argparse.ArgumentParser(
        description="S2 Auto-Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Search command
    search_parser = subparsers.add_parser("search", help="Search by keyword")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--domain", default="causal_inference", help="Target domain")
    search_parser.add_argument("--year-from", type=int, help="Minimum publication year")
    search_parser.add_argument("--min-citations", type=int, default=50, help="Minimum citations")
    search_parser.add_argument("--limit", type=int, default=100, help="Max papers")
    search_parser.add_argument(
        "--include-closed", action="store_true", help="Include closed access"
    )
    search_parser.add_argument("--dry-run", action="store_true", help="Don't queue papers")

    # Topics command
    topics_parser = subparsers.add_parser("topics", help="Discover from pre-configured topics")
    topics_parser.add_argument("--domain", default="causal_inference", help="Target domain")
    topics_parser.add_argument("--year-from", type=int, help="Minimum publication year")
    topics_parser.add_argument("--min-citations", type=int, default=50, help="Minimum citations")
    topics_parser.add_argument(
        "--limit-per-topic", type=int, default=30, help="Max papers per topic"
    )
    topics_parser.add_argument(
        "--include-closed", action="store_true", help="Include closed access"
    )
    topics_parser.add_argument("--dry-run", action="store_true", help="Don't queue papers")

    # Author command
    author_parser = subparsers.add_parser("author", help="Discover by author")
    author_parser.add_argument("author_id", help="S2 author ID")
    author_parser.add_argument("--domain", default="causal_inference", help="Target domain")
    author_parser.add_argument("--min-citations", type=int, default=20, help="Minimum citations")
    author_parser.add_argument("--limit", type=int, default=100, help="Max papers")
    author_parser.add_argument(
        "--include-closed", action="store_true", help="Include closed access"
    )
    author_parser.add_argument("--dry-run", action="store_true", help="Don't queue papers")

    # Traverse command
    traverse_parser = subparsers.add_parser("traverse", help="Traverse citation graph")
    traverse_parser.add_argument(
        "--seed-paper-id",
        action="append",
        dest="seed_paper_ids",
        help="Specific paper ID(s) to start from (can repeat)",
    )
    traverse_parser.add_argument(
        "--seed-from-corpus",
        action="store_true",
        help="Use top-cited papers from corpus as seeds",
    )
    traverse_parser.add_argument("--domain", default="causal_inference", help="Target domain")
    traverse_parser.add_argument(
        "--depth", type=int, default=1, help="Traversal depth (1-2 recommended)"
    )
    traverse_parser.add_argument(
        "--direction",
        choices=["citations", "references", "both"],
        default="both",
        help="Traversal direction",
    )
    traverse_parser.add_argument("--min-citations", type=int, default=30, help="Minimum citations")
    traverse_parser.add_argument("--limit-per-paper", type=int, default=50, help="Max per paper")
    traverse_parser.add_argument(
        "--include-closed", action="store_true", help="Include closed access"
    )
    traverse_parser.add_argument("--dry-run", action="store_true", help="Don't queue papers")

    # Queue status command
    subparsers.add_parser("queue-status", help="Show queue status")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show discovery statistics")
    stats_parser.add_argument("--days", type=int, default=30, help="Look back period")

    args = parser.parse_args()

    if args.command == "search":
        asyncio.run(
            discover_by_search(
                query=args.query,
                domain_id=args.domain,
                year_from=args.year_from,
                min_citations=args.min_citations,
                open_access_only=not args.include_closed,
                limit=args.limit,
                dry_run=args.dry_run,
            )
        )
    elif args.command == "topics":
        asyncio.run(
            discover_by_topics(
                domain_id=args.domain,
                year_from=args.year_from,
                min_citations=args.min_citations,
                open_access_only=not args.include_closed,
                limit_per_topic=args.limit_per_topic,
                dry_run=args.dry_run,
            )
        )
    elif args.command == "author":
        asyncio.run(
            discover_by_author(
                author_id=args.author_id,
                domain_id=args.domain,
                min_citations=args.min_citations,
                open_access_only=not args.include_closed,
                limit=args.limit,
                dry_run=args.dry_run,
            )
        )
    elif args.command == "traverse":
        asyncio.run(
            discover_by_traversal(
                seed_paper_ids=args.seed_paper_ids,
                seed_from_corpus=args.seed_from_corpus,
                domain_id=args.domain,
                depth=args.depth,
                direction=args.direction,
                min_citations=args.min_citations,
                open_access_only=not args.include_closed,
                limit_per_paper=args.limit_per_paper,
                dry_run=args.dry_run,
            )
        )
    elif args.command == "queue-status":
        asyncio.run(show_queue_status())
    elif args.command == "stats":
        asyncio.run(show_discovery_stats(days=args.days))


if __name__ == "__main__":
    main()
