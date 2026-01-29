#!/usr/bin/env python3
"""Process pending items from the ingestion queue.

Downloads open-access PDFs from queue and ingests them into the corpus.
Supports batch processing, retry logic, and progress tracking.

Usage:
    python scripts/process_ingestion_queue.py --limit 50
    python scripts/process_ingestion_queue.py --retry-failed --max-retries 3
    python scripts/process_ingestion_queue.py --dry-run

Example:
    # Process 100 highest-priority pending papers
    python scripts/process_ingestion_queue.py --limit 100

    # Retry failed items (up to 3 attempts)
    python scripts/process_ingestion_queue.py --retry-failed

    # Preview what would be processed
    python scripts/process_ingestion_queue.py --dry-run --limit 20
"""

import argparse
import asyncio
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# Add packages to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "s2-client" / "src"))

from research_kb_common import get_logger
from research_kb_contracts import SourceType
from research_kb_pdf import PDFDispatcher
from research_kb_storage import (
    DatabaseConfig,
    get_connection_pool,
)
from research_kb_storage.queue_store import QueueStatus, QueueStore
from research_kb_storage.discovery_store import DiscoveryStore

logger = get_logger(__name__)

# Default fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "papers" / "acquired"


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Sanitize text for use in filename."""
    import re
    text = re.sub(r'[<>:"/\\|?*]', "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-.]", "", text)
    return text.lower()[:max_length]


def generate_filename(item: dict) -> str:
    """Generate standardized filename from queue item."""
    # Get first author's last name
    author = "unknown"
    if item.get("authors"):
        parts = item["authors"][0].split()
        if parts:
            author = sanitize_filename(parts[-1], max_length=20)

    # Get title snippet
    title = sanitize_filename(item.get("title") or "untitled", max_length=30)

    # Get year
    year = item.get("year") or "nd"

    return f"{author}_{title}_{year}.pdf"


async def download_pdf(url: str, timeout: float = 60.0) -> Optional[bytes]:
    """Download PDF from URL.

    Args:
        url: URL to download from
        timeout: Request timeout in seconds

    Returns:
        PDF bytes or None on failure
    """
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        headers={"User-Agent": "research-kb/1.0.0 (corpus expansion)"},
    ) as client:
        try:
            logger.info("downloading_pdf", url=url[:80])
            response = await client.get(url)
            response.raise_for_status()

            content = response.content
            if not content.startswith(b"%PDF"):
                logger.warning("not_a_pdf", url=url[:80])
                return None

            return content

        except httpx.HTTPStatusError as e:
            logger.error("http_error", url=url[:80], status=e.response.status_code)
            return None
        except httpx.RequestError as e:
            logger.error("request_error", url=url[:80], error=str(e))
            return None


async def process_queue_item(
    item: dict,
    dispatcher: PDFDispatcher,
    fixtures_dir: Path,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Process a single queue item.

    Args:
        item: Queue item dict from QueueStore.get_pending()
        dispatcher: PDFDispatcher for ingestion
        fixtures_dir: Directory to save downloaded PDFs
        dry_run: If True, only log what would be done

    Returns:
        Tuple of (success, error_message)
    """
    queue_id = item["id"]
    title = item.get("title", "Unknown")[:50]
    pdf_url = item.get("pdf_url")

    if not pdf_url:
        return False, "no_pdf_url"

    if dry_run:
        logger.info("dry_run_would_process", title=title, url=pdf_url[:60])
        return True, ""

    # Update status to downloading
    await QueueStore.update_status(queue_id, QueueStatus.DOWNLOADING)

    # Download PDF
    content = await download_pdf(pdf_url)
    if not content:
        await QueueStore.update_status(queue_id, QueueStatus.FAILED, error_message="download_failed")
        return False, "download_failed"

    # Generate filename and save
    filename = generate_filename(item)
    file_hash = hashlib.sha256(content).hexdigest()

    # Handle collision
    save_path = fixtures_dir / filename
    if save_path.exists():
        save_path = fixtures_dir / f"{save_path.stem}_{file_hash[:8]}.pdf"

    save_path.write_bytes(content)
    logger.info("pdf_saved", path=str(save_path), size_kb=len(content) // 1024)

    # Update status to extracting
    await QueueStore.update_status(queue_id, QueueStatus.EXTRACTING, pdf_path=str(save_path))

    # Prepare metadata from queue item
    metadata = item.get("metadata") or {}
    metadata.update({
        "s2_paper_id": item.get("s2_paper_id"),
        "doi": item.get("doi"),
        "arxiv_id": item.get("arxiv_id"),
        "venue": item.get("venue"),
        "citation_count": metadata.get("citation_count"),
        "queue_id": str(queue_id),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
    })

    # Ingest PDF
    try:
        result = await dispatcher.ingest_pdf(
            pdf_path=save_path,
            source_type=SourceType.PAPER,
            title=item.get("title") or "Unknown",
            authors=item.get("authors"),
            year=item.get("year"),
            metadata=metadata,
        )

        # Update status to completed
        await QueueStore.update_status(queue_id, QueueStatus.COMPLETED)

        logger.info(
            "ingestion_complete",
            title=title,
            source_id=str(result.source.id),
            chunks=result.chunk_count,
        )

        return True, ""

    except Exception as e:
        error_msg = str(e)[:500]
        await QueueStore.update_status(queue_id, QueueStatus.FAILED, error_message=error_msg)
        logger.error("ingestion_failed", title=title, error=error_msg)
        return False, error_msg


async def process_pending_queue(
    limit: int = 50,
    domain_id: Optional[str] = None,
    dry_run: bool = False,
    skip_failed: bool = True,
) -> dict:
    """Process pending items from the ingestion queue.

    Args:
        limit: Maximum items to process
        domain_id: Optional filter by domain
        dry_run: If True, only log what would be done
        skip_failed: If True, continue on individual failures

    Returns:
        Summary dict with counts
    """
    # Ensure fixtures directory exists
    fixtures_dir = FIXTURES_DIR
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dispatcher
    dispatcher = PDFDispatcher()

    # Get pending items
    pending = await QueueStore.get_pending(limit=limit, domain_id=domain_id)

    if not pending:
        logger.info("no_pending_items")
        return {"processed": 0, "success": 0, "failed": 0, "skipped": 0}

    logger.info("processing_queue", pending=len(pending), limit=limit, dry_run=dry_run)

    stats = {"processed": 0, "success": 0, "failed": 0, "skipped": 0}

    for i, item in enumerate(pending, 1):
        try:
            success, error = await process_queue_item(
                item=item,
                dispatcher=dispatcher,
                fixtures_dir=fixtures_dir,
                dry_run=dry_run,
            )

            stats["processed"] += 1
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                if not skip_failed:
                    logger.error("stopping_on_failure", error=error)
                    break

            # Progress log every 10 items
            if i % 10 == 0:
                logger.info("progress", processed=i, total=len(pending), success=stats["success"])

        except Exception as e:
            logger.error("unexpected_error", item_id=str(item["id"]), error=str(e))
            stats["failed"] += 1
            if not skip_failed:
                raise

    # Log discovery audit
    if not dry_run and stats["processed"] > 0:
        try:
            await DiscoveryStore.log_discovery(
                domain_id=domain_id or "causal_inference",
                discovery_method="queue_processing",
                papers_found=len(pending),
                papers_acquired=stats["success"],
                papers_skipped=stats["skipped"],
                papers_failed=stats["failed"],
                metadata={"limit": limit, "dry_run": dry_run},
            )
        except Exception as e:
            logger.warning("discovery_log_failed", error=str(e))

    return stats


async def retry_failed_items(
    max_retries: int = 3,
    domain_id: Optional[str] = None,
) -> int:
    """Reset failed items for retry.

    Args:
        max_retries: Maximum retry attempts per item
        domain_id: Optional filter by domain

    Returns:
        Number of items reset
    """
    reset_count = await QueueStore.retry_failed(
        max_retries=max_retries,
        domain_id=domain_id,
    )

    logger.info("failed_items_reset", count=reset_count, max_retries=max_retries)
    return reset_count


async def show_queue_stats() -> None:
    """Display current queue statistics."""
    stats = await QueueStore.get_stats()

    print("\n" + "=" * 60)
    print("Ingestion Queue Statistics")
    print("=" * 60)
    print(f"Total items: {stats['total']}")
    print("\nBy status:")
    for status, count in stats.get("by_status", {}).items():
        print(f"  {status}: {count}")
    print("\nBy domain:")
    for domain, count in stats.get("by_domain", {}).items():
        print(f"  {domain}: {count}")
    if stats.get("with_retries", 0) > 0:
        print(f"\nItems with retries: {stats['with_retries']}")
        print(f"Max retry count: {stats['max_retries']}")
        print(f"Avg retry count: {stats['avg_retries']:.1f}")
    print("=" * 60 + "\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process pending items from the ingestion queue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum items to process (default: 50)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter by domain (e.g., causal_inference)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually doing it",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reset failed items for retry (does not process)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for --retry-failed (default: 3)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show queue statistics and exit",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop processing on first failure",
    )

    args = parser.parse_args()

    # Initialize database connection
    config = DatabaseConfig()
    await get_connection_pool(config)

    if args.stats:
        await show_queue_stats()
        return

    if args.retry_failed:
        reset = await retry_failed_items(
            max_retries=args.max_retries,
            domain_id=args.domain,
        )
        print(f"\nReset {reset} items for retry.\n")
        return

    # Process queue
    stats = await process_pending_queue(
        limit=args.limit,
        domain_id=args.domain,
        dry_run=args.dry_run,
        skip_failed=not args.stop_on_failure,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Queue Processing Summary")
    print("=" * 60)
    print(f"Processed: {stats['processed']}")
    print(f"Success:   {stats['success']}")
    print(f"Failed:    {stats['failed']}")
    print(f"Skipped:   {stats['skipped']}")
    if args.dry_run:
        print("\n(Dry run - no changes made)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
