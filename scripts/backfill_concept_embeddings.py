#!/usr/bin/env python3
"""Backfill embeddings for existing concepts.

This script populates the `embedding` column for all concepts that don't have
embeddings yet. Embeddings are generated from: "{canonical_name}: {definition}"

Features:
- Checkpointing: Progress saved every N batches, resume with --resume
- Graceful shutdown: Ctrl+C saves checkpoint before exit
- Dead Letter Queue: Failed concepts saved to .dlq/embedding_backfill/
- Batched updates: Single transaction per batch
- Retry logic: Transient failures retried with exponential backoff

Prerequisites:
- Embedding server must be running: python -m research_kb_pdf.embed_server
- Database must be accessible

Usage:
    python scripts/backfill_concept_embeddings.py [--batch-size 100] [--dry-run]
    python scripts/backfill_concept_embeddings.py --resume  # Continue from checkpoint
    python scripts/backfill_concept_embeddings.py --clear-checkpoint  # Start fresh

Example:
    # Check embedding server is running
    curl -s http://localhost:8765/health || python -m research_kb_pdf.embed_server &

    # Run backfill
    python scripts/backfill_concept_embeddings.py

    # Verify
    psql -c "SELECT COUNT(*) FROM concepts WHERE embedding IS NOT NULL"
"""

import argparse
import asyncio
import json
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Add packages to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))

from pgvector.asyncpg import register_vector

from research_kb_common import get_logger
from research_kb_pdf.embedding_client import EmbeddingClient
from research_kb_storage.concept_store import ConceptStore
from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)

# Checkpoint and DLQ paths
CHECKPOINT_FILE = Path(__file__).parent.parent / ".embedding_backfill_checkpoint.json"
DLQ_DIR = Path(__file__).parent.parent / ".dlq" / "embedding_backfill"


@dataclass
class BackfillStats:
    """Statistics from backfill run."""

    total: int = 0
    updated: int = 0
    errors: int = 0
    skipped: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def concepts_per_second(self) -> float:
        if self.duration_seconds == 0:
            return 0
        return self.updated / self.duration_seconds

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "updated": self.updated,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration_seconds": self.duration_seconds,
            "concepts_per_second": round(self.concepts_per_second, 2),
        }


class EmbeddingBackfillRunner:
    """Manages embedding backfill with checkpointing and graceful shutdown.

    This class provides:
    - Periodic checkpointing to JSON file
    - Graceful shutdown on SIGINT/SIGTERM
    - Dead letter queue for failed concepts
    - Batched database transactions
    - Retry logic with exponential backoff
    """

    def __init__(
        self,
        batch_size: int = 100,
        checkpoint_interval: int = 5,
        dry_run: bool = False,
    ):
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.dry_run = dry_run

        self._shutdown_requested = False
        self._processed_ids: set[UUID] = set()
        self.client: Optional[EmbeddingClient] = None
        self.stats = BackfillStats()

        # Original signal handlers (restored on cleanup)
        self._original_sigint = None
        self._original_sigterm = None

    def setup_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_shutdown)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_shutdown)
        logger.debug("signal_handlers_installed")

    def restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handle_shutdown(self, signum: int, frame) -> None:
        """Handle shutdown signal by saving checkpoint."""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info("shutdown_requested", signal=signal_name)
        print(f"\n⚡ {signal_name} received - saving checkpoint...")

        self._shutdown_requested = True
        self.save_checkpoint()
        print(f"✅ Checkpoint saved. Resume with: --resume")

    def load_checkpoint(self) -> set[UUID]:
        """Load processed concept IDs from checkpoint file."""
        if not CHECKPOINT_FILE.exists():
            return set()

        try:
            with open(CHECKPOINT_FILE) as f:
                data = json.load(f)
                processed = {UUID(id_str) for id_str in data.get("processed_ids", [])}
                logger.info(
                    "checkpoint_loaded",
                    count=len(processed),
                    timestamp=data.get("timestamp"),
                )
                return processed
        except Exception as e:
            logger.warning("checkpoint_load_failed", error=str(e))
            return set()

    def save_checkpoint(self) -> None:
        """Save processed concept IDs to checkpoint file."""
        try:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(
                    {
                        "processed_ids": [str(id) for id in self._processed_ids],
                        "timestamp": datetime.now().isoformat(),
                        "stats": self.stats.to_dict(),
                    },
                    f,
                    indent=2,
                )
            logger.debug("checkpoint_saved", count=len(self._processed_ids))
        except Exception as e:
            logger.error("checkpoint_save_failed", error=str(e))

    def clear_checkpoint(self) -> None:
        """Remove checkpoint file."""
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("checkpoint_cleared")
            print("✅ Checkpoint cleared")

    def save_to_dlq(self, concept_id: UUID, canonical_name: str, error: str) -> None:
        """Save failed concept to dead letter queue for later analysis."""
        DLQ_DIR.mkdir(parents=True, exist_ok=True)
        dlq_file = DLQ_DIR / f"{concept_id}.json"

        try:
            with open(dlq_file, "w") as f:
                json.dump(
                    {
                        "concept_id": str(concept_id),
                        "canonical_name": canonical_name,
                        "error": error,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
            logger.debug("dlq_saved", concept_id=str(concept_id))
        except Exception as e:
            logger.error("dlq_save_failed", concept_id=str(concept_id), error=str(e))

    @staticmethod
    def format_embedding_text(concept) -> str:
        """Format concept data for embedding.

        Uses canonical_name + definition for semantic richness.
        If no definition, uses just the canonical name.
        """
        if concept.definition:
            return f"{concept.canonical_name}: {concept.definition}"
        return concept.canonical_name

    async def get_concepts_without_embeddings(
        self,
        resume: bool = False,
    ) -> list:
        """Fetch all concepts that don't have embeddings yet.

        When resume=True, also filters out previously processed concepts
        from the checkpoint (in case they failed after embedding but before
        the DB write).
        """
        all_concepts = []
        offset = 0
        batch_size = 1000

        # Load checkpoint if resuming
        if resume:
            self._processed_ids = self.load_checkpoint()
            if self._processed_ids:
                print(f"📋 Loaded checkpoint: {len(self._processed_ids)} already processed")

        while True:
            concepts = await ConceptStore.list_all(limit=batch_size, offset=offset)
            if not concepts:
                break

            # Filter to concepts without embeddings
            concepts_needing = [c for c in concepts if c.embedding is None]

            # If resuming, also filter out previously processed
            if resume and self._processed_ids:
                concepts_needing = [
                    c for c in concepts_needing if c.id not in self._processed_ids
                ]

            all_concepts.extend(concepts_needing)
            offset += batch_size

            logger.debug(
                "fetched_concepts_batch",
                offset=offset,
                batch_count=len(concepts),
                needing_embeddings=len(concepts_needing),
            )

        return all_concepts

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, OSError, TimeoutError)),
    )
    def embed_batch_with_retry(self, texts: list[str]) -> list:
        """Generate embeddings with retry on transient failures."""
        return self.client.embed_batch(texts)

    async def update_batch(
        self,
        concepts: list,
        embeddings: list,
    ) -> tuple[int, int]:
        """Update multiple concepts in a single transaction.

        Returns:
            Tuple of (updated_count, error_count)
        """
        pool = await get_connection_pool()
        updated = 0
        errors = 0

        async with pool.acquire() as conn:
            await register_vector(conn)  # Required for pgvector type handling
            async with conn.transaction():
                for concept, embedding in zip(concepts, embeddings):
                    try:
                        await conn.execute(
                            "UPDATE concepts SET embedding = $1 WHERE id = $2",
                            embedding,
                            concept.id,
                        )
                        updated += 1
                        self._processed_ids.add(concept.id)
                    except Exception as e:
                        logger.error(
                            "concept_update_failed",
                            concept_id=str(concept.id),
                            error=str(e),
                        )
                        self.save_to_dlq(concept.id, concept.canonical_name, str(e))
                        errors += 1

        return updated, errors

    async def run(self, resume: bool = False) -> dict:
        """Execute the backfill operation.

        Args:
            resume: If True, continue from checkpoint

        Returns:
            Stats dictionary with counts
        """
        logger.info(
            "backfill_starting",
            batch_size=self.batch_size,
            checkpoint_interval=self.checkpoint_interval,
            dry_run=self.dry_run,
            resume=resume,
        )

        # Setup signal handlers
        self.setup_signal_handlers()

        try:
            # Check embedding server is available
            try:
                self.client = EmbeddingClient()
                status = self.client.ping()
                logger.info("embedding_server_connected", status=status)
            except ConnectionError as e:
                logger.error("embedding_server_not_available", error=str(e))
                print("\n❌ Embedding server not running!")
                print("Start it with: python -m research_kb_pdf.embed_server")
                return {"error": "embedding_server_not_available"}

            # Fetch concepts without embeddings
            concepts = await self.get_concepts_without_embeddings(resume=resume)
            self.stats.total = len(concepts)

            if self.stats.total == 0:
                logger.info("no_concepts_need_embeddings")
                print("\n✅ All concepts already have embeddings!")
                self.clear_checkpoint()
                return self.stats.to_dict()

            logger.info("concepts_to_backfill", count=self.stats.total)
            print(f"\n📊 Found {self.stats.total} concepts without embeddings")

            if self.dry_run:
                print("🔍 DRY RUN - no changes will be made")
                for concept in concepts[:5]:
                    text = self.format_embedding_text(concept)
                    print(f"  - {concept.canonical_name}: '{text[:80]}...'")
                return {**self.stats.to_dict(), "dry_run": True}

            # Process in batches
            batches_since_checkpoint = 0

            for i in range(0, self.stats.total, self.batch_size):
                # Check for shutdown request
                if self._shutdown_requested:
                    logger.info("shutdown_during_backfill", processed=self.stats.updated)
                    break

                batch = concepts[i : i + self.batch_size]
                texts = [self.format_embedding_text(c) for c in batch]

                try:
                    # Generate embeddings with retry
                    embeddings = self.embed_batch_with_retry(texts)

                    # Update database in single transaction
                    updated, errors = await self.update_batch(batch, embeddings)
                    self.stats.updated += updated
                    self.stats.errors += errors

                except Exception as e:
                    logger.error("batch_embedding_failed", batch_start=i, error=str(e))
                    # Save all concepts in failed batch to DLQ
                    for concept in batch:
                        self.save_to_dlq(concept.id, concept.canonical_name, str(e))
                    self.stats.errors += len(batch)

                # Progress update
                progress = min(i + self.batch_size, self.stats.total)
                pct = (progress / self.stats.total) * 100
                print(f"  Progress: {progress}/{self.stats.total} ({pct:.1f}%)", end="\r")

                # Periodic checkpoint
                batches_since_checkpoint += 1
                if batches_since_checkpoint >= self.checkpoint_interval:
                    self.save_checkpoint()
                    batches_since_checkpoint = 0

            print()  # Newline after progress

            self.stats.end_time = datetime.now()
            logger.info(
                "backfill_complete",
                **self.stats.to_dict(),
            )

            # Clear checkpoint on successful completion
            if not self._shutdown_requested and self.stats.errors == 0:
                self.clear_checkpoint()

            return self.stats.to_dict()

        finally:
            self.restore_signal_handlers()


async def verify_embeddings() -> dict:
    """Verify embedding coverage after backfill."""
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM concepts")
        with_embeddings = await conn.fetchval(
            "SELECT COUNT(*) FROM concepts WHERE embedding IS NOT NULL"
        )
        without_embeddings = total - with_embeddings

    return {
        "total_concepts": total,
        "with_embeddings": with_embeddings,
        "without_embeddings": without_embeddings,
        "coverage_pct": (with_embeddings / total * 100) if total > 0 else 0,
    }


def check_dlq() -> int:
    """Check for items in the dead letter queue."""
    if not DLQ_DIR.exists():
        return 0
    return len(list(DLQ_DIR.glob("*.json")))


async def main():
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for concepts without them"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of concepts to process per batch (default: 100)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N batches (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually update database, just show what would be done",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify current embedding coverage, don't backfill",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (continue previous run)",
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear checkpoint file and start fresh",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Concept Embedding Backfill")
    print("=" * 60)

    # Handle clear checkpoint
    if args.clear_checkpoint:
        runner = EmbeddingBackfillRunner()
        runner.clear_checkpoint()
        if not args.verify_only and not args.resume:
            return

    if args.verify_only:
        stats = await verify_embeddings()
        print(f"\n📊 Current embedding coverage:")
        print(f"  Total concepts: {stats['total_concepts']}")
        print(f"  With embeddings: {stats['with_embeddings']}")
        print(f"  Without embeddings: {stats['without_embeddings']}")
        print(f"  Coverage: {stats['coverage_pct']:.1f}%")

        dlq_count = check_dlq()
        if dlq_count > 0:
            print(f"\n⚠️  {dlq_count} items in dead letter queue ({DLQ_DIR})")
        return

    # Run backfill
    runner = EmbeddingBackfillRunner(
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        dry_run=args.dry_run,
    )

    result = await runner.run(resume=args.resume)

    if "error" in result:
        sys.exit(1)

    # Verify final state
    if not args.dry_run:
        print("\n📊 Final verification:")
        stats = await verify_embeddings()
        print(f"  Total concepts: {stats['total_concepts']}")
        print(f"  With embeddings: {stats['with_embeddings']}")
        print(f"  Coverage: {stats['coverage_pct']:.1f}%")

        if stats["without_embeddings"] > 0:
            print(f"\n⚠️  {stats['without_embeddings']} concepts still without embeddings")
            print("  Run with --resume to continue")
        else:
            print("\n✅ All concepts now have embeddings!")

        dlq_count = check_dlq()
        if dlq_count > 0:
            print(f"\n⚠️  {dlq_count} failed concepts in DLQ ({DLQ_DIR})")
            print("  Review and retry manually if needed")


if __name__ == "__main__":
    asyncio.run(main())
