#!/usr/bin/env python3
"""Extract concepts and relationships from ingested chunks.

This script:
1. Loads chunks from the database (with optional filtering)
2. Extracts concepts using LLM (Ollama or Anthropic)
3. Deduplicates concepts by canonical name
4. Stores concepts, relationships, and chunk-concept links
5. Syncs to Neo4j for graph queries
6. Reports extraction statistics

Usage:
    # Process all chunks with Ollama (local)
    python scripts/extract_concepts.py

    # Use Anthropic Haiku (fast iteration)
    python scripts/extract_concepts.py --backend anthropic --model haiku

    # Use Anthropic Opus (production quality)
    python scripts/extract_concepts.py --backend anthropic --model opus

    # Process specific source
    python scripts/extract_concepts.py --source-id <uuid>

    # Dry run (no database writes)
    python scripts/extract_concepts.py --dry-run

    # Resume from checkpoint
    python scripts/extract_concepts.py --resume

    # Limit chunks for testing
    python scripts/extract_concepts.py --limit 50
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "extraction" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_contracts import Chunk, ConceptType, RelationshipType
from research_kb_extraction import (
    ChunkExtraction,
    ConceptExtractor,
    Deduplicator,
    ExtractionMetrics,
    GraphSyncService,
    LLMClient,
    OllamaClient,
    get_llm_client,
)
from research_kb_storage import (
    ChunkConceptStore,
    ChunkStore,
    ConceptStore,
    RelationshipStore,
    SourceStore,
    get_connection_pool,
)

logger = get_logger(__name__)

# Checkpoint file for resume capability
CHECKPOINT_FILE = Path(__file__).parent.parent / ".extraction_checkpoint.json"
DLQ_DIR = Path(__file__).parent.parent / ".dlq" / "extraction"


@dataclass
class ExtractionStats:
    """Statistics from extraction run."""

    chunks_processed: int = 0
    chunks_skipped: int = 0
    chunks_failed: int = 0
    concepts_extracted: int = 0
    concepts_new: int = 0
    concepts_merged: int = 0
    relationships_extracted: int = 0
    relationships_stored: int = 0
    chunk_links_created: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def chunks_per_second(self) -> float:
        if self.duration_seconds == 0:
            return 0
        return self.chunks_processed / self.duration_seconds

    def to_dict(self) -> dict:
        return {
            "chunks_processed": self.chunks_processed,
            "chunks_skipped": self.chunks_skipped,
            "chunks_failed": self.chunks_failed,
            "concepts_extracted": self.concepts_extracted,
            "concepts_new": self.concepts_new,
            "concepts_merged": self.concepts_merged,
            "relationships_extracted": self.relationships_extracted,
            "relationships_stored": self.relationships_stored,
            "chunk_links_created": self.chunk_links_created,
            "duration_seconds": self.duration_seconds,
            "chunks_per_second": self.chunks_per_second,
        }

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Duration: {self.duration_seconds:.1f}s ({self.chunks_per_second:.2f} chunks/sec)")
        print(f"Chunks: {self.chunks_processed} processed, {self.chunks_skipped} skipped, {self.chunks_failed} failed")
        print(f"Concepts: {self.concepts_extracted} extracted, {self.concepts_new} new, {self.concepts_merged} merged")
        print(f"Relationships: {self.relationships_extracted} extracted, {self.relationships_stored} stored")
        print(f"Chunk-Concept Links: {self.chunk_links_created}")
        print("=" * 60)


class ExtractionPipeline:
    """Orchestrates concept extraction from chunks."""

    def __init__(
        self,
        backend: str = "ollama",
        model: Optional[str] = None,
        confidence_threshold: float = 0.7,
        batch_size: int = 10,
        checkpoint_interval: int = 10,  # Reduced from 50 to minimize data loss on crash
        dry_run: bool = False,
        sync_neo4j: bool = True,
        skip_backup: bool = False,
        concurrency: int = 1,
        num_ctx: int = 2048,
    ):
        self.backend = backend
        self.model = model  # None means use backend's default
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.dry_run = dry_run
        self.sync_neo4j = sync_neo4j
        self.skip_backup = skip_backup
        self.concurrency = max(1, concurrency)
        self.num_ctx = num_ctx
        self.backup_path: Optional[str] = None

        # Semaphore to limit concurrent LLM requests
        self._semaphore: Optional[asyncio.Semaphore] = None

        self.llm_client: Optional[LLMClient] = None
        self.extractor: Optional[ConceptExtractor] = None
        self.deduplicator: Optional[Deduplicator] = None
        self.graph_sync: Optional[GraphSyncService] = None

        self.stats = ExtractionStats()
        self.processed_chunk_ids: set[UUID] = set()

        # Concept name -> UUID cache
        self._concept_cache: dict[str, UUID] = {}

        # Lock for database writes (ensure thread safety)
        self._db_lock: Optional[asyncio.Lock] = None

        # Extraction metrics for observability
        self.metrics: Optional[ExtractionMetrics] = None

    async def initialize(self) -> None:
        """Initialize clients and load existing concepts."""
        # Initialize concurrency controls
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._db_lock = asyncio.Lock()

        if self.concurrency > 1:
            logger.info("parallel_mode_enabled", concurrency=self.concurrency)

        # Initialize database pool
        await get_connection_pool()

        # Initialize LLM client using factory
        llm_kwargs = {"temperature": 0.1}
        if self.backend == "ollama":
            llm_kwargs["num_ctx"] = self.num_ctx
        elif self.backend == "llamacpp":
            llm_kwargs["n_ctx"] = self.num_ctx
            llm_kwargs["n_gpu_layers"] = 20  # 20 layers fits in 8GB VRAM with desktop overhead

        self.llm_client = get_llm_client(
            backend=self.backend,
            model=self.model,
            **llm_kwargs,
        )

        # Initialize extraction metrics
        self.metrics = ExtractionMetrics(backend=self.llm_client.extraction_method)

        # Check availability
        if not await self.llm_client.is_available():
            if self.backend == "ollama":
                raise RuntimeError("Ollama server not available. Start with: ollama serve")
            elif self.backend == "instructor":
                raise RuntimeError("Ollama server not available. Start with: ollama serve")
            elif self.backend == "llamacpp":
                raise RuntimeError(
                    "llama-cpp-python not available. Install with:\n"
                    "  sudo ./scripts/install_llama_cpp_cuda.sh\n"
                    "  ./scripts/build_llama_cpp_cuda.sh\n"
                    "  ./scripts/download_gguf_model.sh"
                )
            elif self.backend == "anthropic":
                raise RuntimeError("Anthropic API not available. Check ANTHROPIC_API_KEY.")
            else:
                raise RuntimeError(f"Backend {self.backend} not available.")

        # For Ollama, check if model is loaded
        if self.backend == "ollama" and hasattr(self.llm_client, "is_model_loaded"):
            if not await self.llm_client.is_model_loaded():
                model_name = self.model or "llama3.1:8b"
                raise RuntimeError(f"Model {model_name} not loaded. Pull with: ollama pull {model_name}")

        logger.info(
            "llm_client_connected",
            backend=self.backend,
            extraction_method=self.llm_client.extraction_method,
        )

        # Create pre-extraction backup (HARD REQUIREMENT)
        if self.skip_backup:
            logger.warning("backup_skipped", message="--skip-backup used. Data loss risk!")
            print("\n⚠️  WARNING: Pre-extraction backup skipped. Use at your own risk!\n")
        elif not self.dry_run:
            print("\n📦 Creating pre-extraction backup (required for data protection)...")
            backup_script = Path(__file__).parent / "backup_db.sh"
            try:
                result = subprocess.run(
                    [str(backup_script), "--pre-extraction"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.backup_path = result.stdout.strip().split("\n")[-1]  # Last line has file path
                logger.info("pre_extraction_backup_created", backup_path=self.backup_path)
                print(f"✅ Backup created: {self.backup_path}\n")
            except subprocess.CalledProcessError as e:
                logger.error("backup_failed", error=e.stderr)
                raise RuntimeError(
                    f"Pre-extraction backup failed. Cannot proceed without backup.\n"
                    f"Error: {e.stderr}\n"
                    f"Use --skip-backup to bypass (NOT recommended)"
                )

        # Initialize extractor and deduplicator
        self.deduplicator = Deduplicator()
        self.extractor = ConceptExtractor(
            ollama_client=self.llm_client,  # Still called ollama_client internally
            deduplicator=self.deduplicator,
            confidence_threshold=self.confidence_threshold,
        )

        # Initialize Neo4j sync if enabled
        if self.sync_neo4j:
            self.graph_sync = GraphSyncService()
            if await self.graph_sync.is_available():
                logger.info("neo4j_connected")
            else:
                logger.warning("neo4j_unavailable", message="Graph sync disabled")
                self.graph_sync = None

        # Load existing concepts into deduplicator cache
        await self._load_existing_concepts()

    async def _load_existing_concepts(self) -> None:
        """Load existing concepts into cache for deduplication."""
        concepts = await ConceptStore.list_all(limit=10000)
        for concept in concepts:
            self._concept_cache[concept.canonical_name] = concept.id
            self.deduplicator.register_known_concept(concept.canonical_name, concept.id)

        logger.info("concepts_loaded_to_cache", count=len(concepts))

    async def close(self) -> None:
        """Clean up resources."""
        if self.extractor:
            await self.extractor.close()
        if self.graph_sync:
            await self.graph_sync.close()

    def load_checkpoint(self) -> set[UUID]:
        """Load processed chunk IDs from checkpoint."""
        if not CHECKPOINT_FILE.exists():
            return set()

        try:
            with open(CHECKPOINT_FILE) as f:
                data = json.load(f)
                return {UUID(id_str) for id_str in data.get("processed_chunk_ids", [])}
        except Exception as e:
            logger.warning("checkpoint_load_failed", error=str(e))
            return set()

    def save_checkpoint(self) -> None:
        """Save processed chunk IDs to checkpoint."""
        try:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({
                    "processed_chunk_ids": [str(id) for id in self.processed_chunk_ids],
                    "timestamp": datetime.now().isoformat(),
                    "stats": self.stats.to_dict(),
                }, f, indent=2)
            logger.debug("checkpoint_saved", count=len(self.processed_chunk_ids))
        except Exception as e:
            logger.warning("checkpoint_save_failed", error=str(e))

    def clear_checkpoint(self) -> None:
        """Clear the checkpoint file."""
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("checkpoint_cleared")

    async def save_to_dlq(self, chunk: Chunk, error: str) -> None:
        """Save failed chunk to dead letter queue."""
        DLQ_DIR.mkdir(parents=True, exist_ok=True)
        dlq_file = DLQ_DIR / f"{chunk.id}.json"

        try:
            with open(dlq_file, "w") as f:
                json.dump({
                    "chunk_id": str(chunk.id),
                    "source_id": str(chunk.source_id),
                    "content_preview": chunk.content[:500],
                    "error": error,
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error("dlq_save_failed", chunk_id=str(chunk.id), error=str(e))

    async def get_chunks_to_process(
        self,
        source_id: Optional[UUID] = None,
        limit: Optional[int] = None,
        resume: bool = False,
    ) -> list[Chunk]:
        """Get chunks that need processing.

        When resume=True, queries chunks that have NO concept links in the database,
        which is more efficient than loading all chunks and filtering in Python.
        """
        batch_limit = limit or 10000

        if resume:
            # Query unprocessed chunks directly from database (much more efficient)
            pool = await get_connection_pool()
            async with pool.acquire() as conn:
                # Get chunks that have no concept links yet
                rows = await conn.fetch(
                    """
                    SELECT c.id, c.source_id, c.content, c.content_hash,
                           c.location, c.page_start, c.page_end,
                           c.embedding, c.metadata, c.created_at
                    FROM chunks c
                    LEFT JOIN chunk_concepts cc ON c.id = cc.chunk_id
                    WHERE cc.chunk_id IS NULL
                    ORDER BY c.created_at
                    LIMIT $1
                    """,
                    batch_limit,
                )

            # Convert to Chunk objects
            chunks = []
            for row in rows:
                chunks.append(Chunk(
                    id=row["id"],
                    source_id=row["source_id"],
                    content=row["content"],
                    content_hash=row["content_hash"],
                    location=row["location"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    embedding=list(row["embedding"]) if row["embedding"] is not None else None,
                    metadata=row["metadata"] or {},
                    created_at=row["created_at"],
                ))

            # Also load checkpoint IDs for tracking (but don't use for filtering)
            checkpoint_ids = self.load_checkpoint()
            self.processed_chunk_ids = checkpoint_ids

            logger.info(
                "resume_from_database",
                unprocessed_chunks=len(chunks),
                checkpoint_size=len(checkpoint_ids),
            )
            return chunks

        # Non-resume mode: get all chunks up to limit
        if source_id:
            chunks = await ChunkStore.list_by_source(source_id, limit=batch_limit)
        else:
            chunks = await ChunkStore.list_all(limit=batch_limit)

        return chunks

    async def process_chunk(self, chunk: Chunk) -> Optional[ChunkExtraction]:
        """Extract concepts from a single chunk with timing."""
        start_time = time.perf_counter()
        try:
            extraction = await self.extractor.extract_from_chunk(chunk)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record metrics
            if self.metrics and extraction:
                self.metrics.record_success(
                    concepts=extraction.concept_count,
                    relationships=extraction.relationship_count,
                    latency_ms=latency_ms,
                )

            return extraction
        except json.JSONDecodeError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error("chunk_json_parse_failed", chunk_id=str(chunk.id), error=str(e))
            if self.metrics:
                self.metrics.record_json_failure(latency_ms=latency_ms)
            await self.save_to_dlq(chunk, f"JSON parse error: {e}")
            return None
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error("chunk_extraction_failed", chunk_id=str(chunk.id), error=str(e))
            if self.metrics:
                self.metrics.record_validation_failure(latency_ms=latency_ms)
            await self.save_to_dlq(chunk, str(e))
            return None

    async def store_extraction(
        self,
        chunk: Chunk,
        extraction: ChunkExtraction,
    ) -> tuple[int, int, int]:
        """Store extracted concepts, relationships, and links.

        Returns:
            Tuple of (new_concepts, relationships_stored, links_created)
        """
        if self.dry_run:
            return (len(extraction.concepts), len(extraction.relationships), len(extraction.concepts))

        new_concepts = 0
        relationships_stored = 0
        links_created = 0

        # Process concepts
        concept_name_to_id: dict[str, UUID] = {}

        for extracted in extraction.concepts:
            canonical = self.deduplicator.to_canonical_name(extracted.name)

            # Check if concept already exists
            if canonical in self._concept_cache:
                concept_id = self._concept_cache[canonical]
                concept_name_to_id[extracted.name.lower()] = concept_id
            else:
                # Create new concept
                try:
                    concept = await ConceptStore.create(
                        name=extracted.name,
                        canonical_name=canonical,
                        concept_type=ConceptType(extracted.concept_type),
                        aliases=extracted.aliases,
                        definition=extracted.definition,
                        extraction_method=self.llm_client.extraction_method,
                        confidence_score=extracted.confidence,
                    )
                    self._concept_cache[canonical] = concept.id
                    concept_name_to_id[extracted.name.lower()] = concept.id
                    new_concepts += 1

                    # Sync to Neo4j
                    if self.graph_sync:
                        await self.graph_sync.sync_concept(
                            concept_id=concept.id,
                            name=concept.name,
                            canonical_name=concept.canonical_name,
                            concept_type=concept.concept_type.value,
                            definition=concept.definition,
                        )

                except Exception as e:
                    logger.warning(
                        "concept_store_failed",
                        name=extracted.name,
                        error=str(e),
                    )

            # Create chunk-concept link
            if extracted.name.lower() in concept_name_to_id:
                try:
                    await ChunkConceptStore.create(
                        chunk_id=chunk.id,
                        concept_id=concept_name_to_id[extracted.name.lower()],
                        mention_type="reference",
                        relevance_score=extracted.confidence,
                    )
                    links_created += 1
                except Exception:
                    pass  # Link may already exist

        # Process relationships
        for rel in extraction.relationships:
            source_name = rel.source_concept.lower()
            target_name = rel.target_concept.lower()

            # Get concept IDs
            source_id = concept_name_to_id.get(source_name)
            target_id = concept_name_to_id.get(target_name)

            # Try canonical name lookup if direct lookup failed
            if not source_id:
                source_canonical = self.deduplicator.to_canonical_name(rel.source_concept)
                source_id = self._concept_cache.get(source_canonical)

            if not target_id:
                target_canonical = self.deduplicator.to_canonical_name(rel.target_concept)
                target_id = self._concept_cache.get(target_canonical)

            if source_id and target_id and source_id != target_id:
                try:
                    relationship = await RelationshipStore.create(
                        source_concept_id=source_id,
                        target_concept_id=target_id,
                        relationship_type=RelationshipType(rel.relationship_type),
                        evidence_chunk_ids=[chunk.id],
                        confidence_score=rel.confidence,
                    )
                    relationships_stored += 1

                    # Sync to Neo4j
                    if self.graph_sync:
                        await self.graph_sync.sync_relationship(
                            relationship_id=relationship.id,
                            source_concept_id=source_id,
                            target_concept_id=target_id,
                            relationship_type=rel.relationship_type,
                            strength=relationship.strength,
                        )

                except Exception as e:
                    # Relationship may already exist
                    logger.debug(
                        "relationship_store_skipped",
                        source=rel.source_concept,
                        target=rel.target_concept,
                        error=str(e),
                    )

        return (new_concepts, relationships_stored, links_created)

    async def _process_chunk_with_semaphore(
        self,
        chunk: Chunk,
    ) -> tuple[Chunk, Optional[ChunkExtraction]]:
        """Process a single chunk with semaphore for rate limiting.

        Returns:
            Tuple of (chunk, extraction) - extraction may be None on failure
        """
        async with self._semaphore:
            extraction = await self.process_chunk(chunk)
            return (chunk, extraction)

    async def _store_result(
        self,
        chunk: Chunk,
        extraction: Optional[ChunkExtraction],
    ) -> None:
        """Store extraction result with proper locking for DB writes."""
        if len(chunk.content) < 100:
            self.stats.chunks_skipped += 1
            return

        if extraction is None:
            self.stats.chunks_failed += 1
            return

        # Lock for database writes to prevent race conditions
        async with self._db_lock:
            # Update stats
            self.stats.chunks_processed += 1
            self.stats.concepts_extracted += extraction.concept_count
            self.stats.relationships_extracted += extraction.relationship_count

            # Store results
            new_concepts, rels_stored, links = await self.store_extraction(chunk, extraction)
            self.stats.concepts_new += new_concepts
            self.stats.concepts_merged += extraction.concept_count - new_concepts
            self.stats.relationships_stored += rels_stored
            self.stats.chunk_links_created += links

            # Track processed chunks
            self.processed_chunk_ids.add(chunk.id)

    async def run(
        self,
        source_id: Optional[UUID] = None,
        limit: Optional[int] = None,
        resume: bool = False,
    ) -> ExtractionStats:
        """Run the extraction pipeline.

        With concurrency > 1, processes chunks in parallel batches.
        """
        await self.initialize()

        try:
            chunks = await self.get_chunks_to_process(source_id, limit, resume)
            total = len(chunks)

            if total == 0:
                logger.info("no_chunks_to_process")
                return self.stats

            logger.info(
                "extraction_started",
                total_chunks=total,
                dry_run=self.dry_run,
                concurrency=self.concurrency,
            )

            # Filter out short chunks upfront
            valid_chunks = [c for c in chunks if len(c.content) >= 100]
            skipped = len(chunks) - len(valid_chunks)
            self.stats.chunks_skipped = skipped

            if self.concurrency == 1:
                # Sequential processing (original behavior)
                for i, chunk in enumerate(valid_chunks):
                    if (i + 1) % 10 == 0 or i == 0:
                        print(f"\rProcessing chunk {i + 1}/{len(valid_chunks)} ({100 * (i + 1) / len(valid_chunks):.1f}%)...", end="", flush=True)

                    extraction = await self.process_chunk(chunk)
                    await self._store_result(chunk, extraction)

                    if (i + 1) % self.checkpoint_interval == 0:
                        self.save_checkpoint()
            else:
                # Parallel processing in batches
                batch_size = self.concurrency * 2  # Process batches of 2x concurrency
                processed = 0

                for batch_start in range(0, len(valid_chunks), batch_size):
                    batch = valid_chunks[batch_start:batch_start + batch_size]

                    # Process batch concurrently
                    tasks = [self._process_chunk_with_semaphore(chunk) for chunk in batch]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Store results sequentially (DB writes need ordering)
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error("batch_chunk_failed", error=str(result))
                            self.stats.chunks_failed += 1
                        else:
                            chunk, extraction = result
                            await self._store_result(chunk, extraction)

                    processed += len(batch)
                    print(f"\rProcessing: {processed}/{len(valid_chunks)} ({100 * processed / len(valid_chunks):.1f}%) [concurrency={self.concurrency}]...", end="", flush=True)

                    # Checkpoint after each batch
                    self.save_checkpoint()

            print()  # New line after progress
            self.stats.end_time = datetime.now()

            # Final checkpoint
            if not self.dry_run:
                self.save_checkpoint()

            return self.stats

        finally:
            await self.close()


async def main():
    parser = argparse.ArgumentParser(description="Extract concepts from ingested chunks")
    parser.add_argument("--source-id", type=str, help="Process only chunks from this source")
    parser.add_argument("--limit", type=int, help="Limit number of chunks to process")
    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=["ollama", "instructor", "llamacpp", "anthropic"],
        help="LLM backend (ollama=local, instructor=ollama+validation, llamacpp=direct, anthropic=API)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name/path (ollama: llama3.1:8b, llamacpp: path/to/model.gguf, anthropic: haiku/sonnet/opus)"
    )
    parser.add_argument("--confidence", type=float, default=0.7, help="Minimum confidence threshold")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent LLM requests (default: 1, max recommended: 20 for Haiku)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Run without storing results")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--clear-checkpoint", action="store_true", help="Clear checkpoint and start fresh")
    parser.add_argument("--no-neo4j", action="store_true", help="Disable Neo4j sync")
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip pre-extraction backup (NOT recommended - risk of data loss)"
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=2048,
        help="Context window size in tokens (default: 2048, reduced from 4096 for better concurrency)"
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Export metrics to Prometheus text file (e.g., /tmp/extraction_metrics.txt)"
    )

    args = parser.parse_args()

    # Clear checkpoint if requested
    if args.clear_checkpoint:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            print("Checkpoint cleared.")
        return

    # Parse source ID
    source_id = UUID(args.source_id) if args.source_id else None

    # Create and run pipeline
    pipeline = ExtractionPipeline(
        backend=args.backend,
        model=args.model,
        confidence_threshold=args.confidence,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        sync_neo4j=not args.no_neo4j,
        skip_backup=args.skip_backup,
        num_ctx=args.num_ctx,
    )

    try:
        stats = await pipeline.run(
            source_id=source_id,
            limit=args.limit,
            resume=args.resume,
        )
        stats.print_summary()

        # Print extraction metrics
        if pipeline.metrics:
            print(pipeline.metrics.summary())

            # Check for alerts
            alerts = pipeline.metrics.check_alerts()
            if alerts:
                print("\n⚠️  ALERTS:")
                for alert in alerts:
                    print(f"  {alert}")

            # Save metrics to file if requested
            if args.metrics_file:
                metrics_path = Path(args.metrics_file)
                pipeline.metrics.save_prometheus(metrics_path)
                print(f"\n📊 Metrics saved to: {metrics_path}")

        # Report final counts
        total_concepts = await ConceptStore.count()
        total_relationships = await RelationshipStore.count()
        print(f"\nDatabase totals: {total_concepts} concepts, {total_relationships} relationships")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved to checkpoint.")
        pipeline.save_checkpoint()
        # Still output partial metrics on interrupt
        if pipeline.metrics and pipeline.metrics.total_chunks > 0:
            print(pipeline.metrics.summary())
    except Exception as e:
        logger.error("extraction_failed", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
