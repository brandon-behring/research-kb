#!/usr/bin/env python3
"""Extract concepts and relationships from ingested chunks.

This script:
1. Loads chunks from the database (with optional filtering)
2. Extracts concepts using Ollama LLM
3. Deduplicates concepts by canonical name
4. Stores concepts, relationships, and chunk-concept links
5. Syncs to Neo4j for graph queries
6. Reports extraction statistics

Usage:
    # Process all chunks
    python scripts/extract_concepts.py

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
import sys
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
    GraphSyncService,
    OllamaClient,
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
        ollama_model: str = "llama3.1:8b",
        confidence_threshold: float = 0.7,
        batch_size: int = 10,
        checkpoint_interval: int = 50,
        dry_run: bool = False,
        sync_neo4j: bool = True,
    ):
        self.ollama_model = ollama_model
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.dry_run = dry_run
        self.sync_neo4j = sync_neo4j

        self.ollama_client: Optional[OllamaClient] = None
        self.extractor: Optional[ConceptExtractor] = None
        self.deduplicator: Optional[Deduplicator] = None
        self.graph_sync: Optional[GraphSyncService] = None

        self.stats = ExtractionStats()
        self.processed_chunk_ids: set[UUID] = set()

        # Concept name -> UUID cache
        self._concept_cache: dict[str, UUID] = {}

    async def initialize(self) -> None:
        """Initialize clients and load existing concepts."""
        # Initialize database pool
        await get_connection_pool()

        # Initialize Ollama client
        self.ollama_client = OllamaClient(
            model=self.ollama_model,
            temperature=0.1,
        )

        # Check Ollama availability
        if not await self.ollama_client.is_available():
            raise RuntimeError("Ollama server not available. Start with: ollama serve")

        if not await self.ollama_client.is_model_loaded():
            raise RuntimeError(f"Model {self.ollama_model} not loaded. Pull with: ollama pull {self.ollama_model}")

        logger.info("ollama_connected", model=self.ollama_model)

        # Initialize extractor and deduplicator
        self.deduplicator = Deduplicator()
        self.extractor = ConceptExtractor(
            ollama_client=self.ollama_client,
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
        """Get chunks that need processing."""
        # Get all chunks
        if source_id:
            chunks = await ChunkStore.list_by_source(source_id, limit=limit or 10000)
        else:
            chunks = await ChunkStore.list_all(limit=limit or 10000)

        # Filter out already processed if resuming
        if resume:
            checkpoint_ids = self.load_checkpoint()
            chunks = [c for c in chunks if c.id not in checkpoint_ids]
            self.processed_chunk_ids = checkpoint_ids
            logger.info("resume_from_checkpoint", skipped=len(checkpoint_ids), remaining=len(chunks))

        return chunks

    async def process_chunk(self, chunk: Chunk) -> Optional[ChunkExtraction]:
        """Extract concepts from a single chunk."""
        try:
            extraction = await self.extractor.extract_from_chunk(chunk)
            return extraction
        except Exception as e:
            logger.error("chunk_extraction_failed", chunk_id=str(chunk.id), error=str(e))
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
                        extraction_method=f"ollama:{self.ollama_model}",
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

    async def run(
        self,
        source_id: Optional[UUID] = None,
        limit: Optional[int] = None,
        resume: bool = False,
    ) -> ExtractionStats:
        """Run the extraction pipeline."""
        await self.initialize()

        try:
            chunks = await self.get_chunks_to_process(source_id, limit, resume)
            total = len(chunks)

            if total == 0:
                logger.info("no_chunks_to_process")
                return self.stats

            logger.info("extraction_started", total_chunks=total, dry_run=self.dry_run)

            for i, chunk in enumerate(chunks):
                # Progress logging
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"\rProcessing chunk {i + 1}/{total} ({100 * (i + 1) / total:.1f}%)...", end="", flush=True)

                # Skip very short chunks
                if len(chunk.content) < 100:
                    self.stats.chunks_skipped += 1
                    continue

                # Extract concepts
                extraction = await self.process_chunk(chunk)

                if extraction is None:
                    self.stats.chunks_failed += 1
                    continue

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

                # Checkpoint periodically
                if (i + 1) % self.checkpoint_interval == 0:
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
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Ollama model to use")
    parser.add_argument("--confidence", type=float, default=0.7, help="Minimum confidence threshold")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--dry-run", action="store_true", help="Run without storing results")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--clear-checkpoint", action="store_true", help="Clear checkpoint and start fresh")
    parser.add_argument("--no-neo4j", action="store_true", help="Disable Neo4j sync")

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
        ollama_model=args.model,
        confidence_threshold=args.confidence,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        sync_neo4j=not args.no_neo4j,
    )

    try:
        stats = await pipeline.run(
            source_id=source_id,
            limit=args.limit,
            resume=args.resume,
        )
        stats.print_summary()

        # Report final counts
        total_concepts = await ConceptStore.count()
        total_relationships = await RelationshipStore.count()
        print(f"\nDatabase totals: {total_concepts} concepts, {total_relationships} relationships")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved to checkpoint.")
        pipeline.save_checkpoint()
    except Exception as e:
        logger.error("extraction_failed", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
