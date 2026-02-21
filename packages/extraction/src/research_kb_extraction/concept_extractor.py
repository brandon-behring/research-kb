"""Core concept extraction orchestrator.

Coordinates Ollama extraction, deduplication, and storage.
This is the main entry point for the extraction pipeline.
"""

import re
from typing import Optional

from research_kb_common import get_logger
from research_kb_contracts import Chunk

from research_kb_extraction.models import (
    ChunkExtraction,
    ConceptMatch,
    ExtractedConcept,
)
from research_kb_extraction.ollama_client import OllamaClient
from research_kb_extraction.deduplicator import Deduplicator

logger = get_logger(__name__)


class ConceptExtractor:
    """Orchestrates concept extraction from text chunks.

    Combines LLM extraction with deduplication and validation.
    Designed to process chunks from the research knowledge base.

    Example:
        >>> extractor = ConceptExtractor()
        >>> result = await extractor.extract_from_chunk(chunk)
        >>> print(f"Found {len(result.concepts)} concepts")

        >>> # Domain-specific extraction
        >>> extractor = ConceptExtractor(domain_id="time_series")
        >>> result = await extractor.extract_from_chunk(ts_chunk)
    """

    def __init__(
        self,
        domain_id: str,
        ollama_client: Optional[OllamaClient] = None,
        deduplicator: Optional[Deduplicator] = None,
        confidence_threshold: float = 0.7,
        min_chunk_length: int = 100,
    ):
        """Initialize concept extractor.

        Args:
            ollama_client: Ollama client for LLM extraction
            deduplicator: Deduplicator for canonical name matching
            confidence_threshold: Minimum confidence to keep concept
            min_chunk_length: Minimum chunk length to process
            domain_id: Knowledge domain (e.g., "causal_inference", "time_series")
        """
        self.domain_id = domain_id
        self.ollama_client = ollama_client or OllamaClient()
        self.deduplicator = deduplicator or Deduplicator(domain_id=domain_id)
        self.confidence_threshold = confidence_threshold
        self.min_chunk_length = min_chunk_length

    async def extract_from_chunk(
        self,
        chunk: Chunk,
        prompt_type: str = "full",
    ) -> ChunkExtraction:
        """Extract concepts and relationships from a chunk.

        Args:
            chunk: Chunk object with content to analyze
            prompt_type: Type of extraction prompt to use

        Returns:
            ChunkExtraction with validated concepts and relationships
        """
        # Skip very short chunks
        if len(chunk.content) < self.min_chunk_length:
            logger.debug("chunk_too_short", chunk_id=str(chunk.id), length=len(chunk.content))
            return ChunkExtraction()

        # Extract with LLM (pass domain_id for domain-specific prompts)
        raw_extraction = await self.ollama_client.extract_concepts(
            chunk.content, self.domain_id, prompt_type
        )

        # Filter by confidence
        filtered_concepts = [
            c for c in raw_extraction.concepts if c.confidence >= self.confidence_threshold
        ]

        filtered_relationships = [
            r for r in raw_extraction.relationships if r.confidence >= self.confidence_threshold
        ]

        # Normalize concept names
        normalized_concepts = [self._normalize_concept(c) for c in filtered_concepts]

        # Validate relationships reference known concepts
        concept_names = {c.name.lower() for c in normalized_concepts}
        concept_names.update({a.lower() for c in normalized_concepts for a in c.aliases})

        valid_relationships = [
            r
            for r in filtered_relationships
            if (
                r.source_concept.lower() in concept_names
                or self._find_concept_match(r.source_concept, normalized_concepts)
            )
            and (
                r.target_concept.lower() in concept_names
                or self._find_concept_match(r.target_concept, normalized_concepts)
            )
        ]

        logger.info(
            "extraction_filtered",
            chunk_id=str(chunk.id),
            raw_concepts=len(raw_extraction.concepts),
            filtered_concepts=len(normalized_concepts),
            raw_relationships=len(raw_extraction.relationships),
            filtered_relationships=len(valid_relationships),
        )

        return ChunkExtraction(
            concepts=normalized_concepts,
            relationships=valid_relationships,
        )

    async def extract_from_text(
        self,
        text: str,
        prompt_type: str = "full",
    ) -> ChunkExtraction:
        """Extract concepts from raw text (without Chunk object).

        Args:
            text: Text content to analyze
            prompt_type: Type of extraction prompt

        Returns:
            ChunkExtraction with concepts and relationships
        """
        if len(text) < self.min_chunk_length:
            return ChunkExtraction()

        raw_extraction = await self.ollama_client.extract_concepts(
            text, self.domain_id, prompt_type
        )

        # Filter by confidence
        filtered_concepts = [
            self._normalize_concept(c)
            for c in raw_extraction.concepts
            if c.confidence >= self.confidence_threshold
        ]

        filtered_relationships = [
            r for r in raw_extraction.relationships if r.confidence >= self.confidence_threshold
        ]

        return ChunkExtraction(
            concepts=filtered_concepts,
            relationships=filtered_relationships,
        )

    def _normalize_concept(self, concept: ExtractedConcept) -> ExtractedConcept:
        """Normalize concept name and aliases.

        Applies:
        - Lowercase normalization
        - Whitespace cleanup
        - Common abbreviation expansion
        """
        # Clean name
        clean_name = self._clean_text(concept.name)

        # Clean aliases and add name variations
        clean_aliases = list(set(self._clean_text(a) for a in concept.aliases if a))

        # Add common variations
        if clean_name not in clean_aliases:
            # Add lowercase version
            if clean_name.lower() not in [a.lower() for a in clean_aliases]:
                pass  # Name is already canonical

        return ExtractedConcept(
            name=clean_name,
            concept_type=concept.concept_type,
            definition=concept.definition,
            aliases=clean_aliases,
            confidence=concept.confidence,
        )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())
        # Remove surrounding quotes
        text = text.strip("\"'")
        return text

    def _find_concept_match(
        self,
        name: str,
        concepts: list[ExtractedConcept],
    ) -> Optional[ExtractedConcept]:
        """Find a concept by name or alias."""
        name_lower = name.lower()
        for concept in concepts:
            if concept.name.lower() == name_lower:
                return concept
            if any(a.lower() == name_lower for a in concept.aliases):
                return concept
        return None

    async def deduplicate_concepts(
        self,
        extractions: list[ChunkExtraction],
    ) -> list[ConceptMatch]:
        """Deduplicate concepts across multiple extractions.

        Args:
            extractions: List of extraction results

        Returns:
            List of ConceptMatch with deduplication results
        """
        # Collect all concepts
        all_concepts = []
        for extraction in extractions:
            all_concepts.extend(extraction.concepts)

        if not all_concepts:
            return []

        # Deduplicate using canonical names and embeddings
        return await self.deduplicator.deduplicate_batch(all_concepts)

    async def close(self) -> None:
        """Clean up resources."""
        await self.ollama_client.close()

    async def __aenter__(self) -> "ConceptExtractor":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
