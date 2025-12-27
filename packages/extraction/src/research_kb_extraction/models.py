"""Pydantic models for concept extraction.

These models define the structured output format for Ollama extraction.
They are designed to work with Ollama's JSON mode for reliable parsing.
"""

from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ExtractedConcept(BaseModel):
    """A concept extracted from text by the LLM.

    Attributes:
        name: The concept name as it appears in text
        concept_type: Classification of the concept
        definition: Brief definition if provided in text
        aliases: Alternative names or abbreviations
        confidence: LLM's confidence in the extraction (0.0-1.0)
    """

    name: str = Field(..., description="Concept name as it appears in text")
    concept_type: Literal[
        "method", "assumption", "problem", "definition", "theorem",
        "concept", "principle", "technique", "model"
    ] = Field(..., description="Classification of the concept")
    definition: Optional[str] = Field(
        None, description="Brief definition if provided in the text"
    )
    aliases: list[str] = Field(
        default_factory=list, description="Alternative names or abbreviations"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in extraction"
    )


class ExtractedRelationship(BaseModel):
    """A relationship between two concepts extracted from text.

    Attributes:
        source_concept: Name of the source concept
        target_concept: Name of the target concept
        relationship_type: Type of relationship (must match schema)
        evidence: Text snippet supporting the relationship
        confidence: LLM's confidence in the extraction
    """

    source_concept: str = Field(..., description="Source concept name")
    target_concept: str = Field(..., description="Target concept name")
    relationship_type: Literal[
        "REQUIRES",
        "USES",
        "ADDRESSES",
        "GENERALIZES",
        "SPECIALIZES",
        "ALTERNATIVE_TO",
        "EXTENDS",
    ] = Field(..., description="Relationship type from ontology")
    evidence: Optional[str] = Field(
        None, description="Text snippet supporting this relationship"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence in relationship"
    )


class ChunkExtraction(BaseModel):
    """Complete extraction result for a single chunk.

    This is the top-level output from the concept extraction pipeline.
    Contains all concepts and relationships found in a text chunk.
    """

    concepts: list[ExtractedConcept] = Field(
        default_factory=list, description="Concepts found in chunk"
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list, description="Relationships between concepts"
    )

    @property
    def concept_count(self) -> int:
        """Number of concepts extracted."""
        return len(self.concepts)

    @property
    def relationship_count(self) -> int:
        """Number of relationships extracted."""
        return len(self.relationships)

    def get_concepts_by_type(self, concept_type: str) -> list[ExtractedConcept]:
        """Filter concepts by type."""
        return [c for c in self.concepts if c.concept_type == concept_type]

    def get_high_confidence_concepts(
        self, threshold: float = 0.7
    ) -> list[ExtractedConcept]:
        """Get concepts above confidence threshold."""
        return [c for c in self.concepts if c.confidence >= threshold]


class StoredConcept(BaseModel):
    """A concept after storage with database ID.

    Extends ExtractedConcept with storage metadata.
    """

    id: UUID
    name: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    concept_type: str
    category: Optional[str] = None
    definition: Optional[str] = None
    embedding: Optional[list[float]] = None
    extraction_method: Optional[str] = None
    confidence_score: Optional[float] = None
    validated: bool = False


class ConceptMatch(BaseModel):
    """Result of concept deduplication/matching.

    Used when checking if an extracted concept matches an existing one.
    """

    extracted: ExtractedConcept
    matched_concept_id: Optional[UUID] = None
    matched_canonical_name: Optional[str] = None
    similarity_score: float = 0.0
    is_new: bool = True

    @property
    def should_merge(self) -> bool:
        """Whether to merge with existing concept (similarity > 0.95)."""
        return self.similarity_score > 0.95 and not self.is_new
