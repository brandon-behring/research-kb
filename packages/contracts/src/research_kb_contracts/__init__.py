"""Research KB Contracts - Pure Pydantic schemas.

Version: 1.0.0 (frozen - breaking changes require new package)

This package contains ONLY Pydantic schemas with no business logic.
Dependencies: pydantic only (no OpenTelemetry, no logging, no DB drivers).
"""

from research_kb_contracts.models import (
    # Core entities
    Chunk,
    ChunkMetadata,
    Citation,
    Source,
    SourceMetadata,
    SourceType,
    # Knowledge graph (Phase 2)
    Concept,
    ConceptRelationship,
    ConceptType,
    ChunkConcept,
    RelationshipType,
    Method,
    Assumption,
    # Ingestion
    IngestionStage,
    IngestionStatus,
    # Search
    SearchResult,
)

__version__ = "1.0.0"

__all__ = [
    # Core entities
    "Chunk",
    "ChunkMetadata",
    "Citation",
    "Source",
    "SourceMetadata",
    "SourceType",
    # Knowledge graph (Phase 2)
    "Concept",
    "ConceptRelationship",
    "ConceptType",
    "ChunkConcept",
    "RelationshipType",
    "Method",
    "Assumption",
    # Ingestion
    "IngestionStage",
    "IngestionStatus",
    # Search
    "SearchResult",
]
