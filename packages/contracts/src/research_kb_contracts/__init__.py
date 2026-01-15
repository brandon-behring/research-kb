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
    # Multi-domain support (Migration 010)
    Domain,
    CrossDomainLink,
    CrossDomainLinkType,
    # Ingestion
    IngestionStage,
    IngestionStatus,
    # Search
    SearchResult,
)

__version__ = "1.1.0"  # Bumped for multi-domain support

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
    # Multi-domain support (Migration 010)
    "Domain",
    "CrossDomainLink",
    "CrossDomainLinkType",
    # Ingestion
    "IngestionStage",
    "IngestionStatus",
    # Search
    "SearchResult",
]
