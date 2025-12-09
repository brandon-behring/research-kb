"""Pydantic models for research-kb system.

These schemas define the contract between all packages.
They match the PostgreSQL schema defined in packages/storage/schema.sql.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Source document types."""

    TEXTBOOK = "textbook"
    PAPER = "paper"
    CODE_REPO = "code_repo"


class IngestionStage(str, Enum):
    """Ingestion pipeline state machine stages."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


# Type alias for JSONB metadata
SourceMetadata = dict[str, Any]
ChunkMetadata = dict[str, Any]


class Source(BaseModel):
    """Source document (textbook, paper, code repository).

    Matches PostgreSQL table: sources
    See: packages/storage/schema.sql:13-32
    """

    id: UUID
    source_type: SourceType
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None

    # File tracking (for idempotency)
    file_path: Optional[str] = None
    file_hash: str = Field(..., description="SHA256 hash for deduplication")

    # Extensible metadata (JSONB in PostgreSQL)
    # Examples: doi, arxiv_id, isbn, git_url, importance_tier
    metadata: SourceMetadata = Field(default_factory=dict)

    created_at: datetime
    updated_at: datetime

    @field_validator("file_hash")
    @classmethod
    def validate_file_hash(cls, v: str) -> str:
        """Ensure file_hash is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_hash must be non-empty")
        return v.strip()


class Chunk(BaseModel):
    """Content unit extracted from a source.

    Matches PostgreSQL table: chunks
    See: packages/storage/schema.sql:41-67
    """

    id: UUID
    source_id: UUID

    # Content
    content: str = Field(..., min_length=1)
    content_hash: str = Field(..., description="Hash for detecting duplicate chunks")

    # Location (for citations)
    location: Optional[str] = Field(
        None, description='Human-readable location like "Chapter 3, Section 3.4, p. 73"'
    )
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    # Semantic search (1024-dim BGE-large-en-v1.5 embeddings)
    embedding: Optional[list[float]] = Field(
        None, description="1024-dim BGE-large-en-v1.5 embedding vector"
    )

    # Extensible metadata (JSONB in PostgreSQL)
    # Examples: chunk_type, parent_chunk_id, concepts, theorem_text, flashcard
    metadata: ChunkMetadata = Field(default_factory=dict)

    created_at: datetime

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dimension(
        cls, v: Optional[list[float]]
    ) -> Optional[list[float]]:
        """Validate embedding is 1024 dimensions if provided (BGE-large-en-v1.5)."""
        if v is not None and len(v) != 1024:
            raise ValueError(
                f"embedding must be 1024 dimensions (BGE-large-en-v1.5), got {len(v)}"
            )
        return v

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure content is non-empty."""
        if not v or not v.strip():
            raise ValueError("content must be non-empty")
        return v


class IngestionStatus(BaseModel):
    """Ingestion pipeline status for a source."""

    source_id: UUID
    stage: IngestionStage
    progress: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Progress 0.0-1.0"
    )
    error_message: Optional[str] = None
    chunks_created: int = Field(default=0, ge=0)
    updated_at: datetime


class Citation(BaseModel):
    """Citation extracted from a source document.

    Matches PostgreSQL table: citations (Phase 1.5.2)
    Parsed from GROBID TEI-XML <listBibl> element.
    Used for BibTeX generation and provenance tracking.
    """

    # Storage fields (optional for extraction, required for storage)
    id: Optional[UUID] = None
    source_id: Optional[UUID] = None

    # Core citation metadata
    authors: list[str] = Field(default_factory=list)
    title: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = Field(None, description="Journal, conference, or publisher")
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    raw_string: str = Field(..., description="Original citation text from document")

    # BibTeX and extraction metadata
    bibtex: Optional[str] = Field(None, description="Generated BibTeX entry")
    extraction_method: Optional[str] = Field(None, description="grobid or manual")
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Extraction confidence 0.0-1.0"
    )

    # Extensible metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Timestamp (set by storage)
    created_at: Optional[datetime] = None

    def to_bibtex_key(self) -> str:
        """Generate a BibTeX citation key.

        Format: firstauthor_year_firstword
        Example: pearl2009causality
        """
        first_author = (
            self.authors[0].split()[-1].lower() if self.authors else "unknown"
        )
        year_str = str(self.year) if self.year else "0000"
        first_word = self.title.split()[0].lower() if self.title else "untitled"
        # Remove non-alphanumeric chars
        first_word = "".join(c for c in first_word if c.isalnum())
        return f"{first_author}{year_str}{first_word}"


class SearchResult(BaseModel):
    """Hybrid search result combining FTS and vector search.

    Returned by search package after orchestrating storage queries.
    """

    chunk: Chunk
    source: Source

    # Scores
    fts_score: Optional[float] = Field(
        None, description="Full-text search ts_rank score"
    )
    vector_score: Optional[float] = Field(
        None, description="Vector cosine similarity (1=identical, 0=opposite)"
    )
    graph_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Graph-based relevance score (0-1, higher=better)",
    )
    citation_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Citation authority score (PageRank-style, 0-1, higher=more authoritative)",
    )
    rerank_score: Optional[float] = Field(
        None,
        description="Cross-encoder reranking score (Phase 3, higher=better)",
    )
    combined_score: float = Field(
        ..., description="Weighted combination of FTS + vector + graph + citation scores"
    )

    # Ranking
    rank: int = Field(..., ge=1, description="1-based rank in result set")

    @field_validator("combined_score")
    @classmethod
    def validate_combined_score(cls, v: float) -> float:
        """Ensure combined_score is non-negative."""
        if v < 0:
            raise ValueError("combined_score must be non-negative")
        return v


class ConceptType(str, Enum):
    """Concept classification types."""

    METHOD = "method"
    ASSUMPTION = "assumption"
    PROBLEM = "problem"
    DEFINITION = "definition"
    THEOREM = "theorem"


class RelationshipType(str, Enum):
    """Concept relationship types."""

    REQUIRES = "REQUIRES"  # Method requires assumption
    USES = "USES"  # Method uses technique
    ADDRESSES = "ADDRESSES"  # Method solves problem
    GENERALIZES = "GENERALIZES"  # Broader concept
    SPECIALIZES = "SPECIALIZES"  # Narrower concept
    ALTERNATIVE_TO = "ALTERNATIVE_TO"  # Competing approaches
    EXTENDS = "EXTENDS"  # Builds upon


class Concept(BaseModel):
    """Knowledge entity extracted from research documents.

    Matches PostgreSQL table: concepts (Phase 2)
    See: packages/storage/migrations/002_knowledge_graph.sql
    """

    id: UUID
    name: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    concept_type: ConceptType
    category: Optional[str] = None  # identification, estimation, testing
    definition: Optional[str] = None

    # Semantic search
    embedding: Optional[list[float]] = Field(
        None, description="1024-dim BGE-large-en-v1.5 embedding"
    )

    # Extraction metadata
    extraction_method: Optional[str] = None  # "ollama:llama3.1:8b", "manual"
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Extraction confidence 0.0-1.0"
    )
    validated: bool = False

    # Extensibility
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dimension(
        cls, v: Optional[list[float]]
    ) -> Optional[list[float]]:
        """Validate embedding is 1024 dimensions if provided."""
        if v is not None and len(v) != 1024:
            raise ValueError(f"embedding must be 1024 dimensions, got {len(v)}")
        return v


class ConceptRelationship(BaseModel):
    """Directed edge between concepts in knowledge graph.

    Matches PostgreSQL table: concept_relationships (Phase 2)
    """

    id: UUID
    source_concept_id: UUID
    target_concept_id: UUID
    relationship_type: RelationshipType
    is_directed: bool = True
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_chunk_ids: list[UUID] = Field(default_factory=list)
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Extraction confidence 0.0-1.0"
    )
    created_at: datetime


class ChunkConcept(BaseModel):
    """Link between chunk and concept (many-to-many junction).

    Matches PostgreSQL table: chunk_concepts (Phase 2)
    """

    chunk_id: UUID
    concept_id: UUID
    mention_type: str = Field(
        default="reference",
        description="How concept appears: defines, reference, example",
    )
    relevance_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Relevance of concept to chunk"
    )
    created_at: datetime


class Method(BaseModel):
    """Specialized attributes for method-type concepts.

    Matches PostgreSQL table: methods (Phase 2 migration 002)
    """

    id: UUID
    concept_id: UUID
    required_assumptions: list[str] = Field(
        default_factory=list,
        description="List of assumption concept names this method requires",
    )
    problem_types: list[str] = Field(
        default_factory=list,
        description="Problem types this method addresses (ATE, ATT, LATE, CATE, etc.)",
    )
    common_estimators: list[str] = Field(
        default_factory=list,
        description="Common estimators used (OLS, 2SLS, matching, etc.)",
    )


class Assumption(BaseModel):
    """Specialized attributes for assumption-type concepts.

    Matches PostgreSQL table: assumptions (Phase 2 migration 002)
    """

    id: UUID
    concept_id: UUID
    mathematical_statement: Optional[str] = Field(
        None, description="Formal mathematical statement of the assumption"
    )
    is_testable: Optional[bool] = Field(
        None, description="Whether this assumption can be empirically tested"
    )
    common_tests: list[str] = Field(
        default_factory=list,
        description="Common tests for this assumption (Hausman, Durbin-Wu-Hausman, etc.)",
    )
    violation_consequences: Optional[str] = Field(
        None, description="Consequences of violating this assumption"
    )
