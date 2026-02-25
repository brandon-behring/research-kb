"""Pydantic schemas for API request/response models.

These models define the API contract for all endpoints.
Rich nested responses with full metadata per user decision.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

# === Enums ===


class ContextType(str, Enum):
    """Context type for search weighting."""

    building = "building"
    auditing = "auditing"
    balanced = "balanced"


class ConceptType(str, Enum):
    """Concept types in the knowledge graph."""

    method = "method"
    assumption = "assumption"
    problem = "problem"
    definition = "definition"
    theorem = "theorem"
    concept = "concept"
    principle = "principle"
    technique = "technique"
    model = "model"


class RelationshipType(str, Enum):
    """Relationship types between concepts."""

    requires = "REQUIRES"
    uses = "USES"
    addresses = "ADDRESSES"
    generalizes = "GENERALIZES"
    specializes = "SPECIALIZES"
    alternative_to = "ALTERNATIVE_TO"
    extends = "EXTENDS"


# === Request Models ===


class SearchRequest(BaseModel):
    """Search request body."""

    query: str = Field(..., description="Search query text", min_length=1)
    limit: int = Field(10, ge=1, le=100, description="Maximum results")
    context_type: ContextType = Field(
        ContextType.balanced,
        description="Context mode for search weighting",
    )
    source_filter: Optional[str] = Field(
        None,
        description="Filter by source type (PAPER, TEXTBOOK, etc.)",
    )
    use_graph: bool = Field(True, description="Enable graph-boosted search")
    graph_weight: float = Field(0.2, ge=0, le=1, description="Graph score weight")
    use_rerank: bool = Field(True, description="Enable cross-encoder reranking")
    use_expand: bool = Field(True, description="Enable query expansion")
    use_citations: bool = Field(True, description="Enable citation authority boosting")
    citation_weight: float = Field(0.15, ge=0, le=1, description="Citation weight (0-1)")


# === Response Models ===


class ScoreBreakdown(BaseModel):
    """Score breakdown for search results."""

    fts: float = Field(0.0, description="Full-text search score")
    vector: float = Field(0.0, description="Vector similarity score")
    graph: float = Field(0.0, description="Knowledge graph score")
    citation: float = Field(0.0, description="Citation authority score")
    combined: float = Field(0.0, description="Final combined score")


class SourceSummary(BaseModel):
    """Source summary in search results."""

    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    source_type: Optional[str] = None


class ChunkSummary(BaseModel):
    """Chunk summary in search results."""

    id: str
    content: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None


class SearchResultItem(BaseModel):
    """Individual search result with full metadata."""

    source: SourceSummary
    chunk: ChunkSummary
    concepts: list[str] = Field(default_factory=list)
    scores: ScoreBreakdown
    combined_score: float


class SearchResponse(BaseModel):
    """Search response with results and metadata."""

    query: str
    expanded_query: Optional[str] = None
    results: list[SearchResultItem]
    metadata: SearchMetadata


class SearchMetadata(BaseModel):
    """Execution metadata for search."""

    execution_time_ms: float
    embedding_time_ms: float
    search_time_ms: float
    result_count: int


# === Source Models ===


class SourceDetail(BaseModel):
    """Full source details."""

    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    source_type: Optional[str] = None
    file_path: Optional[str] = None
    abstract: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: Optional[str] = None


class SourceListResponse(BaseModel):
    """Paginated source list."""

    sources: list[SourceDetail]
    total: int
    limit: int
    offset: int


class ChunkDetail(BaseModel):
    """Full chunk details."""

    id: str
    content: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Optional[dict] = None


class SourceWithChunks(BaseModel):
    """Source with its chunks."""

    source: SourceDetail
    chunks: list[ChunkDetail]
    chunk_count: int


# === Concept Models ===


class ConceptDetail(BaseModel):
    """Full concept details."""

    id: str
    name: str
    canonical_name: str
    concept_type: Optional[ConceptType] = None
    definition: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)


class ConceptListResponse(BaseModel):
    """Concept list response."""

    concepts: list[ConceptDetail]
    total: int


class RelationshipDetail(BaseModel):
    """Concept relationship details."""

    id: str
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    relationship_type: RelationshipType
    confidence: Optional[float] = None


class ConceptWithRelationships(BaseModel):
    """Concept with its relationships."""

    concept: ConceptDetail
    relationships: list[RelationshipDetail]


# === Graph Models ===


class GraphNode(BaseModel):
    """Node in the concept graph."""

    id: str
    name: str
    type: Optional[str] = None


class GraphEdge(BaseModel):
    """Edge in the concept graph."""

    source: str
    target: str
    type: Optional[str] = None


class GraphNeighborhood(BaseModel):
    """Graph neighborhood response."""

    center: GraphNode
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class GraphPath(BaseModel):
    """Path between two concepts."""

    from_concept: str
    to_concept: str
    path: list[GraphNode]
    path_length: int


# === Citation Models ===


class CitationSummary(BaseModel):
    """Citation summary."""

    id: str
    title: str
    year: Optional[int] = None


class SourceCitations(BaseModel):
    """Citation information for a source."""

    source_id: str
    citing_sources: list[CitationSummary]
    cited_sources: list[CitationSummary]
    citation_count: int
    reference_count: int


# === Stats Models ===


class DatabaseStats(BaseModel):
    """Database statistics."""

    sources: int
    chunks: int
    concepts: int
    relationships: int
    citations: int
    chunk_concepts: int


# === Health Models ===


class HealthCheck(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    database: str = "connected"
    embedding_model: str = "ready"


class HealthDetail(BaseModel):
    """Detailed health check."""

    status: str
    version: str
    components: dict[str, str]
    stats: Optional[DatabaseStats] = None


# === Domain Models ===


class DomainStats(BaseModel):
    """Statistics for a knowledge domain."""

    domain_id: str = Field(..., description="Domain identifier")
    name: str = Field(..., description="Human-readable domain name")
    source_count: int = Field(0, description="Number of sources in domain")
    chunk_count: int = Field(0, description="Number of chunks in domain")
    concept_count: int = Field(0, description="Number of concepts in domain")


class DomainListResponse(BaseModel):
    """List of all knowledge domains."""

    domains: list[DomainStats]
    total: int
