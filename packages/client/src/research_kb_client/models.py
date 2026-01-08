"""Pydantic models for research-kb client responses.

Provides typed response models for daemon and CLI queries.
"""

from typing import Optional

from pydantic import BaseModel, Field


class SearchResultChunk(BaseModel):
    """Text chunk from search result."""

    id: str = Field(description="Chunk UUID")
    content: str = Field(description="Text content (may be truncated)")
    page_number: Optional[int] = Field(default=None, description="Page in source document")
    section_header: Optional[str] = Field(default=None, description="Section heading")


class SearchResultSource(BaseModel):
    """Source document metadata."""

    id: str = Field(description="Source UUID")
    title: str = Field(description="Document title")
    authors: Optional[list[str]] = Field(default=None, description="Author names")
    year: Optional[int] = Field(default=None, description="Publication year")
    source_type: str = Field(default="unknown", description="Type: paper, textbook, etc.")


class SearchResult(BaseModel):
    """Single search result with scores and metadata."""

    source: SearchResultSource = Field(description="Source document")
    chunk: SearchResultChunk = Field(description="Matching text chunk")
    score: float = Field(description="Combined relevance score (0-1)")
    fts_score: Optional[float] = Field(default=None, description="Full-text search score")
    vector_score: Optional[float] = Field(default=None, description="Vector similarity score")
    graph_score: Optional[float] = Field(default=None, description="Knowledge graph boost")
    citation_score: Optional[float] = Field(default=None, description="Citation authority score")
    concepts: list[str] = Field(default_factory=list, description="Related concepts")


class SearchResponse(BaseModel):
    """Response from search query."""

    results: list[SearchResult] = Field(default_factory=list)
    query: str = Field(description="Original query")
    expanded_query: Optional[str] = Field(default=None, description="Query after expansion")
    total_count: Optional[int] = Field(default=None, description="Total matches (if known)")


class HealthStatus(BaseModel):
    """Health check response."""

    status: str = Field(description="healthy or unhealthy")
    database: str = Field(default="unknown", description="Database status")
    embed_server: str = Field(default="unknown", description="Embedding server status")
    rerank_server: Optional[str] = Field(default=None, description="Reranking server status")
    uptime_seconds: Optional[float] = Field(default=None, description="Daemon uptime")


class StatsResponse(BaseModel):
    """Database statistics response."""

    sources: int = Field(description="Total sources")
    chunks: int = Field(description="Total chunks")
    concepts: int = Field(description="Total concepts")
    relationships: int = Field(description="Total concept relationships")
    citations: int = Field(default=0, description="Total citation links")
    chunk_concepts: int = Field(default=0, description="Chunk-concept mappings")


class ConceptInfo(BaseModel):
    """Concept from knowledge graph."""

    id: str = Field(description="Concept UUID")
    name: str = Field(description="Canonical name")
    concept_type: str = Field(description="Type: METHOD, ASSUMPTION, etc.")
    description: Optional[str] = Field(default=None)


class ConceptRelationship(BaseModel):
    """Relationship between concepts."""

    source_id: str = Field(description="Source concept UUID")
    target_id: str = Field(description="Target concept UUID")
    relationship_type: str = Field(description="Type: REQUIRES, USES, etc.")
    source_name: Optional[str] = Field(default=None)
    target_name: Optional[str] = Field(default=None)


class ConceptNeighborhood(BaseModel):
    """Graph neighborhood around a concept."""

    center: ConceptInfo = Field(description="Center concept")
    connected_concepts: list[ConceptInfo] = Field(default_factory=list)
    relationships: list[ConceptRelationship] = Field(default_factory=list)
    hops: int = Field(description="Number of hops explored")
