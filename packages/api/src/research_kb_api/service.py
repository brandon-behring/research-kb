"""Shared service layer for research-kb API.

This module provides the core business logic used by both the REST API
and the socket daemon. All operations are async and use the storage layer.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from research_kb_common import get_logger
from research_kb_contracts import Source, SourceType, Concept, ConceptRelationship, ConceptType
from research_kb_pdf import EmbeddingClient
from research_kb_storage import (
    HydeConfig,
    get_hyde_embedding,
    ConceptStore,
    SourceStore,
    ChunkStore,
    RelationshipStore,
    SearchQuery,
    search_hybrid,
    search_hybrid_v2,
    search_vector_only,
    search_with_rerank,
    search_with_expansion,
    get_citing_sources,
    get_cited_sources,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class ContextType(str, Enum):
    """Context type for search weighting."""

    building = "building"  # 20% FTS, 80% vector - favor semantic breadth
    auditing = "auditing"  # 50% FTS, 50% vector - favor precision
    balanced = "balanced"  # 30% FTS, 70% vector - default


@dataclass
class SearchOptions:
    """Options for search operations."""

    query: str
    limit: int = 10
    context_type: ContextType = ContextType.balanced
    source_filter: Optional[str] = None
    use_rerank: bool = True
    use_expand: bool = True
    use_llm_expand: bool = False
    use_citations: bool = True
    citation_weight: float = 0.15
    domain_id: Optional[str] = None  # Filter by knowledge domain (None = all domains)
    fast_mode: bool = False  # Vector-only search for low latency (~200ms vs ~3s)
    hyde_config: Optional[HydeConfig] = None  # HyDE query expansion


@dataclass
class SearchResponse:
    """Rich search response with metadata."""

    query: str
    expanded_query: Optional[str] = None
    results: list[SearchResultDetail] = field(default_factory=list)
    execution_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0


@dataclass
class SearchResultDetail:
    """Detailed search result with score breakdown."""

    source: SourceSummary
    chunk: ChunkSummary
    concepts: list[str] = field(default_factory=list)
    scores: ScoreBreakdown = field(default_factory=lambda: ScoreBreakdown())
    combined_score: float = 0.0


@dataclass
class SourceSummary:
    """Source summary for API responses."""

    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    source_type: Optional[str] = None


@dataclass
class ChunkSummary:
    """Chunk summary for API responses."""

    id: str
    content: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None


@dataclass
class ScoreBreakdown:
    """Score breakdown for search results."""

    fts: float = 0.0
    vector: float = 0.0
    graph: float = 0.0
    citation: float = 0.0
    combined: float = 0.0


# Singleton embedding client with lazy loading
_embedding_client: Optional[EmbeddingClient] = None
_embedding_cache: dict[str, list[float]] = {}
_cache_lock = asyncio.Lock()
_executor = ThreadPoolExecutor(max_workers=2)  # Bounded for GPU work


def get_embedding_client() -> EmbeddingClient:
    """Get or create the embedding client (lazy loading)."""
    global _embedding_client
    if _embedding_client is None:
        logger.info("lazy_loading_embedding_client")
        _embedding_client = EmbeddingClient()
    return _embedding_client


async def get_cached_embedding(text: str) -> list[float]:
    """Get query embedding with caching (fully async, non-blocking).

    Uses embed_query() which adds the BGE query instruction prefix
    for better asymmetric retrieval (short queries finding long documents).

    Thread-safe with asyncio.Lock() and uses run_in_executor() to avoid
    blocking the event loop during CPU/GPU-bound embedding generation.
    """
    # Fast path: check cache under lock
    async with _cache_lock:
        if text in _embedding_cache:
            return _embedding_cache[text]

    # CPU/GPU-bound work runs in thread pool (doesn't block event loop)
    loop = asyncio.get_event_loop()
    client = get_embedding_client()
    embedding = await loop.run_in_executor(_executor, client.embed_query, text)

    # Store result under lock
    async with _cache_lock:
        _embedding_cache[text] = embedding
        # Keep cache bounded
        if len(_embedding_cache) > 1000:
            # Remove oldest entries
            keys = list(_embedding_cache.keys())[:500]
            for k in keys:
                del _embedding_cache[k]

    return embedding


def get_context_weights(context_type: ContextType) -> tuple[float, float]:
    """Get FTS/vector weights based on context type."""
    if context_type == ContextType.building:
        return 0.2, 0.8
    elif context_type == ContextType.auditing:
        return 0.5, 0.5
    else:  # balanced
        return 0.3, 0.7


async def search(options: SearchOptions) -> SearchResponse:
    """Execute a search with all options.

    This is the main search entry point used by both API and daemon.
    """
    start_time = time.perf_counter()
    response = SearchResponse(query=options.query)

    # Generate embedding
    embed_start = time.perf_counter()
    query_embedding = await get_cached_embedding(options.query)
    response.embedding_time_ms = (time.perf_counter() - embed_start) * 1000

    # HyDE: replace query embedding with hypothetical document embedding
    if options.hyde_config and options.hyde_config.enabled:
        hyde_embedding = await get_hyde_embedding(options.query, options.hyde_config)
        if hyde_embedding is not None:
            query_embedding = hyde_embedding
            response.expanded_query = "[HyDE-expanded]"

    # Fast mode: vector-only search (skips FTS, graph, rerank, expansion)
    # ~200ms total vs ~3s for full hybrid search
    if options.fast_mode:
        search_start = time.perf_counter()
        search_query = SearchQuery(
            text=options.query,
            embedding=query_embedding,
            fts_weight=0.0,
            vector_weight=1.0,
            limit=options.limit,
            use_graph=False,
            graph_weight=0.0,
            use_citations=False,
            citation_weight=0.0,
            domain_id=options.domain_id,
        )
        results = await search_vector_only(search_query)
        response.search_time_ms = (time.perf_counter() - search_start) * 1000

        # Convert results to response format
        for result in results:
            detail = SearchResultDetail(
                source=SourceSummary(
                    id=str(result.source.id),
                    title=result.source.title,
                    authors=result.source.authors or [],
                    year=result.source.year,
                    source_type=(
                        result.source.source_type.value if result.source.source_type else None
                    ),
                ),
                chunk=ChunkSummary(
                    id=str(result.chunk.id),
                    content=result.chunk.content,
                    page_start=result.chunk.page_start,
                    page_end=result.chunk.page_end,
                    section=(
                        result.chunk.metadata.get("section_header")
                        if result.chunk.metadata
                        else None
                    ),
                ),
                scores=ScoreBreakdown(
                    fts=0.0,
                    vector=result.vector_score or 0.0,
                    graph=0.0,
                    citation=0.0,
                    combined=result.vector_score or 0.0,
                ),
                combined_score=result.vector_score or 0.0,
            )
            response.results.append(detail)

        response.execution_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "fast_search_complete",
            query=options.query[:50],
            results=len(response.results),
            execution_ms=response.execution_time_ms,
        )
        return response

    # Get weights based on context type. Concept-graph signal retired (RS4, ADR-0001):
    # search combines FTS + vector + citation only.
    fts_weight, vector_weight = get_context_weights(options.context_type)

    search_query = SearchQuery(
        text=options.query,
        embedding=query_embedding,
        fts_weight=fts_weight,
        vector_weight=vector_weight,
        use_citations=options.use_citations,
        citation_weight=options.citation_weight,
        limit=options.limit,
        source_filter=options.source_filter,
        domain_id=options.domain_id,
    )

    # Execute search
    search_start = time.perf_counter()
    expanded_query = None

    if options.use_expand or options.use_llm_expand:
        results, expanded_query = await search_with_expansion(
            search_query,
            use_synonyms=options.use_expand,
            use_graph_expansion=False,
            use_llm_expansion=options.use_llm_expand,
            use_rerank=options.use_rerank,
            rerank_top_k=options.limit,
        )
        if expanded_query:
            response.expanded_query = (
                ", ".join(expanded_query.expanded_terms) if expanded_query.expanded_terms else None
            )
    elif options.use_rerank:
        results = await search_with_rerank(search_query, rerank_top_k=options.limit)
    elif options.use_citations:
        # Use v2 for citation authority boosting
        results = await search_hybrid_v2(search_query)
    else:
        results = await search_hybrid(search_query)

    response.search_time_ms = (time.perf_counter() - search_start) * 1000

    # Convert results to response format
    for result in results:
        detail = SearchResultDetail(
            source=SourceSummary(
                id=str(result.source.id),
                title=result.source.title,
                authors=result.source.authors or [],
                year=result.source.year,
                source_type=(
                    result.source.source_type.value if result.source.source_type else None
                ),
            ),
            chunk=ChunkSummary(
                id=str(result.chunk.id),
                content=result.chunk.content,
                page_start=result.chunk.page_start,
                page_end=result.chunk.page_end,
                section=(
                    result.chunk.metadata.get("section_header") if result.chunk.metadata else None
                ),
            ),
            scores=ScoreBreakdown(
                fts=getattr(result, "fts_score", 0.0) or 0.0,
                vector=getattr(result, "vector_score", 0.0) or 0.0,
                graph=getattr(result, "graph_score", 0.0) or 0.0,
                citation=getattr(result, "citation_score", 0.0) or 0.0,
                combined=result.combined_score,
            ),
            combined_score=result.combined_score,
        )
        response.results.append(detail)

    response.execution_time_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "search_complete",
        query=options.query[:50],
        results=len(response.results),
        execution_ms=response.execution_time_ms,
    )

    return response


async def get_sources(
    limit: int = 100,
    offset: int = 0,
    source_type: Optional[str] = None,
    domain_id: Optional[str] = None,
) -> list[Source]:
    """Get sources with optional filtering."""
    st = SourceType(source_type.lower()) if source_type else None
    return await SourceStore.list_all(
        limit=limit, offset=offset, source_type=st, domain_id=domain_id
    )


async def count_sources(
    source_type: Optional[str] = None,
    domain_id: Optional[str] = None,
) -> int:
    """Count sources with optional type and domain filters."""
    return await SourceStore.count(source_type=source_type, domain_id=domain_id)


async def get_source_by_id(source_id: str) -> Optional[Source]:
    """Get a single source by ID."""
    return await SourceStore.get_by_id(UUID(source_id))


async def get_source_chunks(source_id: str, limit: int = 100) -> list:
    """Get chunks for a source."""
    return await ChunkStore.list_by_source(UUID(source_id), limit=limit)


async def get_concepts(
    query: Optional[str] = None,
    limit: int = 100,
    concept_type: Optional[str] = None,
    domain_id: Optional[str] = None,
) -> list[Concept]:
    """Get concepts with optional search/filtering.

    Args:
        query: Optional fuzzy text query against concept names/aliases.
        limit: Max results.
        concept_type: Optional ConceptType filter.
        domain_id: Optional knowledge-domain filter (Issue #4).
    """
    ct = ConceptType(concept_type.lower()) if concept_type else None
    if query:
        return await ConceptStore.search(query, limit=limit, concept_type=ct, domain_id=domain_id)
    return await ConceptStore.list_all(limit=limit, concept_type=ct, domain_id=domain_id)


async def get_concept_by_id(concept_id: str) -> Optional[Concept]:
    """Get a single concept by ID."""
    return await ConceptStore.get(UUID(concept_id))


async def get_concept_relationships(concept_id: str) -> list[ConceptRelationship]:
    """Get relationships for a concept."""
    return await RelationshipStore.list_all_for_concept(UUID(concept_id))


async def get_stats() -> dict:
    """Get database statistics."""
    from research_kb_storage import get_connection_pool, DatabaseConfig

    pool = await get_connection_pool(DatabaseConfig())

    async with pool.acquire() as conn:
        sources = await conn.fetchval("SELECT COUNT(*) FROM sources")
        chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        concepts = await conn.fetchval("SELECT COUNT(*) FROM concepts")
        relationships = await conn.fetchval("SELECT COUNT(*) FROM concept_relationships")
        citations = await conn.fetchval("SELECT COUNT(*) FROM citations")
        chunk_concepts = await conn.fetchval("SELECT COUNT(*) FROM chunk_concepts")

    return {
        "sources": sources,
        "chunks": chunks,
        "concepts": concepts,
        "relationships": relationships,
        "citations": citations,
        "chunk_concepts": chunk_concepts,
    }


async def get_citations_for_source(source_id: str) -> dict:
    """Get citation information for a source."""
    citing = await get_citing_sources(UUID(source_id))
    cited = await get_cited_sources(UUID(source_id))

    return {
        "source_id": source_id,
        "citing_sources": [
            {"id": str(s["id"]), "title": s["title"], "year": s.get("year")} for s in citing
        ],
        "cited_sources": [
            {"id": str(s["id"]), "title": s["title"], "year": s.get("year")} for s in cited
        ],
    }
