"""Search endpoints.

Provides hybrid search combining FTS, vector similarity, and knowledge graph signals.
"""

from __future__ import annotations

from fastapi import APIRouter

from research_kb_api import schemas
from research_kb_api import service

router = APIRouter()


@router.post("", response_model=schemas.SearchResponse)
async def search(request: schemas.SearchRequest) -> schemas.SearchResponse:
    """Execute hybrid search.

    Combines full-text search, vector similarity, and optionally knowledge graph
    signals. Supports context-aware weighting for building vs auditing use cases.

    Parameters
    ----------
    request : SearchRequest
        - query: Search text
        - limit: Max results (1-100, default 10)
        - context_type: building/auditing/balanced (affects FTS/vector weights)
        - source_filter: Optional filter by source type
        - use_graph: Enable graph-boosted search
        - graph_weight: Weight for graph signal (0-1)
        - use_rerank: Enable cross-encoder reranking
        - use_expand: Enable query expansion

    Returns
    -------
    SearchResponse
        Results with full metadata, score breakdowns, and execution timing.
    """
    # Map schema context type to service context type
    context_map = {
        schemas.ContextType.building: service.ContextType.building,
        schemas.ContextType.auditing: service.ContextType.auditing,
        schemas.ContextType.balanced: service.ContextType.balanced,
    }

    options = service.SearchOptions(
        query=request.query,
        limit=request.limit,
        context_type=context_map[request.context_type],
        source_filter=request.source_filter,
        use_graph=request.use_graph,
        graph_weight=request.graph_weight,
        use_rerank=request.use_rerank,
        use_expand=request.use_expand,
        use_citations=request.use_citations,
        citation_weight=request.citation_weight,
    )

    response = await service.search(options)

    # Convert service response to API schema
    results = [
        schemas.SearchResultItem(
            source=schemas.SourceSummary(
                id=r.source.id,
                title=r.source.title,
                authors=r.source.authors,
                year=r.source.year,
                source_type=r.source.source_type,
            ),
            chunk=schemas.ChunkSummary(
                id=r.chunk.id,
                content=r.chunk.content,
                page_start=r.chunk.page_start,
                page_end=r.chunk.page_end,
                section=r.chunk.section,
            ),
            concepts=r.concepts,
            scores=schemas.ScoreBreakdown(
                fts=r.scores.fts,
                vector=r.scores.vector,
                graph=r.scores.graph,
                citation=r.scores.citation,
                combined=r.scores.combined,
            ),
            combined_score=r.combined_score,
        )
        for r in response.results
    ]

    return schemas.SearchResponse(
        query=response.query,
        expanded_query=response.expanded_query,
        results=results,
        metadata=schemas.SearchMetadata(
            execution_time_ms=response.execution_time_ms,
            embedding_time_ms=response.embedding_time_ms,
            search_time_ms=response.search_time_ms,
            result_count=len(results),
        ),
    )
