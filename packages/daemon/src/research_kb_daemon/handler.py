"""JSON-RPC method handlers for daemon.

Implements:
- search: Hybrid search with optional graph/citation boosting
- health: System health check
- stats: Database statistics
"""

import time
from typing import Any, Optional

from research_kb_common import get_logger
from research_kb_contracts import SearchResult
from research_kb_storage import (
    SearchQuery,
    search_hybrid,
    search_hybrid_v2,
    SourceStore,
    ChunkStore,
    ConceptStore,
)

from research_kb_daemon.pool import get_pool, get_embed_client

logger = get_logger(__name__)

# Track daemon start time
_start_time = time.time()


def _result_to_dict(result: SearchResult) -> dict[str, Any]:
    """Convert SearchResult to JSON-serializable dict.

    Args:
        result: SearchResult instance

    Returns:
        Dict representation
    """
    return {
        "chunk_id": str(result.chunk.id) if result.chunk else None,
        "source_id": str(result.source.id) if result.source else None,
        "content": result.chunk.content if result.chunk else None,
        "source_title": result.source.title if result.source else None,
        "source_authors": result.source.authors if result.source else None,
        "source_year": result.source.year if result.source else None,
        "page_number": result.chunk.metadata.get("page_number") if result.chunk else None,
        "section_header": result.chunk.metadata.get("section_header") if result.chunk else None,
        "fts_score": result.fts_score,
        "vector_score": result.vector_score,
        "graph_score": result.graph_score,
        "citation_score": result.citation_score,
        "combined_score": result.combined_score,
    }


async def handle_search(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Handle search method.

    Args:
        params: Search parameters
            - query (str): Search query text (required)
            - limit (int): Max results (default: 10)
            - context_type (str): Weight preset - "building", "auditing", "balanced"
            - use_graph (bool): Enable graph boosting
            - use_citations (bool): Enable citation authority
            - graph_weight (float): Graph weight if use_graph
            - citation_weight (float): Citation weight if use_citations

    Returns:
        List of search results

    Raises:
        ValueError: If required params missing
    """
    query_text = params.get("query")
    if not query_text:
        raise ValueError("Missing required parameter: query")

    limit = params.get("limit", 10)
    context_type = params.get("context_type", "balanced")
    use_graph = params.get("use_graph", False)
    use_citations = params.get("use_citations", False)

    # Context-based weight presets
    weight_presets = {
        "building": (0.2, 0.8),   # Favor semantic breadth
        "auditing": (0.5, 0.5),   # Favor precision
        "balanced": (0.3, 0.7),   # Default
    }
    fts_weight, vector_weight = weight_presets.get(context_type, (0.3, 0.7))

    # Get query embedding
    embed_client = get_embed_client()
    embedding = await embed_client.embed_query(query_text)

    # Build search query
    search_query = SearchQuery(
        text=query_text,
        embedding=embedding,
        fts_weight=fts_weight,
        vector_weight=vector_weight,
        limit=limit,
        use_graph=use_graph,
        graph_weight=params.get("graph_weight", 0.15) if use_graph else 0.0,
        use_citations=use_citations,
        citation_weight=params.get("citation_weight", 0.15) if use_citations else 0.0,
    )

    # Execute search
    logger.info(
        "search_request",
        query=query_text[:50],
        limit=limit,
        context=context_type,
        use_graph=use_graph,
        use_citations=use_citations,
    )

    # Use v2 search if graph or citations enabled, otherwise basic hybrid
    if use_graph or use_citations:
        results = await search_hybrid_v2(search_query)
    else:
        results = await search_hybrid(search_query)

    return [_result_to_dict(r) for r in results]


async def handle_health(params: dict[str, Any]) -> dict[str, Any]:
    """Handle health method.

    Returns:
        Health status with component checks
    """
    uptime = time.time() - _start_time

    # Check database
    db_status = "unknown"
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {e}"

    # Check embed server
    embed_client = get_embed_client()
    embed_health = await embed_client.health_check()

    overall = "healthy"
    if db_status != "healthy" or embed_health.get("status") != "healthy":
        overall = "degraded"

    return {
        "status": overall,
        "uptime_seconds": round(uptime, 1),
        "database": db_status,
        "embed_server": embed_health,
    }


async def handle_stats(params: dict[str, Any]) -> dict[str, Any]:
    """Handle stats method.

    Returns:
        Database statistics
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Get counts
        sources_count = await conn.fetchval("SELECT COUNT(*) FROM sources")
        chunks_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        concepts_count = await conn.fetchval("SELECT COUNT(*) FROM concepts")
        citations_count = await conn.fetchval("SELECT COUNT(*) FROM citations")
        relationships_count = await conn.fetchval("SELECT COUNT(*) FROM concept_relationships")

        # Get source type breakdown
        source_types = await conn.fetch(
            "SELECT source_type, COUNT(*) as count FROM sources GROUP BY source_type"
        )

    return {
        "sources": sources_count,
        "chunks": chunks_count,
        "concepts": concepts_count,
        "citations": citations_count,
        "relationships": relationships_count,
        "source_types": {row["source_type"]: row["count"] for row in source_types},
    }


# Method registry
METHODS = {
    "search": handle_search,
    "health": handle_health,
    "stats": handle_stats,
}


async def dispatch(method: str, params: Optional[dict[str, Any]] = None) -> Any:
    """Dispatch JSON-RPC method call.

    Args:
        method: Method name
        params: Method parameters

    Returns:
        Method result

    Raises:
        ValueError: If method not found
    """
    handler = METHODS.get(method)
    if handler is None:
        raise ValueError(f"Method not found: {method}")

    return await handler(params or {})
