"""JSON-RPC method handlers for daemon.

Implements:
- search: Hybrid search with optional graph/citation boosting
- fast_search: Vector-only search for low latency
- graph_path: Find path between concepts
- citation_info: Get citation authority for sources
- health: System health check
- stats: Database statistics
"""

import time
from typing import Any, Optional
from uuid import UUID

from research_kb_common import get_logger
from research_kb_contracts import SearchResult
from research_kb_storage import (
    SearchQuery,
    is_kuzu_ready,
    search_hybrid,
    search_hybrid_v2,
    search_vector_only,
)
from research_kb_storage.concept_store import ConceptStore as ConceptStoreClass
from research_kb_storage.graph_queries import find_shortest_path

from research_kb_daemon.metrics import (
    DAEMON_UPTIME,
    REQUEST_COUNT,
    REQUEST_DURATION,
)
from research_kb_daemon.pool import get_embed_client, get_pool, get_rerank_client

logger = get_logger(__name__)


async def _resolve_concept(name_or_id: str):
    """Resolve a concept by name, canonical name, or UUID.

    Tries multiple strategies:
    1. Exact canonical name match
    2. Lowercase canonical name match
    3. UUID lookup
    4. Fuzzy name match via ILIKE

    Returns:
        Concept if found, None otherwise
    """
    # Try canonical name (lowercase)
    concept = await ConceptStoreClass.get_by_canonical_name(name_or_id.lower())
    if concept:
        return concept

    # Try as-is (case sensitive)
    concept = await ConceptStoreClass.get_by_canonical_name(name_or_id)
    if concept:
        return concept

    # Try as UUID
    try:
        return await ConceptStoreClass.get(UUID(name_or_id))
    except (ValueError, TypeError):
        pass

    # Try fuzzy search via database
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id FROM concepts
            WHERE canonical_name ILIKE $1
               OR name ILIKE $1
            LIMIT 1
            """,
            f"%{name_or_id}%",
        )
        if row:
            return await ConceptStoreClass.get(row["id"])

    return None


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
        "page_number": (result.chunk.metadata.get("page_number") if result.chunk else None),
        "section_header": (result.chunk.metadata.get("section_header") if result.chunk else None),
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
            - domain (str): Knowledge domain to search (None = all domains)

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
    use_hyde = params.get("use_hyde", False)
    domain_id = params.get("domain")  # None = all domains

    # Context-based weight presets
    weight_presets = {
        "building": (0.2, 0.8),  # Favor semantic breadth
        "auditing": (0.5, 0.5),  # Favor precision
        "balanced": (0.3, 0.7),  # Default
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
        domain_id=domain_id,
    )

    # Execute search
    logger.info(
        "search_request",
        query=query_text[:50],
        limit=limit,
        context=context_type,
        use_graph=use_graph,
        use_citations=use_citations,
        domain=domain_id,
    )

    # Route through search_with_expansion when HyDE is requested
    if use_hyde:
        from research_kb_storage import HydeConfig, search_with_expansion

        hyde_config = HydeConfig(
            enabled=True,
            backend=params.get("hyde_backend", "ollama"),
            model=params.get("hyde_model", "llama3.1:8b"),
        )
        results, _ = await search_with_expansion(
            search_query,
            use_synonyms=False,
            use_graph_expansion=False,
            use_rerank=False,
            hyde_config=hyde_config,
        )
    # Use v2 search if graph or citations enabled, otherwise basic hybrid
    elif use_graph or use_citations:
        results = await search_hybrid_v2(search_query)
    else:
        results = await search_hybrid(search_query)

    return [_result_to_dict(r) for r in results]


async def handle_fast_search(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Handle fast_search method - vector-only search for low latency.

    Optimized for latency-sensitive contexts (ProactiveContext integration).
    Skips FTS, graph, citation, and normalization overhead.

    Performance: ~30ms database + ~150ms embedding = ~200ms total
    (vs ~3s for hybrid search with normalization)

    Args:
        params: Search parameters
            - query (str): Search query text (required)
            - limit (int): Max results (default: 5)
            - domain (str): Knowledge domain to search (None = all domains)

    Returns:
        List of search results (vector scores only)

    Raises:
        ValueError: If required params missing
    """
    query_text = params.get("query")
    if not query_text:
        raise ValueError("Missing required parameter: query")

    limit = params.get("limit", 5)
    domain_id = params.get("domain")

    # Get query embedding
    embed_client = get_embed_client()
    embedding = await embed_client.embed_query(query_text)

    # Build minimal search query (vector-only)
    search_query = SearchQuery(
        text=query_text,
        embedding=embedding,
        fts_weight=0.0,
        vector_weight=1.0,
        limit=limit,
        use_graph=False,
        graph_weight=0.0,
        use_citations=False,
        citation_weight=0.0,
        domain_id=domain_id,
    )

    logger.info(
        "fast_search_request",
        query=query_text[:50],
        limit=limit,
        domain=domain_id,
    )

    # Execute vector-only search
    results = await search_vector_only(search_query)

    return [_result_to_dict(r) for r in results]


async def handle_health(params: dict[str, Any]) -> dict[str, Any]:
    """Handle health method.

    Returns:
        Health status with component checks:
        - database: PostgreSQL connectivity
        - kuzu_graph: KuzuDB graph database for fast traversal
        - embed_server: BGE embedding service
        - rerank_server: Cross-encoder reranking service
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

    # Check KuzuDB graph database
    kuzu_status = "unhealthy"
    kuzu_reason = ""
    try:
        if is_kuzu_ready():
            kuzu_status = "healthy"
        else:
            kuzu_status = "unhealthy"
            kuzu_reason = "KuzuDB not ready - graph queries will use slow PostgreSQL fallback"
    except Exception as e:
        kuzu_status = f"unhealthy: {e}"

    # Check embed server
    embed_client = get_embed_client()
    embed_health = await embed_client.health_check()

    # Check rerank server
    rerank_client = get_rerank_client()
    rerank_health = await rerank_client.health_check()

    # Determine overall status
    # healthy = all services up including KuzuDB
    # degraded = core services up, optional (reranker, kuzu) down
    # unhealthy = core services (db, embed) down
    overall = "healthy"
    if db_status != "healthy" or embed_health.get("status") != "healthy":
        overall = "unhealthy"
    elif rerank_health.get("status") != "healthy" or kuzu_status != "healthy":
        overall = "degraded"

    kuzu_health = {"status": kuzu_status}
    if kuzu_reason:
        kuzu_health["reason"] = kuzu_reason

    # KuzuDB warmup status
    from research_kb_daemon.warmup import warmup_status

    return {
        "status": overall,
        "uptime_seconds": round(uptime, 1),
        "database": db_status,
        "kuzu_graph": kuzu_health,
        "kuzu_warmup": warmup_status(),
        "embed_server": embed_health,
        "rerank_server": rerank_health,
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


async def handle_graph_path(params: dict[str, Any]) -> dict[str, Any]:
    """Handle graph_path method - find path between concepts.

    Optimized for latency via KuzuDB (~288ms) vs PostgreSQL fallback (~94s).

    Args:
        params: Parameters
            - start (str): Start concept name or ID (required)
            - end (str): End concept name or ID (required)
            - max_hops (int): Maximum path length (default: 5)

    Returns:
        Path info with concepts and relationships

    Raises:
        ValueError: If required params missing or concepts not found
    """
    start = params.get("start")
    end = params.get("end")
    max_hops = params.get("max_hops", 5)

    if not start or not end:
        raise ValueError("Missing required parameters: start, end")

    logger.info("graph_path_request", start=start[:30], end=end[:30], max_hops=max_hops)

    # Resolve concept names to IDs - try multiple formats
    start_concept = await _resolve_concept(start)
    if not start_concept:
        raise ValueError(f"Concept not found: {start}")

    end_concept = await _resolve_concept(end)
    if not end_concept:
        raise ValueError(f"Concept not found: {end}")

    # Find path
    path = await find_shortest_path(start_concept.id, end_concept.id, max_hops)

    if not path:
        return {
            "found": False,
            "start": start_concept.name,
            "end": end_concept.name,
            "path": [],
            "hops": 0,
        }

    # Format path for JSON response
    path_items = []
    for concept, relationship in path:
        item = {
            "concept_id": str(concept.id),
            "concept_name": concept.name,
            "concept_type": (concept.concept_type.value if concept.concept_type else None),
        }
        if relationship:
            item["relationship"] = {
                "type": relationship.relationship_type.value,
                "strength": relationship.strength,
                "confidence": relationship.confidence_score,
            }
        path_items.append(item)

    return {
        "found": True,
        "start": start_concept.name,
        "end": end_concept.name,
        "path": path_items,
        "hops": len(path) - 1,
    }


async def handle_citation_info(params: dict[str, Any]) -> dict[str, Any]:
    """Handle citation_info method - get citation authority for sources.

    Args:
        params: Parameters
            - source_ids (list[str]): List of source UUIDs (required)

    Returns:
        Citation authority info for each source including:
        - authority: PageRank-style citation authority score
        - references_count: Number of citations/references in this source
        - title: Source title

    Raises:
        ValueError: If required params missing
    """
    source_ids = params.get("source_ids", [])
    if not source_ids:
        raise ValueError("Missing required parameter: source_ids")

    logger.info("citation_info_request", count=len(source_ids))

    pool = await get_pool()
    results = {}

    async with pool.acquire() as conn:
        for sid in source_ids:
            try:
                source_uuid = UUID(sid)
                row = await conn.fetchrow(
                    """
                    SELECT
                        s.id,
                        s.title,
                        s.citation_authority,
                        (SELECT COUNT(*) FROM citations WHERE source_id = s.id) as references_count
                    FROM sources s
                    WHERE s.id = $1
                    """,
                    source_uuid,
                )
                if row:
                    results[sid] = {
                        "title": row["title"],
                        "authority": row["citation_authority"],
                        "references": row["references_count"],
                    }
            except (ValueError, TypeError):
                logger.warning("invalid_source_id", source_id=sid)

    return {"sources": results}


# Method registry
METHODS = {
    "search": handle_search,
    "fast_search": handle_fast_search,
    "graph_path": handle_graph_path,
    "citation_info": handle_citation_info,
    "health": handle_health,
    "stats": handle_stats,
}


async def dispatch(method: str, params: Optional[dict[str, Any]] = None) -> Any:
    """Dispatch JSON-RPC method call with Prometheus instrumentation.

    Records:
    - Request count by method and status (success/error)
    - Request duration histogram by method and status
    - Daemon uptime gauge

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
        REQUEST_COUNT.labels(method=method, status="not_found").inc()
        raise ValueError(f"Method not found: {method}")

    # Update uptime gauge on every request
    DAEMON_UPTIME.set(time.time() - _start_time)

    start = time.monotonic()
    try:
        result = await handler(params or {})
        duration = time.monotonic() - start
        REQUEST_DURATION.labels(method=method, status="success").observe(duration)
        REQUEST_COUNT.labels(method=method, status="success").inc()
        return result
    except Exception:
        duration = time.monotonic() - start
        REQUEST_DURATION.labels(method=method, status="error").observe(duration)
        REQUEST_COUNT.labels(method=method, status="error").inc()
        raise
