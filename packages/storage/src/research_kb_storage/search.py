"""Hybrid search combining FTS and vector similarity.

Provides:
- Full-text search (PostgreSQL ts_rank)
- Vector similarity search (pgvector cosine similarity)
- Hybrid search with weighted combination
- Cross-encoder reranking (Phase 3)

Score semantics:
- fts_score: Higher = better match (PostgreSQL ts_rank)
- vector_score: Higher = more similar (1=identical, 0=opposite)
- combined_score: Weighted combination, higher = better
- rerank_score: Cross-encoder relevance (higher = better)
"""

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import asyncpg
from pgvector.asyncpg import register_vector
from research_kb_common import SearchError, get_logger
from research_kb_contracts import Chunk, SearchResult, Source

from research_kb_storage.connection import get_connection_pool

if TYPE_CHECKING:
    from research_kb_storage.query_expander import ExpandedQuery, HydeConfig

logger = get_logger(__name__)

# Limit concurrent graph-boosted searches to prevent memory accumulation.
# The daemon allows 50 concurrent connections, but each graph-boosted search
# can trigger a large KuzuDB SHORTEST query + in-memory DataFrame processing.
# Capping at 3 prevents OOM while still serving multiple users.
_graph_semaphore: asyncio.Semaphore = asyncio.Semaphore(3)


@dataclass
class SearchQuery:
    """Hybrid search query configuration.

    Attributes:
        text: Query text for full-text search
        embedding: Query embedding vector for vector search (1024-dim, BGE-large-en-v1.5)
        fts_weight: Weight for FTS score (default: 0.3)
        vector_weight: Weight for vector score (default: 0.7)
        limit: Maximum number of results (default: 10)
        source_filter: Optional source type filter
        graph_weight: Weight for graph score (default: 0.0, disabled)
        use_graph: Enable graph-boosted search (default: False)
        max_hops: Maximum hops for graph traversal (default: 2)
        citation_weight: Weight for citation authority score (default: 0.0, Phase 3)
        use_citations: Enable citation authority boosting (default: False)
        domain_id: Filter by knowledge domain (None = all domains, Phase 4)
    """

    text: Optional[str] = None
    embedding: Optional[list[float]] = None
    fts_weight: float = 0.3
    vector_weight: float = 0.7
    limit: int = 10
    source_filter: Optional[str] = None  # Filter by source_type

    # Graph-boosted search (Phase 2+)
    graph_weight: float = 0.0  # Default 0.0 for backwards compatibility
    use_graph: bool = False  # Explicit opt-in flag
    max_hops: int = 2  # For compute_graph_score()

    # Citation authority boosting (Phase 3)
    citation_weight: float = 0.0  # Default 0.0 for backwards compatibility
    use_citations: bool = False  # Explicit opt-in flag

    # Multi-domain support (Phase 4)
    domain_id: Optional[str] = None  # Filter by domain (None = all domains)

    # Scoring method (Phase 3)
    scoring_method: str = "weighted"  # "weighted" or "rrf"

    def __post_init__(self):
        """Validate search query."""
        if self.text is None and self.embedding is None:
            raise ValueError("Must provide at least one of: text, embedding")

        if self.scoring_method not in ("weighted", "rrf"):
            raise ValueError(
                f"scoring_method must be 'weighted' or 'rrf', got '{self.scoring_method}'"
            )

        if self.embedding is not None and len(self.embedding) != 1024:
            raise ValueError(
                f"Embedding must be 1024 dimensions (BGE-large-en-v1.5), got {len(self.embedding)}"
            )

        # Normalize weights to sum to 1
        # Determine active weights based on flags
        active_weights = [
            ("fts", self.fts_weight),
            ("vector", self.vector_weight),
        ]
        if self.use_graph:
            active_weights.append(("graph", self.graph_weight))
        if self.use_citations:
            active_weights.append(("citation", self.citation_weight))

        total = sum(w for _, w in active_weights)
        if total <= 0:
            raise ValueError("At least one weight must be positive")

        # Normalize
        self.fts_weight = self.fts_weight / total
        self.vector_weight = self.vector_weight / total
        if self.use_graph:
            self.graph_weight = self.graph_weight / total
        if self.use_citations:
            self.citation_weight = self.citation_weight / total


def compute_rrf_score(rankings: dict[str, int], k: int = 60) -> float:
    """Compute Reciprocal Rank Fusion (RRF) score from multiple ranking signals.

    RRF is a parameter-free rank aggregation method that combines rankings from
    multiple signals into a single score. It often outperforms weighted sums
    when signals have different score distributions.

    Formula: score = Σ 1/(k + rank) for each signal

    Reference:
        Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
        Reciprocal rank fusion outperforms condorcet and individual rank learning methods.
        SIGIR.

    Args:
        rankings: Dict mapping signal name to rank (1-indexed, lower = better)
                 Example: {"fts": 3, "vector": 1, "graph": 5}
        k: Smoothing constant (default 60 per original paper)
           Higher k reduces impact of high-ranked items

    Returns:
        RRF score (higher = better). Range depends on number of signals.
        With 4 signals, theoretical max is ~0.066 (all rank 1)

    Example:
        >>> compute_rrf_score({"fts": 1, "vector": 3})
        0.03278688524590164  # 1/61 + 1/63
        >>> compute_rrf_score({"fts": 1, "vector": 1, "graph": 1, "citation": 1})
        0.06557377049180328  # 4 * (1/61)
    """
    if not rankings:
        return 0.0
    return sum(1.0 / (k + rank) for rank in rankings.values() if rank is not None)


def _compute_ranks_by_signal(results: list) -> dict[str, dict[str, int]]:
    """Compute per-signal rankings for a list of search results.

    Groups results by each signal (FTS, vector, graph, citation) and assigns
    ranks based on descending score order.

    Args:
        results: List of SearchResult objects with score attributes

    Returns:
        Dict mapping chunk_id -> signal_name -> rank
        Example: {"chunk-uuid-1": {"fts": 2, "vector": 1}, ...}
    """
    from collections import defaultdict

    # Group chunk IDs by their score in each signal
    chunk_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for r in results:
        chunk_id = str(r.chunk.id)
        if r.fts_score is not None:
            chunk_scores[chunk_id]["fts"] = r.fts_score
        if r.vector_score is not None:
            chunk_scores[chunk_id]["vector"] = r.vector_score
        if r.graph_score is not None:
            chunk_scores[chunk_id]["graph"] = r.graph_score
        if r.citation_score is not None:
            chunk_scores[chunk_id]["citation"] = r.citation_score

    # Compute ranks for each signal
    rankings: dict[str, dict[str, int]] = defaultdict(dict)
    for signal in ["fts", "vector", "graph", "citation"]:
        # Get all chunks with this signal's score
        scored = [
            (chunk_id, scores.get(signal))
            for chunk_id, scores in chunk_scores.items()
            if scores.get(signal) is not None
        ]
        # Sort by score descending (higher = better)
        scored.sort(key=lambda x: x[1], reverse=True)
        # Assign ranks
        for rank, (chunk_id, _) in enumerate(scored, start=1):
            rankings[chunk_id][signal] = rank

    return dict(rankings)


async def search_hybrid(query: SearchQuery) -> list[SearchResult]:
    """Execute hybrid search combining FTS and vector similarity.

    Strategy:
    1. FTS: PostgreSQL full-text search with ts_rank scoring
    2. Vector: pgvector cosine similarity (lower = more similar)
    3. Combine: Normalize scores and apply weighted combination

    Args:
        query: Search query configuration

    Returns:
        List of SearchResults ranked by combined score

    Raises:
        SearchError: If search fails

    Example:
        >>> results = await search_hybrid(SearchQuery(
        ...     text="backdoor criterion",
        ...     embedding=[0.1] * 384,
        ...     fts_weight=0.3,
        ...     vector_weight=0.7,
        ...     limit=5
        ... ))
    """
    pool = await get_connection_pool()

    try:
        async with pool.acquire() as conn:
            await register_vector(conn)
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

            # Build query based on available search modes
            if query.text and query.embedding:
                # Hybrid: FTS + Vector
                results = await _hybrid_search(conn, query)
            elif query.text:
                # FTS only
                results = await _fts_search(conn, query)
            elif query.embedding:
                # Vector only
                results = await _vector_search(conn, query)
            else:
                raise SearchError("No search criteria provided")

            # Apply RRF reranking if requested (otherwise weighted scores from SQL)
            if query.scoring_method == "rrf" and results:
                chunk_rankings = _compute_ranks_by_signal(results)
                for result in results:
                    chunk_id = str(result.chunk.id)
                    rankings = chunk_rankings.get(chunk_id, {})
                    result.combined_score = compute_rrf_score(rankings)
                # Re-sort by RRF score and reassign ranks
                results.sort(key=lambda r: r.combined_score, reverse=True)
                for rank, result in enumerate(results, start=1):
                    result.rank = rank

            logger.info(
                "search_completed",
                mode=(
                    "hybrid"
                    if (query.text and query.embedding)
                    else ("fts" if query.text else "vector")
                ),
                result_count=len(results),
                scoring_method=query.scoring_method,
            )

            return results

    except SearchError:
        raise
    except Exception as e:
        logger.error("search_failed", error=str(e))
        raise SearchError(f"Search failed: {e}") from e


async def search_hybrid_v2(query: SearchQuery) -> list[SearchResult]:
    """Execute hybrid search v2 with graph boosting and citation authority.

    Enhanced search combining FTS + vector + graph + citation signals for improved relevance.

    Strategy:
    1. Extract concepts from query text
    2. Execute base FTS + vector search (fetch 2x limit for re-ranking)
    3. Fetch chunk-concept links for all results (batch operation)
    4. Compute graph scores using concept relationships
    5. Fetch citation authority for each result's source (batch operation)
    6. Re-rank with 4-way combination: fts + vector + graph + citation

    Args:
        query: Search query with use_graph=True and/or use_citations=True

    Returns:
        List of SearchResults ranked by combined FTS + vector + graph + citation score

    Raises:
        ValueError: If neither use_graph nor use_citations is True
        SearchError: If search fails

    Example:
        >>> results = await search_hybrid_v2(SearchQuery(
        ...     text="instrumental variables",
        ...     embedding=[0.1] * 1024,
        ...     fts_weight=0.2,
        ...     vector_weight=0.4,
        ...     graph_weight=0.2,
        ...     citation_weight=0.2,
        ...     use_graph=True,
        ...     use_citations=True,
        ...     max_hops=2,
        ...     limit=10
        ... ))
    """
    if not query.use_graph and not query.use_citations:
        raise ValueError("search_hybrid_v2 requires use_graph=True and/or use_citations=True")

    pool = await get_connection_pool()

    try:
        # Step 1: Extract concepts from query text
        from research_kb_storage.query_extractor import extract_query_concepts
        from research_kb_storage.chunk_concept_store import ChunkConceptStore
        from research_kb_storage.graph_queries import compute_weighted_graph_score

        query_concept_ids = []
        if query.text:
            query_concept_ids = await extract_query_concepts(
                query.text, min_confidence=0.6, max_concepts=5
            )

        logger.info(
            "graph_search_query_concepts",
            query_text=query.text[:100] if query.text else None,
            concept_count=len(query_concept_ids),
        )

        # Step 2: Get base results (FTS + vector), fetch 2x limit for re-ranking
        async with pool.acquire() as conn:
            await register_vector(conn)
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

            # Use larger limit for initial fetch to allow re-ranking
            fetch_limit = query.limit * 2

            # Build base query based on available search modes
            if query.text and query.embedding:
                # Hybrid: FTS + Vector
                base_results = await _hybrid_search_for_rerank(conn, query, fetch_limit)
            elif query.text:
                # FTS only
                temp_query = SearchQuery(
                    text=query.text,
                    fts_weight=1.0,
                    vector_weight=0.0,
                    limit=fetch_limit,
                    source_filter=query.source_filter,
                    domain_id=query.domain_id,
                )
                base_results = await _fts_search(conn, temp_query)
            elif query.embedding:
                # Vector only
                temp_query = SearchQuery(
                    embedding=query.embedding,
                    fts_weight=0.0,
                    vector_weight=1.0,
                    limit=fetch_limit,
                    source_filter=query.source_filter,
                    domain_id=query.domain_id,
                )
                base_results = await _vector_search(conn, temp_query)
            else:
                raise SearchError("No search criteria provided")

        # Step 3: Fetch chunk-concept links with mention metadata (batch operation)
        chunk_ids = [result.chunk.id for result in base_results]
        chunk_concepts_info: dict = {}
        if query.use_graph:
            # Use enhanced function that returns (concept_id, mention_type, relevance_score)
            chunk_concepts_info = await ChunkConceptStore.get_concept_info_for_chunks(chunk_ids)

        # Step 4: Compute graph scores in batch (1 KuzuDB query instead of ~20)
        if query.use_graph:
            from research_kb_storage.graph_queries import (
                apply_mention_weights,
                _check_kuzu_ready,
            )

            # Build parallel lists of chunk concept IDs and mention info
            all_chunk_concept_ids: list[list] = []
            all_mention_infos: list[dict | None] = []
            for result in base_results:
                chunk_info = chunk_concepts_info.get(result.chunk.id, [])
                chunk_concept_ids = [cid for cid, _, _ in chunk_info]
                mention_info = (
                    {cid: (mtype, rel) for cid, mtype, rel in chunk_info}
                    if chunk_concept_ids
                    else None
                )
                all_chunk_concept_ids.append(chunk_concept_ids)
                all_mention_infos.append(mention_info)

            if query_concept_ids and any(all_chunk_concept_ids):
                # Use batch scoring when KuzuDB is available (1 query for all results)
                if _check_kuzu_ready():
                    from research_kb_storage.kuzu_store import (
                        compute_batch_graph_scores,
                    )

                    async with _graph_semaphore:
                        batch_scores = await compute_batch_graph_scores(
                            query_concept_ids,
                            all_chunk_concept_ids,
                            max_hops=query.max_hops,
                        )

                    # Apply mention weights as post-processing per chunk
                    for i, result in enumerate(base_results):
                        raw_score = batch_scores[i]
                        result.graph_score = apply_mention_weights(
                            raw_score,
                            all_chunk_concept_ids[i],
                            all_mention_infos[i],
                        )
                else:
                    # Fallback: per-result scoring via PostgreSQL CTEs
                    async with _graph_semaphore:
                        for i, result in enumerate(base_results):
                            chunk_concept_ids = all_chunk_concept_ids[i]
                            if query_concept_ids and chunk_concept_ids:
                                graph_score, _ = await compute_weighted_graph_score(
                                    query_concept_ids,
                                    chunk_concept_ids,
                                    max_hops=query.max_hops,
                                    chunk_mention_info=all_mention_infos[i],
                                )
                            else:
                                graph_score = 0.0
                            result.graph_score = graph_score
            else:
                # No query concepts or no chunk concepts — zero scores
                for result in base_results:
                    result.graph_score = 0.0

        # Step 4b: Fetch citation authority for each source (batch operation)
        source_authorities = {}
        if query.use_citations:
            source_ids = list({result.source.id for result in base_results})
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, citation_authority
                    FROM sources
                    WHERE id = ANY($1)
                    """,
                    source_ids,
                )
                source_authorities = {row["id"]: row["citation_authority"] or 0.0 for row in rows}

            # Store citation score in each result
            for result in base_results:
                result.citation_score = source_authorities.get(result.source.id, 0.0)

        # Step 4c: Renormalize weights if signals contributed nothing
        # This prevents penalizing FTS/vector when no concepts/citations match
        has_graph_contribution = query.use_graph and any(
            r.graph_score and r.graph_score > 0 for r in base_results
        )
        has_citation_contribution = query.use_citations and any(
            r.citation_score and r.citation_score > 0 for r in base_results
        )

        # Compute effective weights (local vars — never mutate the input query)
        fts_w = query.fts_weight
        vector_w = query.vector_weight
        graph_w = query.graph_weight if has_graph_contribution else 0.0
        citation_w = query.citation_weight if has_citation_contribution else 0.0

        # Renormalize to contributing signals only
        total_weight = fts_w + vector_w + graph_w + citation_w
        if total_weight > 0:
            fts_w /= total_weight
            vector_w /= total_weight
            graph_w /= total_weight
            citation_w /= total_weight

        logger.debug(
            "weights_after_renormalization",
            fts=fts_w,
            vector=vector_w,
            graph=graph_w,
            citation=citation_w,
            has_graph=has_graph_contribution,
            has_citation=has_citation_contribution,
        )

        # Step 5: Re-rank with scoring method (weighted sum or RRF)
        if query.scoring_method == "rrf":
            # RRF: Reciprocal Rank Fusion (parameter-free, rank-based)
            # Compute per-signal ranks for all results
            chunk_rankings = _compute_ranks_by_signal(base_results)

            for result in base_results:
                chunk_id = str(result.chunk.id)
                rankings = chunk_rankings.get(chunk_id, {})
                result.combined_score = compute_rrf_score(rankings)

            logger.debug(
                "rrf_scoring_applied",
                num_results=len(base_results),
                sample_rankings=(list(chunk_rankings.items())[:3] if chunk_rankings else []),
            )
        else:
            # Weighted sum (default): normalize and combine scores
            for result in base_results:
                # Get individual scores (already normalized by base search)
                fts_score_norm = result.fts_score if result.fts_score is not None else 0.0
                vector_score_norm = result.vector_score if result.vector_score is not None else 0.0
                graph_score_norm = result.graph_score if result.graph_score is not None else 0.0
                citation_score_norm = (
                    result.citation_score if result.citation_score is not None else 0.0
                )

                # Normalize FTS score (already 0-1 from ts_rank, just need to handle None)
                # Vector score already 0-1 similarity
                # Graph score already 0-1
                # Citation authority already 0-1 (PageRank normalized)

                # Compute combined score with 4-way weighting
                result.combined_score = (
                    fts_w * fts_score_norm
                    + vector_w * vector_score_norm
                    + graph_w * graph_score_norm
                    + citation_w * citation_score_norm
                )

        # Sort by combined score and apply final limit
        base_results.sort(key=lambda r: r.combined_score, reverse=True)
        final_results = base_results[: query.limit]

        # Update ranks
        for rank, result in enumerate(final_results, start=1):
            result.rank = rank

        logger.info(
            "enhanced_search_completed",
            result_count=len(final_results),
            query_concepts=len(query_concept_ids),
            graph_weight=graph_w,
            citation_weight=citation_w,
            scoring_method=query.scoring_method,
        )

        return final_results

    except SearchError:
        raise
    except Exception as e:
        logger.error("graph_search_failed", error=str(e))
        raise SearchError(f"Graph-boosted search failed: {e}") from e


async def search_with_rerank(
    query: SearchQuery,
    rerank_top_k: int = 10,
    fetch_multiplier: int = 5,
) -> list[SearchResult]:
    """Execute hybrid search with cross-encoder reranking.

    Two-stage retrieval:
    1. Fast retrieval: FTS + vector + graph returns top-(limit * fetch_multiplier)
    2. Accurate reranking: Cross-encoder reranks to top-rerank_top_k

    Args:
        query: Search query configuration
        rerank_top_k: Number of results to return after reranking
        fetch_multiplier: How many more candidates to fetch for reranking
                         (e.g., 5 means fetch 50 candidates to rerank to 10)

    Returns:
        List of SearchResults ranked by cross-encoder score

    Raises:
        SearchError: If search fails
        ConnectionError: If rerank server not available

    Example:
        >>> results = await search_with_rerank(
        ...     SearchQuery(text="instrumental variables", embedding=embed("IV")),
        ...     rerank_top_k=10
        ... )
    """
    # Import here to avoid circular dependency
    from research_kb_pdf.rerank_client import RerankClient

    # Step 1: Fetch more candidates for reranking
    original_limit = query.limit
    query.limit = rerank_top_k * fetch_multiplier

    # Use graph-boosted search if enabled, otherwise basic hybrid
    if query.use_graph:
        candidates = await search_hybrid_v2(query)
    else:
        candidates = await search_hybrid(query)

    # Restore original limit
    query.limit = original_limit

    if not candidates:
        return []

    # Step 2: Rerank with cross-encoder
    client = RerankClient()

    if not client.is_available():
        logger.warning(
            "rerank_server_unavailable",
            message="Returning results without reranking",
        )
        # Return top results without reranking
        return candidates[:rerank_top_k]

    try:
        reranked = client.rerank_search_results(
            query=query.text,
            results=candidates,
            top_k=rerank_top_k,
        )

        logger.info(
            "search_reranked",
            candidates=len(candidates),
            reranked=len(reranked),
        )

        return reranked

    except Exception as e:
        logger.warning(
            "rerank_failed",
            error=str(e),
            message="Returning results without reranking",
        )
        # Graceful fallback: return unreranked results
        return candidates[:rerank_top_k]


def _build_hybrid_sql() -> str:
    """Build the shared hybrid search SQL query.

    Used by both _hybrid_search and _hybrid_search_for_rerank — the only
    difference is the limit parameter ($5) passed at execution time.

    Parameters: $1=text, $2=embedding, $3=fts_weight, $4=vector_weight,
                $5=limit, $6=source_filter, $7=domain_id
    """
    return """
    WITH fts_results AS (
        SELECT
            c.id,
            c.source_id,
            ts_rank(c.fts_vector, plainto_tsquery('english', $1)) AS fts_score
        FROM chunks c
        WHERE c.fts_vector @@ plainto_tsquery('english', $1)
          AND c.embedding IS NOT NULL
          AND ($7::text IS NULL OR c.domain_id = $7)
    ),
    vector_results AS (
        SELECT
            c.id,
            c.source_id,
            c.embedding <=> $2::vector(1024) AS vector_distance
        FROM chunks c
        WHERE c.embedding IS NOT NULL
          AND ($7::text IS NULL OR c.domain_id = $7)
    ),
    combined AS (
        SELECT
            COALESCE(fts.id, vec.id) AS chunk_id,
            COALESCE(fts.source_id, vec.source_id) AS source_id,
            COALESCE(fts.fts_score, 0) AS fts_score,
            COALESCE(vec.vector_distance, 2.0) AS vector_distance
        FROM fts_results fts
        FULL OUTER JOIN vector_results vec ON fts.id = vec.id
    ),
    with_similarity AS (
        SELECT
            chunk_id,
            source_id,
            fts_score,
            vector_distance,
            -- Convert vector distance to similarity (0=identical, 2=opposite -> 1=identical, 0=opposite)
            1.0 - (vector_distance / 2.0) AS vector_similarity
        FROM combined
    ),
    normalized AS (
        SELECT
            chunk_id,
            source_id,
            fts_score,
            vector_distance,
            vector_similarity,
            -- Normalize FTS score (min-max to 0-1 within result set)
            CASE
                WHEN MAX(fts_score) OVER () - MIN(fts_score) OVER () > 0
                THEN (fts_score - MIN(fts_score) OVER ()) / (MAX(fts_score) OVER () - MIN(fts_score) OVER ())
                WHEN MAX(fts_score) OVER () > 0 THEN 1.0
                ELSE 0
            END AS fts_normalized,
            -- Normalize vector similarity (min-max to 0-1 within result set)
            CASE
                WHEN MAX(vector_similarity) OVER () - MIN(vector_similarity) OVER () > 0
                THEN (vector_similarity - MIN(vector_similarity) OVER ()) / (MAX(vector_similarity) OVER () - MIN(vector_similarity) OVER ())
                WHEN MAX(vector_similarity) OVER () > 0 THEN 1.0
                ELSE 0
            END AS vector_normalized
        FROM with_similarity
    )
    SELECT
        c.id, c.source_id, c.domain_id, c.content, c.content_hash, c.location,
        c.page_start, c.page_end, c.embedding,
        c.metadata AS chunk_metadata,
        c.created_at AS chunk_created_at,
        s.id AS source__id, s.source_type, s.title, s.authors, s.year,
        s.domain_id AS source_domain_id, s.file_path, s.file_hash,
        s.metadata AS source_metadata,
        s.created_at AS source_created_at, s.updated_at,
        n.fts_score,
        n.vector_distance,
        ($3 * n.fts_normalized + $4 * n.vector_normalized) AS combined_score
    FROM normalized n
    JOIN chunks c ON c.id = n.chunk_id
    JOIN sources s ON s.id = n.source_id
    WHERE ($6::text IS NULL OR s.source_type = $6)
    ORDER BY combined_score DESC
    LIMIT $5
    """


async def _hybrid_search_for_rerank(
    conn: asyncpg.Connection, query: SearchQuery, limit: int
) -> list[SearchResult]:
    """Execute hybrid search for re-ranking.

    Same as _hybrid_search but with custom limit and returns mutable results.
    """
    sql = _build_hybrid_sql()

    rows = await conn.fetch(
        sql,
        query.text,
        query.embedding,
        float(query.fts_weight),  # Explicit float for PostgreSQL type inference
        float(query.vector_weight),
        limit,
        query.source_filter,
        query.domain_id,
    )

    return [await _row_to_search_result(row, rank + 1) for rank, row in enumerate(rows)]


async def _hybrid_search(conn: asyncpg.Connection, query: SearchQuery) -> list[SearchResult]:
    """Execute hybrid search (FTS + vector).

    Combined score = (fts_weight * fts_score_normalized) + (vector_weight * vector_score_normalized)
    """
    sql = _build_hybrid_sql()

    rows = await conn.fetch(
        sql,
        query.text,
        query.embedding,
        query.fts_weight,
        query.vector_weight,
        query.limit,
        query.source_filter,
        query.domain_id,
    )

    return [await _row_to_search_result(row, rank + 1) for rank, row in enumerate(rows)]


async def _fts_search(conn: asyncpg.Connection, query: SearchQuery) -> list[SearchResult]:
    """Execute FTS-only search."""
    sql = """
    SELECT
        c.id, c.source_id, c.domain_id, c.content, c.content_hash, c.location,
        c.page_start, c.page_end, c.embedding,
        c.metadata AS chunk_metadata,
        c.created_at AS chunk_created_at,
        s.id AS source__id, s.source_type, s.title, s.authors, s.year,
        s.domain_id AS source_domain_id, s.file_path, s.file_hash,
        s.metadata AS source_metadata,
        s.created_at AS source_created_at, s.updated_at,
        ts_rank(c.fts_vector, plainto_tsquery('english', $1)) AS fts_score
    FROM chunks c
    JOIN sources s ON s.id = c.source_id
    WHERE c.fts_vector @@ plainto_tsquery('english', $1)
      AND ($3::text IS NULL OR s.source_type = $3)
      AND ($4::text IS NULL OR c.domain_id = $4)
    ORDER BY fts_score DESC
    LIMIT $2
    """

    rows = await conn.fetch(sql, query.text, query.limit, query.source_filter, query.domain_id)

    return [
        await _row_to_search_result(row, rank + 1, fts_only=True) for rank, row in enumerate(rows)
    ]


async def _vector_search(conn: asyncpg.Connection, query: SearchQuery) -> list[SearchResult]:
    """Execute vector-only search."""
    sql = """
    SELECT
        c.id, c.source_id, c.domain_id, c.content, c.content_hash, c.location,
        c.page_start, c.page_end, c.embedding,
        c.metadata AS chunk_metadata,
        c.created_at AS chunk_created_at,
        s.id AS source__id, s.source_type, s.title, s.authors, s.year,
        s.domain_id AS source_domain_id, s.file_path, s.file_hash,
        s.metadata AS source_metadata,
        s.created_at AS source_created_at, s.updated_at,
        c.embedding <=> $1::vector(1024) AS vector_distance
    FROM chunks c
    JOIN sources s ON s.id = c.source_id
    WHERE c.embedding IS NOT NULL
      AND ($3::text IS NULL OR s.source_type = $3)
      AND ($4::text IS NULL OR c.domain_id = $4)
    ORDER BY vector_distance ASC
    LIMIT $2
    """

    rows = await conn.fetch(sql, query.embedding, query.limit, query.source_filter, query.domain_id)

    return [
        await _row_to_search_result(row, rank + 1, vector_only=True)
        for rank, row in enumerate(rows)
    ]


async def search_vector_only(query: SearchQuery) -> list[SearchResult]:
    """Fast vector-only search for latency-sensitive contexts.

    Skips FTS and normalization for maximum speed. Uses IVFFlat index
    for efficient approximate nearest neighbor search.

    Performance: ~30ms database + ~150ms embedding = ~200ms total
    (vs ~3s for hybrid search with normalization)

    Use cases:
    - ProactiveContext injection (latency budget <500ms)
    - Real-time suggestions
    - Quick relevance checks

    Args:
        query: Search query with embedding (text optional, embedding required)

    Returns:
        List of SearchResults ordered by vector similarity

    Raises:
        SearchError: If search fails or no embedding provided

    Example:
        >>> results = await search_vector_only(SearchQuery(
        ...     embedding=embed("instrumental variables"),
        ...     limit=5
        ... ))
    """
    if query.embedding is None:
        raise SearchError("search_vector_only requires an embedding")

    pool = await get_connection_pool()

    try:
        async with pool.acquire() as conn:
            await register_vector(conn)
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

            results = await _vector_search(conn, query)

            logger.info(
                "search_completed",
                mode="vector_only",
                result_count=len(results),
            )

            return results

    except SearchError:
        raise
    except Exception as e:
        logger.error("search_failed", error=str(e))
        raise SearchError(f"Vector search failed: {e}") from e


async def search_with_expansion(
    query: SearchQuery,
    use_synonyms: bool = True,
    use_graph_expansion: bool = True,
    use_llm_expansion: bool = False,
    use_rerank: bool = True,
    rerank_top_k: int = 10,
    hyde_config: Optional["HydeConfig"] = None,
) -> tuple[list[SearchResult], Optional["ExpandedQuery"]]:
    """Execute search with query expansion, optional HyDE, and optional reranking.

    Full-featured search combining:
    1. Query expansion (synonyms, graph, optional LLM)
    1.5. HyDE embedding (optional -- replaces query embedding with hypothetical doc embedding)
    2. Hybrid search (FTS + vector + graph signals)
    3. Cross-encoder reranking (optional)

    The expansion is applied to FTS search text, improving recall.
    HyDE improves vector similarity for terse queries (e.g., "IV", "DML").

    Args:
        query: Search query configuration
        use_synonyms: Enable synonym expansion (fast, deterministic)
        use_graph_expansion: Enable graph-based expansion (~10ms)
        use_llm_expansion: Enable LLM expansion via Ollama (~200-500ms)
        use_rerank: Enable cross-encoder reranking
        rerank_top_k: Number of results to return after reranking
        hyde_config: HyDE configuration (None or disabled = skip HyDE)

    Returns:
        Tuple of (results, expanded_query)
        - results: List of SearchResults
        - expanded_query: ExpandedQuery with expansion details (or None if no expansion)

    Example:
        >>> from research_kb_storage import HydeConfig
        >>> results, expansion = await search_with_expansion(
        ...     SearchQuery(text="IV", embedding=embed("IV")),
        ...     hyde_config=HydeConfig(enabled=True, backend="ollama"),
        ... )
    """
    from research_kb_storage.query_expander import QueryExpander

    expanded_query = None

    # Step 1: Expand query if requested and text is provided
    if query.text and (use_synonyms or use_graph_expansion or use_llm_expansion):
        try:
            expander = QueryExpander.from_yaml()

            # Add Ollama client if LLM expansion requested
            if use_llm_expansion:
                try:
                    from research_kb_extraction.ollama_client import OllamaClient

                    expander.ollama_client = OllamaClient()
                except ImportError:
                    logger.debug("ollama_client_not_available")

            expanded_query = await expander.expand(
                query.text,
                use_synonyms=use_synonyms,
                use_graph=use_graph_expansion,
                use_llm=use_llm_expansion,
            )

            # Use expanded FTS query if we got expansions
            if expanded_query.expanded_terms:
                logger.info(
                    "query_expanded_for_search",
                    original=query.text,
                    expansion_count=len(expanded_query.expanded_terms),
                    sources=list(expanded_query.expansion_sources.keys()),
                )

        except Exception as e:
            logger.warning(
                "query_expansion_failed",
                error=str(e),
                message="Proceeding with original query",
            )

    # Step 1.5: HyDE embedding (optional)
    if hyde_config and hyde_config.enabled and query.text:
        try:
            from research_kb_storage.query_expander import get_hyde_embedding

            hyde_embedding = await get_hyde_embedding(query.text, hyde_config)
            if hyde_embedding:
                query.embedding = hyde_embedding
                logger.info("hyde_embedding_applied", query=query.text[:50])
        except Exception as e:
            logger.warning(
                "hyde_embedding_failed_gracefully",
                error=str(e),
                message="Proceeding with original embedding",
            )

    # Step 2: Execute search
    if use_rerank:
        results = await search_with_rerank(
            query,
            rerank_top_k=rerank_top_k,
        )
    elif query.use_graph:
        results = await search_hybrid_v2(query)
    else:
        results = await search_hybrid(query)

    return results, expanded_query


async def _row_to_search_result(
    row: asyncpg.Record,
    rank: int,
    fts_only: bool = False,
    vector_only: bool = False,
) -> SearchResult:
    """Convert database row to SearchResult.

    Args:
        row: Database row with chunk + source + scores
        rank: 1-based rank in result set
        fts_only: True if FTS-only search
        vector_only: True if vector-only search

    Returns:
        SearchResult
    """
    # Extract chunk data
    chunk = Chunk(
        id=row["id"],
        source_id=row["source_id"],
        domain_id=row["domain_id"],
        content=row["content"],
        content_hash=row["content_hash"],
        location=row["location"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        embedding=list(row["embedding"]) if row["embedding"] is not None else None,
        metadata=row["chunk_metadata"],  # Chunk metadata (section, heading_level)
        created_at=row["chunk_created_at"],
    )

    # Extract source data
    source = Source(
        id=row["source_id"],
        source_type=row["source_type"],
        title=row["title"],
        authors=row["authors"],
        year=row["year"],
        domain_id=row["source_domain_id"],
        file_path=row["file_path"],
        file_hash=row["file_hash"],
        metadata=dict(row["source_metadata"]),  # Source metadata (arxiv_id, etc.)
        created_at=row["source_created_at"],
        updated_at=row["updated_at"],
    )

    # Extract scores
    fts_score = row.get("fts_score")
    vector_distance = row.get("vector_distance")

    # Convert distance to similarity (Phase 1.5.3)
    # Distance: 0=identical, 2=opposite → Similarity: 1=identical, 0=opposite
    vector_similarity = None
    if vector_distance is not None:
        vector_similarity = 1.0 - (vector_distance / 2.0)

    # Calculate combined score
    if fts_only:
        combined_score = fts_score
    elif vector_only:
        combined_score = vector_similarity
    else:
        combined_score = row["combined_score"]

    return SearchResult(
        chunk=chunk,
        source=source,
        fts_score=fts_score,
        vector_score=vector_similarity,  # Now returns similarity, not distance
        combined_score=combined_score,
        rank=rank,
    )
