"""Hybrid search combining FTS and vector similarity.

Provides:
- Full-text search (PostgreSQL ts_rank)
- Vector similarity search (pgvector cosine similarity)
- Hybrid search with weighted combination

Score semantics:
- fts_score: Higher = better match (PostgreSQL ts_rank)
- vector_score: Higher = more similar (1=identical, 0=opposite)
- combined_score: Weighted combination, higher = better
"""

import json
from dataclasses import dataclass
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector
from research_kb_common import SearchError, get_logger
from research_kb_contracts import Chunk, SearchResult, Source

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


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

    def __post_init__(self):
        """Validate search query."""
        if self.text is None and self.embedding is None:
            raise ValueError("Must provide at least one of: text, embedding")

        if self.embedding is not None and len(self.embedding) != 1024:
            raise ValueError(
                f"Embedding must be 1024 dimensions (BGE-large-en-v1.5), got {len(self.embedding)}"
            )

        # Normalize weights to sum to 1
        if self.use_graph:
            # Three-way normalization (FTS + vector + graph)
            total = self.fts_weight + self.vector_weight + self.graph_weight
            if total > 0:
                self.fts_weight = self.fts_weight / total
                self.vector_weight = self.vector_weight / total
                self.graph_weight = self.graph_weight / total
            else:
                raise ValueError("At least one weight must be positive")
        else:
            # Two-way normalization (FTS + vector only, backwards compatible)
            total = self.fts_weight + self.vector_weight
            if total > 0:
                self.fts_weight = self.fts_weight / total
                self.vector_weight = self.vector_weight / total
            else:
                raise ValueError("At least one weight must be positive")


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

            logger.info(
                "search_completed",
                mode=(
                    "hybrid"
                    if (query.text and query.embedding)
                    else ("fts" if query.text else "vector")
                ),
                result_count=len(results),
            )

            return results

    except SearchError:
        raise
    except Exception as e:
        logger.error("search_failed", error=str(e))
        raise SearchError(f"Search failed: {e}") from e


async def search_hybrid_v2(query: SearchQuery) -> list[SearchResult]:
    """Execute hybrid search v2 with graph boosting.

    Enhanced search combining FTS + vector + graph signals for improved relevance.

    Strategy:
    1. Extract concepts from query text
    2. Execute base FTS + vector search (fetch 2x limit for re-ranking)
    3. Fetch chunk-concept links for all results (batch operation)
    4. Compute graph scores using concept relationships
    5. Re-rank with 3-way combination: fts_weight*FTS + vector_weight*vector + graph_weight*graph

    Args:
        query: Search query with use_graph=True and graph_weight > 0

    Returns:
        List of SearchResults ranked by combined FTS + vector + graph score

    Raises:
        ValueError: If use_graph=False (must explicitly opt-in)
        SearchError: If search fails

    Example:
        >>> results = await search_hybrid_v2(SearchQuery(
        ...     text="instrumental variables",
        ...     embedding=[0.1] * 1024,
        ...     fts_weight=0.2,
        ...     vector_weight=0.5,
        ...     graph_weight=0.3,
        ...     use_graph=True,
        ...     max_hops=2,
        ...     limit=10
        ... ))
    """
    if not query.use_graph:
        raise ValueError("search_hybrid_v2 requires use_graph=True (explicit opt-in)")

    pool = await get_connection_pool()

    try:
        # Step 1: Extract concepts from query text
        from research_kb_storage.query_extractor import extract_query_concepts
        from research_kb_storage.chunk_concept_store import ChunkConceptStore
        from research_kb_storage.graph_queries import compute_graph_score

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
                )
                base_results = await _vector_search(conn, temp_query)
            else:
                raise SearchError("No search criteria provided")

        # Step 3: Fetch chunk-concept links (batch operation)
        chunk_ids = [result.chunk.id for result in base_results]
        chunk_concepts = await ChunkConceptStore.get_concept_ids_for_chunks(chunk_ids)

        # Step 4: Compute graph scores for each result
        for result in base_results:
            chunk_concept_ids = chunk_concepts.get(result.chunk.id, [])

            if query_concept_ids and chunk_concept_ids:
                # Compute graph score using shortest paths
                graph_score = await compute_graph_score(
                    query_concept_ids,
                    chunk_concept_ids,
                    max_hops=query.max_hops,
                )
            else:
                # No concepts extracted - graph score = 0
                graph_score = 0.0

            # Store graph score in result
            result.graph_score = graph_score

        # Step 5: Re-rank with 3-way scoring
        for result in base_results:
            # Get individual scores (already normalized by base search)
            fts_score_norm = result.fts_score if result.fts_score is not None else 0.0
            vector_score_norm = (
                result.vector_score if result.vector_score is not None else 0.0
            )
            graph_score_norm = (
                result.graph_score if result.graph_score is not None else 0.0
            )

            # Normalize FTS score (already 0-1 from ts_rank, just need to handle None)
            # Vector score already 0-1 similarity
            # Graph score already 0-1

            # Compute combined score with 3-way weighting
            result.combined_score = (
                query.fts_weight * fts_score_norm
                + query.vector_weight * vector_score_norm
                + query.graph_weight * graph_score_norm
            )

        # Sort by combined score and apply final limit
        base_results.sort(key=lambda r: r.combined_score, reverse=True)
        final_results = base_results[: query.limit]

        # Update ranks
        for rank, result in enumerate(final_results, start=1):
            result.rank = rank

        logger.info(
            "graph_search_completed",
            result_count=len(final_results),
            query_concepts=len(query_concept_ids),
            graph_weight=query.graph_weight,
        )

        return final_results

    except SearchError:
        raise
    except Exception as e:
        logger.error("graph_search_failed", error=str(e))
        raise SearchError(f"Graph-boosted search failed: {e}") from e


async def _hybrid_search_for_rerank(
    conn: asyncpg.Connection, query: SearchQuery, limit: int
) -> list[SearchResult]:
    """Execute hybrid search for re-ranking.

    Same as _hybrid_search but with custom limit and returns mutable results.
    """
    sql = """
    WITH fts_results AS (
        SELECT
            c.id,
            c.source_id,
            ts_rank(c.fts_vector, plainto_tsquery('english', $1)) AS fts_score
        FROM chunks c
        WHERE c.fts_vector @@ plainto_tsquery('english', $1)
          AND c.embedding IS NOT NULL
    ),
    vector_results AS (
        SELECT
            c.id,
            c.source_id,
            c.embedding <=> $2::vector(1024) AS vector_distance
        FROM chunks c
        WHERE c.embedding IS NOT NULL
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
    normalized AS (
        SELECT
            chunk_id,
            source_id,
            fts_score,
            vector_distance,
            -- Normalize FTS score (0-1)
            CASE
                WHEN MAX(fts_score) OVER () > 0
                THEN fts_score / MAX(fts_score) OVER ()
                ELSE 0
            END AS fts_normalized,
            -- Normalize vector distance (convert to similarity)
            1.0 - (vector_distance / 2.0) AS vector_normalized
        FROM combined
    )
    SELECT
        c.id, c.source_id, c.content, c.content_hash, c.location,
        c.page_start, c.page_end, c.embedding,
        c.metadata AS chunk_metadata,
        c.created_at AS chunk_created_at,
        s.id AS source__id, s.source_type, s.title, s.authors, s.year,
        s.file_path, s.file_hash,
        s.metadata AS source_metadata,
        s.created_at AS source_created_at, s.updated_at,
        n.fts_score,
        n.vector_distance,
        (n.fts_normalized + n.vector_normalized) / 2.0 AS combined_score
    FROM normalized n
    JOIN chunks c ON c.id = n.chunk_id
    JOIN sources s ON s.id = n.source_id
    WHERE ($6::text IS NULL OR s.source_type = $6)
    ORDER BY combined_score DESC
    LIMIT $5
    """

    rows = await conn.fetch(
        sql,
        query.text,
        query.embedding,
        query.fts_weight,
        query.vector_weight,
        limit,
        query.source_filter,
    )

    return [await _row_to_search_result(row, rank + 1) for rank, row in enumerate(rows)]


async def _hybrid_search(
    conn: asyncpg.Connection, query: SearchQuery
) -> list[SearchResult]:
    """Execute hybrid search (FTS + vector).

    Combined score = (fts_weight * fts_score_normalized) + (vector_weight * vector_score_normalized)
    """
    sql = """
    WITH fts_results AS (
        SELECT
            c.id,
            c.source_id,
            ts_rank(c.fts_vector, plainto_tsquery('english', $1)) AS fts_score
        FROM chunks c
        WHERE c.fts_vector @@ plainto_tsquery('english', $1)
          AND c.embedding IS NOT NULL
    ),
    vector_results AS (
        SELECT
            c.id,
            c.source_id,
            c.embedding <=> $2::vector(1024) AS vector_distance
        FROM chunks c
        WHERE c.embedding IS NOT NULL
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
    normalized AS (
        SELECT
            chunk_id,
            source_id,
            fts_score,
            vector_distance,
            -- Normalize FTS score (0-1)
            CASE
                WHEN MAX(fts_score) OVER () > 0
                THEN fts_score / MAX(fts_score) OVER ()
                ELSE 0
            END AS fts_normalized,
            -- Normalize vector distance (convert to similarity: 0=identical, 2=opposite)
            1.0 - (vector_distance / 2.0) AS vector_normalized
        FROM combined
    )
    SELECT
        c.id, c.source_id, c.content, c.content_hash, c.location,
        c.page_start, c.page_end, c.embedding,
        c.metadata AS chunk_metadata,
        c.created_at AS chunk_created_at,
        s.id AS source__id, s.source_type, s.title, s.authors, s.year,
        s.file_path, s.file_hash,
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

    rows = await conn.fetch(
        sql,
        query.text,
        query.embedding,
        query.fts_weight,
        query.vector_weight,
        query.limit,
        query.source_filter,
    )

    return [await _row_to_search_result(row, rank + 1) for rank, row in enumerate(rows)]


async def _fts_search(
    conn: asyncpg.Connection, query: SearchQuery
) -> list[SearchResult]:
    """Execute FTS-only search."""
    sql = """
    SELECT
        c.id, c.source_id, c.content, c.content_hash, c.location,
        c.page_start, c.page_end, c.embedding,
        c.metadata AS chunk_metadata,
        c.created_at AS chunk_created_at,
        s.id AS source__id, s.source_type, s.title, s.authors, s.year,
        s.file_path, s.file_hash,
        s.metadata AS source_metadata,
        s.created_at AS source_created_at, s.updated_at,
        ts_rank(c.fts_vector, plainto_tsquery('english', $1)) AS fts_score
    FROM chunks c
    JOIN sources s ON s.id = c.source_id
    WHERE c.fts_vector @@ plainto_tsquery('english', $1)
      AND ($3::text IS NULL OR s.source_type = $3)
    ORDER BY fts_score DESC
    LIMIT $2
    """

    rows = await conn.fetch(sql, query.text, query.limit, query.source_filter)

    return [
        await _row_to_search_result(row, rank + 1, fts_only=True)
        for rank, row in enumerate(rows)
    ]


async def _vector_search(
    conn: asyncpg.Connection, query: SearchQuery
) -> list[SearchResult]:
    """Execute vector-only search."""
    sql = """
    SELECT
        c.id, c.source_id, c.content, c.content_hash, c.location,
        c.page_start, c.page_end, c.embedding,
        c.metadata AS chunk_metadata,
        c.created_at AS chunk_created_at,
        s.id AS source__id, s.source_type, s.title, s.authors, s.year,
        s.file_path, s.file_hash,
        s.metadata AS source_metadata,
        s.created_at AS source_created_at, s.updated_at,
        c.embedding <=> $1::vector(1024) AS vector_distance
    FROM chunks c
    JOIN sources s ON s.id = c.source_id
    WHERE c.embedding IS NOT NULL
      AND ($3::text IS NULL OR s.source_type = $3)
    ORDER BY vector_distance ASC
    LIMIT $2
    """

    rows = await conn.fetch(sql, query.embedding, query.limit, query.source_filter)

    return [
        await _row_to_search_result(row, rank + 1, vector_only=True)
        for rank, row in enumerate(rows)
    ]


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
