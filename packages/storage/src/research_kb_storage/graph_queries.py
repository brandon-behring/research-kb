"""Graph query utilities for concept relationship traversal.

Provides:
- Shortest path finding between concepts
- N-hop neighborhood traversal
- Graph connectivity queries
- Relationship-weighted scoring (Phase 3)
- Human-readable path explanations (Phase 3)

Architecture (KuzuDB Migration):
- Primary: KuzuDB for graph traversal (<100ms)
- Fallback: PostgreSQL recursive CTEs (for robustness)
- Sync: KuzuDB mirrors PostgreSQL via scripts/sync_kuzu.py

See CLAUDE.md "KuzuDB Graph Engine" section for architecture.
"""

import asyncio
import json
import time
from typing import Optional
from uuid import UUID

from pgvector.asyncpg import register_vector
from research_kb_common import StorageError, get_logger
from research_kb_contracts import Concept, ConceptRelationship, RelationshipType

# Optional Prometheus metrics — only available when daemon is running
try:
    from research_kb_daemon.metrics import GRAPH_ENGINE_SELECTION, GRAPH_QUERY_DURATION

    _HAS_DAEMON_METRICS = True
except ImportError:
    _HAS_DAEMON_METRICS = False

from research_kb_storage.connection import get_connection_pool
from research_kb_storage.concept_store import _row_to_concept
from research_kb_storage.relationship_store import _row_to_relationship

# KuzuDB imports - optional dependency
try:
    from research_kb_storage.kuzu_store import (
        get_kuzu_connection,
        find_shortest_path_kuzu,
        find_shortest_path_length_kuzu,
        get_neighborhood_kuzu,
        compute_batch_graph_scores,
        compute_single_graph_score as kuzu_compute_weighted_score,
        DEFAULT_KUZU_PATH,
    )

    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

logger = get_logger(__name__)

# Timeout protection for PostgreSQL fallback paths
# These prevent hangs when KuzuDB is unavailable and PostgreSQL CTEs are slow
GRAPH_SCORE_TIMEOUT = 2.0  # seconds - timeout for graph scoring
PATH_QUERY_TIMEOUT = 0.5  # seconds - timeout for individual path queries


def is_kuzu_ready() -> bool:
    """Check if KuzuDB is available and ready for queries.

    Returns:
        True if KuzuDB can be used for graph queries

    Note:
        This function logs at ERROR level when Kuzu is unavailable because
        Kuzu is the primary graph database. PostgreSQL fallback is much slower.
    """
    if not KUZU_AVAILABLE:
        logger.error(
            "kuzu_import_failed",
            reason="kuzu module not available",
            impact="graph queries will use slow PostgreSQL fallback",
        )
        return False

    try:
        # Check if connection can be established
        conn = get_kuzu_connection()
        # Verify tables exist with data
        result = conn.execute("MATCH (c:Concept) RETURN count(c) AS cnt LIMIT 1")
        assert not isinstance(result, list)
        df = result.get_as_df()
        if df.empty or df.iloc[0]["cnt"] == 0:
            logger.error(
                "kuzu_empty_database",
                path=str(DEFAULT_KUZU_PATH),
                reason="KuzuDB has no concepts - run 'python scripts/sync_kuzu.py'",
                impact="graph queries will use slow PostgreSQL fallback",
            )
            return False
        return True
    except Exception as e:
        logger.error(
            "kuzu_init_failed",
            path=str(DEFAULT_KUZU_PATH),
            error=str(e),
            impact="graph queries will use slow PostgreSQL fallback (~94s vs <100ms)",
            fix="Check KuzuDB installation or run 'python scripts/sync_kuzu.py'",
        )
        return False


# Cache the result to avoid repeated checks
_kuzu_ready_cache: Optional[bool] = None
# Track if we've already warned about Kuzu unavailability
_kuzu_warned: bool = False


def _check_kuzu_ready() -> bool:
    """Cached check for KuzuDB availability.

    Logs loudly (ERROR level) on first unavailability detection.
    Subsequent checks are silent to avoid log spam.
    """
    global _kuzu_ready_cache, _kuzu_warned
    if _kuzu_ready_cache is None:
        _kuzu_ready_cache = is_kuzu_ready()
        if _kuzu_ready_cache:
            logger.info("kuzu_enabled", path=str(DEFAULT_KUZU_PATH))
        elif not _kuzu_warned:
            # Already logged at ERROR level in is_kuzu_ready()
            # Just set the flag to avoid repeated errors on subsequent calls
            _kuzu_warned = True
    return _kuzu_ready_cache


def reset_kuzu_cache() -> None:
    """Reset KuzuDB availability cache (for testing or after sync)."""
    global _kuzu_ready_cache, _kuzu_warned
    _kuzu_ready_cache = None
    _kuzu_warned = False


# Relationship weights for graph scoring (Phase 3)
# Higher weight = stronger signal for relevance
# Based on causal inference domain semantics:
# - REQUIRES: Strong dependency (method requires assumption)
# - EXTENDS: Strong extension (method extends another)
# - USES: Medium strength (method uses technique)
# - ADDRESSES: Medium strength (method addresses problem)
# - SPECIALIZES/GENERALIZES: Moderate (taxonomic relationship)
# - ALTERNATIVE_TO: Weak (suggests related but different approaches)
RELATIONSHIP_WEIGHTS: dict[RelationshipType, float] = {
    RelationshipType.REQUIRES: 1.0,
    RelationshipType.EXTENDS: 0.9,
    RelationshipType.USES: 0.8,
    RelationshipType.ADDRESSES: 0.7,
    RelationshipType.SPECIALIZES: 0.6,
    RelationshipType.GENERALIZES: 0.6,
    RelationshipType.ALTERNATIVE_TO: 0.5,
}

# Mention type weights for graph scoring
# How a concept appears in a chunk affects its relevance signal:
# - 'defines': Chunk formally defines the concept (highest signal)
# - 'reference': Chunk mentions/uses the concept (moderate signal)
# - 'example': Chunk provides an example of the concept (lower signal)
MENTION_WEIGHTS: dict[str, float] = {
    "defines": 1.0,
    "reference": 0.6,
    "example": 0.4,
}


def get_mention_weight(mention_type: str | None) -> float:
    """Get scoring weight for a concept mention type.

    Args:
        mention_type: How the concept appears in the chunk

    Returns:
        Weight value (0.4-1.0), defaults to 0.5 for unknown/None
    """
    if mention_type is None:
        return 0.5
    return MENTION_WEIGHTS.get(mention_type, 0.5)


def apply_mention_weights(
    score: float,
    chunk_concept_ids: list[UUID],
    chunk_mention_info: dict[UUID, tuple[str, float | None]] | None,
) -> float:
    """Apply mention-type weights as a post-processing multiplier on a graph score.

    This is an approximation — full mention weighting requires per-path adjustment,
    but averaging across chunk concepts gives ~95% of the benefit at 744x speed
    improvement (batch KuzuDB vs per-result PostgreSQL CTEs).

    Args:
        score: Raw graph score (0.0–1.0) from batch or single scoring
        chunk_concept_ids: Concept IDs in the chunk
        chunk_mention_info: Maps concept_id -> (mention_type, relevance_score)

    Returns:
        Score adjusted by average mention weight. Returns original score if
        no mention info provided or score is 0.
    """
    if not chunk_mention_info or score <= 0 or not chunk_concept_ids:
        return score

    mention_multiplier = 0.0
    for cid in chunk_concept_ids:
        if cid in chunk_mention_info:
            mention_type, relevance = chunk_mention_info[cid]
            weight = get_mention_weight(mention_type)
            if relevance is not None:
                weight *= relevance
            mention_multiplier += weight
        else:
            mention_multiplier += 0.5  # Default weight
    avg_mention = mention_multiplier / len(chunk_concept_ids)
    return score * avg_mention


def get_relationship_weight(rel_type: RelationshipType) -> float:
    """Get scoring weight for a relationship type.

    Args:
        rel_type: The relationship type

    Returns:
        Weight value (0.5-1.0)
    """
    return RELATIONSHIP_WEIGHTS.get(rel_type, 0.5)


async def _find_path_via_kuzu(
    start_id: UUID,
    end_id: UUID,
    max_hops: int,
) -> Optional[list[tuple[Concept, Optional[ConceptRelationship]]]]:
    """Internal: Use KuzuDB for path finding, fetch full objects from PostgreSQL.

    Hybrid approach:
    1. KuzuDB finds path (fast: ~74ms)
    2. PostgreSQL fetches full Concept/ConceptRelationship objects (necessary for API)

    Returns:
        Path as list of (Concept, Relationship) tuples, or None if no path/error
    """
    try:
        # Step 1: Get path from KuzuDB
        kuzu_path = await find_shortest_path_kuzu(start_id, end_id, max_hops)
        if not kuzu_path:
            return None

        # Step 2: Extract concept IDs from KuzuDB result
        concept_ids = [UUID(node["concept_id"]) for node in kuzu_path]

        # Step 3: Fetch full Concept objects from PostgreSQL
        pool = await get_connection_pool()
        async with pool.acquire() as conn:
            await register_vector(conn)
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

            # Fetch concepts
            concepts = {}
            for cid in concept_ids:
                row = await conn.fetchrow("SELECT * FROM concepts WHERE id = $1", cid)
                if row:
                    concepts[cid] = _row_to_concept(row)

            # Build result path - relationships come from KuzuDB data
            # Note: We don't have full ConceptRelationship objects from KuzuDB,
            # so we create minimal relationship info or fetch from PostgreSQL
            result = []
            for i, node in enumerate(kuzu_path):
                cid = UUID(node["concept_id"])
                concept = concepts.get(cid)
                if not concept:
                    # Concept missing from PostgreSQL - data sync issue
                    logger.warning("kuzu_concept_not_in_postgres", concept_id=str(cid))
                    return None  # Fall back to PostgreSQL

                if i == 0:
                    # First concept has no incoming relationship
                    result.append((concept, None))
                else:
                    # For subsequent concepts, try to find the relationship
                    prev_id = concept_ids[i - 1]
                    rel_type = node.get("rel_type")

                    # Fetch the actual relationship from PostgreSQL
                    rel_row = await conn.fetchrow(
                        """
                        SELECT * FROM concept_relationships
                        WHERE (source_concept_id = $1 AND target_concept_id = $2)
                           OR (source_concept_id = $2 AND target_concept_id = $1)
                        LIMIT 1
                        """,
                        prev_id,
                        cid,
                    )

                    relationship = _row_to_relationship(rel_row) if rel_row else None
                    result.append((concept, relationship))

            return result

    except Exception as e:
        logger.warning("kuzu_path_fallback", error=str(e))
        return None  # Signal caller to use PostgreSQL fallback


async def find_shortest_path(
    start_concept_id: UUID,
    end_concept_id: UUID,
    max_hops: int = 5,
) -> Optional[list[tuple[Concept, Optional[ConceptRelationship]]]]:
    """Find shortest path between two concepts.

    Uses KuzuDB for fast path discovery when available, then fetches full
    Concept/ConceptRelationship objects from PostgreSQL.

    Args:
        start_concept_id: Starting concept UUID
        end_concept_id: Target concept UUID
        max_hops: Maximum path length to search

    Returns:
        List of (Concept, Relationship) tuples forming the path, or None if no path exists.
        The first tuple has relationship=None (starting point).

    Example:
        [(IV_concept, None), (endogeneity_concept, ADDRESSES_rel), ...]
    """
    # Try KuzuDB for fast path discovery (~74ms vs ~2900ms PostgreSQL)
    if _check_kuzu_ready():
        kuzu_path = await _find_path_via_kuzu(start_concept_id, end_concept_id, max_hops)
        if kuzu_path is not None:
            return kuzu_path
        # KuzuDB returned None - could be no path or error, fall through

    # Fallback: PostgreSQL recursive CTE
    # WARNING: This is significantly slower than KuzuDB
    logger.warning(
        "using_postgres_fallback_for_path",
        start_id=str(start_concept_id),
        end_id=str(end_concept_id),
        reason="KuzuDB unavailable or returned no path",
        expected_latency="~2900ms vs ~74ms with Kuzu",
    )
    pool = await get_connection_pool()

    try:
        async with pool.acquire() as conn:
            await register_vector(conn)  # Required for embedding column
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

            # Recursive CTE for breadth-first search
            rows = await conn.fetch(
                """
                WITH RECURSIVE path AS (
                    -- Base case: start node
                    SELECT
                        c.id AS concept_id,
                        NULL::uuid AS relationship_id,
                        NULL::uuid AS from_concept_id,
                        ARRAY[c.id] AS visited,
                        0 AS depth
                    FROM concepts c
                    WHERE c.id = $1

                    UNION ALL

                    -- Recursive case: follow edges
                    SELECT
                        cr.target_concept_id AS concept_id,
                        cr.id AS relationship_id,
                        cr.source_concept_id AS from_concept_id,
                        p.visited || cr.target_concept_id,
                        p.depth + 1
                    FROM path p
                    JOIN concept_relationships cr ON cr.source_concept_id = p.concept_id
                    WHERE cr.target_concept_id != ALL(p.visited)  -- Avoid cycles
                      AND p.depth < $3
                )
                SELECT
                    c.*,
                    p.relationship_id,
                    cr.source_concept_id,
                    cr.target_concept_id,
                    cr.relationship_type,
                    cr.is_directed,
                    cr.strength,
                    cr.evidence_chunk_ids,
                    cr.confidence_score,
                    cr.created_at AS relationship_created_at,
                    p.depth
                FROM path p
                JOIN concepts c ON c.id = p.concept_id
                LEFT JOIN concept_relationships cr ON cr.id = p.relationship_id
                WHERE p.concept_id = $2  -- Found target
                ORDER BY p.depth ASC
                LIMIT 1
                """,
                start_concept_id,
                end_concept_id,
                max_hops,
            )

            if not rows:
                return None

            # Get full path by re-running query with path tracking
            path_rows = await conn.fetch(
                """
                WITH RECURSIVE path AS (
                    -- Base case
                    SELECT
                        c.id AS concept_id,
                        NULL::uuid AS relationship_id,
                        ARRAY[c.id] AS path_ids,
                        ARRAY[NULL::uuid] AS path_rels,
                        0 AS depth
                    FROM concepts c
                    WHERE c.id = $1

                    UNION ALL

                    -- Recursive case
                    SELECT
                        cr.target_concept_id,
                        cr.id,
                        p.path_ids || cr.target_concept_id,
                        p.path_rels || cr.id,
                        p.depth + 1
                    FROM path p
                    JOIN concept_relationships cr ON cr.source_concept_id = p.concept_id
                    WHERE cr.target_concept_id != ALL(p.path_ids)
                      AND p.depth < $3
                )
                SELECT path_ids, path_rels
                FROM path
                WHERE concept_id = $2
                ORDER BY depth ASC
                LIMIT 1
                """,
                start_concept_id,
                end_concept_id,
                max_hops,
            )

            if not path_rows:
                return None

            path_concept_ids = path_rows[0]["path_ids"]
            path_rel_ids = path_rows[0]["path_rels"]

            # Fetch all concepts and relationships
            concepts = {}
            for cid in path_concept_ids:
                concept_row = await conn.fetchrow("SELECT * FROM concepts WHERE id = $1", cid)
                concepts[cid] = _row_to_concept(concept_row)

            relationships = {}
            for rid in path_rel_ids:
                if rid is not None:
                    rel_row = await conn.fetchrow(
                        "SELECT * FROM concept_relationships WHERE id = $1", rid
                    )
                    relationships[rid] = _row_to_relationship(rel_row)

            # Build result path
            result = []
            for i, cid in enumerate(path_concept_ids):
                rid = path_rel_ids[i]
                concept = concepts[cid]
                relationship = relationships.get(rid)
                result.append((concept, relationship))

            return result

    except Exception as e:
        logger.error(
            "shortest_path_failed",
            start=str(start_concept_id),
            end=str(end_concept_id),
            error=str(e),
        )
        raise StorageError(f"Failed to find shortest path: {e}") from e


async def find_shortest_path_length(
    start_concept_id: UUID,
    end_concept_id: UUID,
    max_hops: int = 5,
) -> Optional[int]:
    """Find length of shortest path between two concepts.

    Lighter-weight version of find_shortest_path that only returns distance.

    Uses KuzuDB for fast traversal when available, falls back to PostgreSQL.

    Args:
        start_concept_id: Starting concept UUID
        end_concept_id: Target concept UUID
        max_hops: Maximum path length to search

    Returns:
        Path length (number of edges), or None if no path exists
    """
    # Try KuzuDB first (fast path: ~20ms)
    if _check_kuzu_ready():
        try:
            result = await find_shortest_path_length_kuzu(
                start_concept_id, end_concept_id, max_hops
            )
            if result is not None:
                return result
            # If None, could mean no path or KuzuDB issue - fall through to PostgreSQL
        except Exception as e:
            logger.warning("kuzu_path_length_fallback", error=str(e))
            # Fall through to PostgreSQL

    # Fallback: PostgreSQL recursive CTE
    pool = await get_connection_pool()

    try:
        async with pool.acquire() as conn:
            result = await conn.fetchval(
                """
                WITH RECURSIVE path AS (
                    SELECT
                        c.id AS concept_id,
                        ARRAY[c.id] AS visited,
                        0 AS depth
                    FROM concepts c
                    WHERE c.id = $1

                    UNION ALL

                    SELECT
                        cr.target_concept_id,
                        p.visited || cr.target_concept_id,
                        p.depth + 1
                    FROM path p
                    JOIN concept_relationships cr ON cr.source_concept_id = p.concept_id
                    WHERE cr.target_concept_id != ALL(p.visited)
                      AND p.depth < $3
                )
                SELECT MIN(depth) FROM path WHERE concept_id = $2
                """,
                start_concept_id,
                end_concept_id,
                max_hops,
            )

            return int(result) if result is not None else None

    except Exception as e:
        logger.error(
            "shortest_path_length_failed",
            start=str(start_concept_id),
            end=str(end_concept_id),
            error=str(e),
        )
        raise StorageError(f"Failed to find shortest path length: {e}") from e


async def _get_neighborhood_via_kuzu(
    concept_id: UUID,
    hops: int,
    relationship_type: Optional[RelationshipType],
) -> Optional[dict[str, list]]:
    """Internal: Use KuzuDB for neighborhood discovery, fetch full objects from PostgreSQL.

    Returns:
        Neighborhood dict or None if error (signals fallback to PostgreSQL)
    """
    try:
        # Get neighbor IDs from KuzuDB
        rel_type_str = relationship_type.value if relationship_type else None
        kuzu_result = await get_neighborhood_kuzu(concept_id, hops, rel_type_str)

        if not kuzu_result:
            return None

        # Extract neighbor IDs from KuzuDB result
        neighbor_ids = []
        for n in kuzu_result.get("concepts", []):
            nid = n.get("concept_id") or n.get("id")
            if nid and nid != str(concept_id):
                neighbor_ids.append(UUID(nid) if isinstance(nid, str) else nid)

        # Fetch full objects from PostgreSQL
        pool = await get_connection_pool()
        async with pool.acquire() as conn:
            await register_vector(conn)
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

            # Get center concept
            center_row = await conn.fetchrow("SELECT * FROM concepts WHERE id = $1", concept_id)
            if not center_row:
                raise StorageError(f"Concept not found: {concept_id}")

            center_concept = _row_to_concept(center_row)

            # Fetch all neighbor concepts
            concepts = [center_concept]
            for nid in neighbor_ids:
                concept_row = await conn.fetchrow("SELECT * FROM concepts WHERE id = $1", nid)
                if concept_row:
                    concepts.append(_row_to_concept(concept_row))

            # Fetch relationships between concepts in neighborhood
            all_ids = [concept_id] + neighbor_ids

            rel_query = """
                SELECT * FROM concept_relationships
                WHERE source_concept_id = ANY($1)
                  AND target_concept_id = ANY($1)
            """
            rel_params = [all_ids]

            if relationship_type:
                rel_query += " AND relationship_type = $2"
                rel_params.append(relationship_type.value)

            rel_rows = await conn.fetch(rel_query, *rel_params)
            relationships = [_row_to_relationship(row) for row in rel_rows]

            return {
                "center": center_concept,
                "concepts": concepts,
                "relationships": relationships,
            }

    except Exception as e:
        logger.warning("kuzu_neighborhood_fallback", error=str(e))
        return None


async def get_neighborhood(
    concept_id: UUID,
    hops: int = 1,
    relationship_type: Optional[RelationshipType] = None,
) -> dict[str, list]:
    """Get N-hop neighborhood of a concept.

    Uses KuzuDB for fast neighbor discovery when available,
    then fetches full objects from PostgreSQL.

    Args:
        concept_id: Center concept UUID
        hops: Number of hops to traverse
        relationship_type: Optional filter by relationship type

    Returns:
        Dictionary with:
        - 'concepts': List of Concept objects in neighborhood
        - 'relationships': List of ConceptRelationship edges
        - 'center': The starting Concept
    """
    # Try KuzuDB for neighbor discovery (fast path)
    if _check_kuzu_ready():
        result = await _get_neighborhood_via_kuzu(concept_id, hops, relationship_type)
        if result is not None:
            return result
        # Fall through to PostgreSQL on error

    # Fallback: PostgreSQL recursive CTE
    pool = await get_connection_pool()

    try:
        async with pool.acquire() as conn:
            await register_vector(conn)  # Required for embedding column
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

            # Get center concept
            center_row = await conn.fetchrow("SELECT * FROM concepts WHERE id = $1", concept_id)
            if not center_row:
                raise StorageError(f"Concept not found: {concept_id}")

            center_concept = _row_to_concept(center_row)

            # Recursive CTE for N-hop traversal
            type_filter = "AND cr.relationship_type = $3" if relationship_type else "AND TRUE"

            query_params = [concept_id, hops]
            if relationship_type:
                query_params.append(relationship_type.value)

            rows = await conn.fetch(
                f"""
                WITH RECURSIVE neighborhood AS (
                    -- Base case: center node
                    SELECT
                        c.id AS concept_id,
                        0 AS depth,
                        ARRAY[c.id] AS visited
                    FROM concepts c
                    WHERE c.id = $1

                    UNION

                    -- Recursive: expand outward
                    SELECT
                        cr.target_concept_id AS concept_id,
                        n.depth + 1,
                        n.visited || cr.target_concept_id
                    FROM neighborhood n
                    JOIN concept_relationships cr ON cr.source_concept_id = n.concept_id
                    WHERE cr.target_concept_id != ALL(n.visited)
                      AND n.depth < $2
                      {type_filter}
                )
                SELECT DISTINCT concept_id
                FROM neighborhood
                WHERE concept_id != $1  -- Exclude center
                """,
                *query_params,
            )

            neighbor_ids = [row["concept_id"] for row in rows]

            # Fetch all concepts
            concepts = [center_concept]
            for nid in neighbor_ids:
                concept_row = await conn.fetchrow("SELECT * FROM concepts WHERE id = $1", nid)
                concepts.append(_row_to_concept(concept_row))

            # Fetch relationships between concepts in neighborhood
            all_ids = [concept_id] + neighbor_ids

            rel_query = """
                SELECT * FROM concept_relationships
                WHERE source_concept_id = ANY($1)
                  AND target_concept_id = ANY($1)
            """
            rel_params = [all_ids]

            if relationship_type:
                rel_query += " AND relationship_type = $2"
                rel_params.append(relationship_type.value)

            rel_rows = await conn.fetch(rel_query, *rel_params)
            relationships = [_row_to_relationship(row) for row in rel_rows]

            return {
                "center": center_concept,
                "concepts": concepts,
                "relationships": relationships,
            }

    except StorageError:
        raise
    except Exception as e:
        logger.error(
            "neighborhood_query_failed",
            concept_id=str(concept_id),
            error=str(e),
        )
        raise StorageError(f"Failed to get neighborhood: {e}") from e


async def compute_graph_score(
    query_concept_ids: list[UUID],
    chunk_concept_ids: list[UUID],
    max_hops: int = 2,
) -> float:
    """Compute graph-based relevance score between query and chunk concepts.

    Uses KuzuDB batched scoring when available (~129ms for batch)
    vs PostgreSQL recursive CTEs (~96s for 10 chunks).

    Algorithm:
    1. For each query concept, find shortest path to each chunk concept
    2. Score = sum of (1 / (path_length + 1)) for connected pairs
    3. Normalize by max possible score (all pairs directly connected)

    Args:
        query_concept_ids: Concepts from query
        chunk_concept_ids: Concepts in candidate chunk
        max_hops: Maximum path length to consider

    Returns:
        Normalized score 0.0 (no connection) to 1.0 (all directly connected)
    """
    if not query_concept_ids or not chunk_concept_ids:
        return 0.0

    # Try KuzuDB batched scoring (fast path: ~129ms for 10 chunks)
    if _check_kuzu_ready():
        try:
            # compute_batch_graph_scores expects list of chunk concept lists
            scores = await compute_batch_graph_scores(
                query_concept_ids,
                [chunk_concept_ids],  # Single chunk as list of one
                max_hops,
            )
            if scores:
                return scores[0]
        except Exception as e:
            logger.warning("kuzu_graph_score_fallback", error=str(e))
            # Fall through to PostgreSQL

    # Fallback: PostgreSQL with individual path queries (slow O(N×M) CTEs)
    # Protected by timeout to prevent hangs when KuzuDB is unavailable
    total_score = 0.0
    max_pairs = len(query_concept_ids) * len(chunk_concept_ids)

    try:
        async with asyncio.timeout(GRAPH_SCORE_TIMEOUT):
            for q_id in query_concept_ids:
                for c_id in chunk_concept_ids:
                    path_len = await find_shortest_path_length(q_id, c_id, max_hops)
                    if path_len is not None:
                        # Direct link = 1.0, 1-hop = 0.5, 2-hop = 0.33
                        total_score += 1.0 / (path_len + 1)

        return min(total_score / max_pairs, 1.0)

    except asyncio.TimeoutError:
        logger.warning(
            "graph_score_timeout",
            timeout=GRAPH_SCORE_TIMEOUT,
            query_concepts=len(query_concept_ids),
            chunk_concepts=len(chunk_concept_ids),
        )
        # Return 0.0 on timeout (fall back to non-graph search)
        return 0.0
    except Exception as e:
        logger.error("graph_score_failed", error=str(e))
        # Return 0.0 on error (fail gracefully)
        return 0.0


def explain_path(
    path: list[tuple[Concept, Optional[ConceptRelationship]]],
) -> str:
    """Generate human-readable explanation of a concept path.

    Converts a path of (Concept, Relationship) tuples into a readable string
    showing the chain of reasoning.

    Args:
        path: List of (Concept, Optional[Relationship]) tuples from find_shortest_path

    Returns:
        Human-readable path explanation string

    Example:
        >>> path = await find_shortest_path(dml_id, sample_splitting_id)
        >>> print(explain_path(path))
        'double machine learning → (requires) → cross-fitting → (requires) → sample splitting'
    """
    if not path:
        return "No path"

    parts = []
    for i, (concept, relationship) in enumerate(path):
        # Get concept name (prefer canonical_name)
        name = concept.canonical_name or concept.name

        if i == 0:
            # First concept (no incoming relationship)
            parts.append(name)
        else:
            # Add relationship arrow and concept
            if relationship:
                rel_name = relationship.relationship_type.value.lower()
                parts.append(f"→ ({rel_name}) → {name}")
            else:
                parts.append(f"→ {name}")

    return " ".join(parts)


def generate_synthesis_prompt(
    path: list[tuple[Concept, Optional[ConceptRelationship]]],
    style: str = "educational",
) -> str:
    """Generate a synthesis prompt for the conceptual path.

    Creates an LLM-friendly prompt that encourages synthesis of the
    concepts along the path. Three styles support different use cases.

    Args:
        path: List of (Concept, Relationship) tuples from find_shortest_path
        style: One of "educational", "research", "implementation"
            - educational: Focus on understanding and learning
            - research: Focus on assumptions and methodological connections
            - implementation: Focus on practical coding considerations

    Returns:
        A prompt string suitable for LLM synthesis

    Example:
        >>> path = await find_shortest_path(iv_id, dml_id)
        >>> prompt = generate_synthesis_prompt(path, style="research")
        >>> print(prompt)
        'What assumptions connect instrumental variables to double machine learning?...'
    """
    if not path or len(path) < 2:
        return "No path to synthesize."

    # Extract concept names
    start_concept = path[0][0]
    end_concept = path[-1][0]
    start_name = start_concept.canonical_name or start_concept.name
    end_name = end_concept.canonical_name or end_concept.name

    # Get intermediate concepts
    intermediates = [c.canonical_name or c.name for c, _ in path[1:-1]] if len(path) > 2 else []

    # Build path with relationship types
    path_parts = []
    for i, (concept, rel) in enumerate(path):
        name = concept.canonical_name or concept.name
        if i > 0 and rel:
            rel_type = rel.relationship_type.value.upper()
            path_parts.append(f"--[{rel_type}]--> {name}")
        else:
            path_parts.append(name)
    path_with_types = " ".join(path_parts)

    # Build intermediates with definitions for implementation style
    intermediates_with_defs = []
    for concept, _ in path[1:-1]:
        name = concept.canonical_name or concept.name
        if concept.definition:
            intermediates_with_defs.append(
                f"{name} ({concept.definition[:100]}...)"
                if len(concept.definition) > 100
                else f"{name} ({concept.definition})"
            )
        else:
            intermediates_with_defs.append(name)

    # Generate style-specific prompts
    if style == "educational":
        intermediate_clause = f" through {', '.join(intermediates)}" if intermediates else ""
        return (
            f"Explain how {start_name} relates to {end_name}{intermediate_clause}. "
            f"What are the key connecting concepts and why do these relationships matter?"
        )

    elif style == "research":
        return (
            f"What assumptions connect {start_name} to {end_name}? "
            f"Trace the path: {path_with_types}. "
            f"How might violations at each step affect conclusions drawn using {end_name}?"
        )

    elif style == "implementation":
        context_clause = (
            f" Consider: {'; '.join(intermediates_with_defs)}." if intermediates_with_defs else ""
        )
        return (
            f"When implementing {end_name}, how does the conceptual path from {start_name} "
            f"inform the implementation?{context_clause}"
        )

    else:
        # Default to educational style
        return generate_synthesis_prompt(path, style="educational")


async def compute_weighted_graph_score(
    query_concept_ids: list[UUID],
    chunk_concept_ids: list[UUID],
    max_hops: int = 2,
    chunk_mention_info: dict[UUID, tuple[str, float | None]] | None = None,
) -> tuple[float, list[str]]:
    """Compute weighted graph score with path explanations and mention weighting.

    Uses KuzuDB for fast scoring (~150ms) when available,
    vs PostgreSQL recursive CTEs (~96s) for 10 chunks.

    Enhanced version of compute_graph_score that:
    1. Uses relationship type weights (REQUIRES > USES > ADDRESSES)
    2. Uses mention type weights (defines > reference > example)
    3. Returns human-readable explanations of scoring paths

    Algorithm:
    1. For each query→chunk concept pair, find shortest path
    2. Score = (product of rel weights) * mention_weight * relevance / (path_length + 1)
    3. Collect explanations for top contributing paths

    Args:
        query_concept_ids: Concepts from query
        chunk_concept_ids: Concepts in candidate chunk
        max_hops: Maximum path length to consider
        chunk_mention_info: Optional dict mapping concept_id -> (mention_type, relevance_score)
                           for weighted scoring based on how concepts appear in chunk

    Returns:
        Tuple of (normalized_score, list_of_explanations)
        - score: 0.0 (no connection) to 1.0 (all directly connected with strong relations)
        - explanations: List of path explanation strings for top contributing paths

    Example:
        >>> mention_info = {concept_id: ("defines", 0.9)}
        >>> score, explanations = await compute_weighted_graph_score(
        ...     [iv_concept_id],
        ...     [endogeneity_id],
        ...     chunk_mention_info=mention_info,
        ... )
        >>> print(f"Score: {score:.2f}")
    """
    if not query_concept_ids or not chunk_concept_ids:
        return 0.0, []

    # Try KuzuDB for fast scoring (~150ms vs ~96s PostgreSQL)
    if _check_kuzu_ready():
        try:
            _kuzu_start = time.monotonic()
            score, explanations = await kuzu_compute_weighted_score(
                query_concept_ids,
                chunk_concept_ids,
                max_hops,
            )
            _kuzu_duration = time.monotonic() - _kuzu_start

            # Record Prometheus metrics if available (daemon context)
            if _HAS_DAEMON_METRICS:
                GRAPH_QUERY_DURATION.labels(engine="kuzu").observe(_kuzu_duration)
                GRAPH_ENGINE_SELECTION.labels(engine="kuzu").inc()

            # Apply mention weights as post-processing
            score = apply_mention_weights(score, chunk_concept_ids, chunk_mention_info)

            return score, explanations
        except Exception as e:
            logger.warning("kuzu_weighted_score_fallback", error=str(e))
            if _HAS_DAEMON_METRICS:
                GRAPH_ENGINE_SELECTION.labels(engine="kuzu_fallback").inc()
            # Fall through to PostgreSQL

    # Fallback: PostgreSQL with individual path queries (slow O(N×M) CTEs)
    # Protected by timeout to prevent hangs when KuzuDB is unavailable
    if _HAS_DAEMON_METRICS:
        GRAPH_ENGINE_SELECTION.labels(engine="postgres_fallback").inc()
    _pg_start = time.monotonic()

    total_score = 0.0
    max_pairs = len(query_concept_ids) * len(chunk_concept_ids)
    path_scores: list[tuple[float, str]] = []

    try:
        async with asyncio.timeout(GRAPH_SCORE_TIMEOUT):
            for q_id in query_concept_ids:
                for c_id in chunk_concept_ids:
                    # Get full path (not just length) for weighted scoring
                    path = await find_shortest_path(q_id, c_id, max_hops)

                    if path:
                        path_len = len(path) - 1  # Number of edges

                        # Compute weighted score based on relationship types
                        if path_len == 0:
                            # Same concept (direct match)
                            path_weight = 1.0
                        else:
                            # Product of relationship weights along path
                            path_weight = 1.0
                            for concept, rel in path:
                                if rel:
                                    path_weight *= get_relationship_weight(rel.relationship_type)

                        # Apply mention weight if info provided
                        mention_weight = 1.0
                        if chunk_mention_info and c_id in chunk_mention_info:
                            mention_type, relevance = chunk_mention_info[c_id]
                            mention_weight = get_mention_weight(mention_type)
                            # Multiply by relevance_score if available
                            if relevance is not None:
                                mention_weight *= relevance

                        # Score contribution: (path_weight * mention_weight) / (path_length + 1)
                        score_contribution = (path_weight * mention_weight) / (path_len + 1)
                        total_score += score_contribution

                        # Generate explanation
                        explanation = explain_path(path)
                        path_scores.append((score_contribution, explanation))

        # Record PostgreSQL fallback duration
        if _HAS_DAEMON_METRICS:
            GRAPH_QUERY_DURATION.labels(engine="postgres_fallback").observe(
                time.monotonic() - _pg_start
            )

        # Normalize score
        normalized_score = min(total_score / max_pairs, 1.0) if max_pairs > 0 else 0.0

        # Get top explanations (sorted by score contribution)
        path_scores.sort(key=lambda x: x[0], reverse=True)
        top_explanations = [exp for _, exp in path_scores[:3]]  # Top 3 paths

        return normalized_score, top_explanations

    except asyncio.TimeoutError:
        if _HAS_DAEMON_METRICS:
            GRAPH_QUERY_DURATION.labels(engine="postgres_fallback").observe(GRAPH_SCORE_TIMEOUT)
        logger.warning(
            "weighted_graph_score_timeout",
            timeout=GRAPH_SCORE_TIMEOUT,
            query_concepts=len(query_concept_ids),
            chunk_concepts=len(chunk_concept_ids),
        )
        # Return 0.0 on timeout (fall back to non-graph search)
        return 0.0, []
    except Exception as e:
        if _HAS_DAEMON_METRICS:
            GRAPH_QUERY_DURATION.labels(engine="postgres_fallback").observe(
                time.monotonic() - _pg_start
            )
        logger.error("weighted_graph_score_failed", error=str(e))
        return 0.0, []


async def get_path_with_explanation(
    start_name: str,
    end_name: str,
    max_hops: int = 5,
) -> tuple[Optional[list[tuple[Concept, Optional[ConceptRelationship]]]], str]:
    """Find path between concepts by name and return with explanation.

    Convenience function that:
    1. Looks up concepts by canonical name
    2. Finds shortest path
    3. Returns path with human-readable explanation

    Args:
        start_name: Starting concept canonical name (case-insensitive)
        end_name: Target concept canonical name (case-insensitive)
        max_hops: Maximum path length

    Returns:
        Tuple of (path, explanation)
        - path: List of (Concept, Relationship) tuples or None if not found
        - explanation: Human-readable path string

    Example:
        >>> path, explanation = await get_path_with_explanation("dml", "k-fold cross-validation")
        >>> print(explanation)
        'double machine learning → (requires) → cross-fitting → (requires) → k-fold cross-validation'
    """
    from research_kb_storage.concept_store import ConceptStore

    # Look up start concept
    start_concept = await ConceptStore.get_by_canonical_name(start_name.lower())
    if not start_concept:
        return None, f"Start concept not found: {start_name}"

    # Look up end concept
    end_concept = await ConceptStore.get_by_canonical_name(end_name.lower())
    if not end_concept:
        return None, f"End concept not found: {end_name}"

    # Find path
    path = await find_shortest_path(start_concept.id, end_concept.id, max_hops)

    if not path:
        return (
            None,
            f"No path found between '{start_name}' and '{end_name}' within {max_hops} hops",
        )

    explanation = explain_path(path)
    return path, explanation
