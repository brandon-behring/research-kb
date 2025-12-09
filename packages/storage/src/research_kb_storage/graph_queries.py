"""Graph query utilities for concept relationship traversal.

Provides:
- Shortest path finding between concepts
- N-hop neighborhood traversal
- Graph connectivity queries
- Relationship-weighted scoring (Phase 3)
- Human-readable path explanations (Phase 3)

Uses PostgreSQL recursive CTEs for graph operations.
Master Plan Reference: Lines 616-673 (Phase 2 knowledge graph)
"""

import json
from typing import Optional
from uuid import UUID

from pgvector.asyncpg import register_vector
from research_kb_common import StorageError, get_logger
from research_kb_contracts import Concept, ConceptRelationship, RelationshipType

from research_kb_storage.connection import get_connection_pool
from research_kb_storage.concept_store import _row_to_concept
from research_kb_storage.relationship_store import _row_to_relationship

logger = get_logger(__name__)


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


def get_relationship_weight(rel_type: RelationshipType) -> float:
    """Get scoring weight for a relationship type.

    Args:
        rel_type: The relationship type

    Returns:
        Weight value (0.5-1.0)
    """
    return RELATIONSHIP_WEIGHTS.get(rel_type, 0.5)


async def find_shortest_path(
    start_concept_id: UUID,
    end_concept_id: UUID,
    max_hops: int = 5,
) -> Optional[list[tuple[Concept, Optional[ConceptRelationship]]]]:
    """Find shortest path between two concepts.

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
                concept_row = await conn.fetchrow(
                    "SELECT * FROM concepts WHERE id = $1", cid
                )
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

    Args:
        start_concept_id: Starting concept UUID
        end_concept_id: Target concept UUID
        max_hops: Maximum path length to search

    Returns:
        Path length (number of edges), or None if no path exists
    """
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

            return result

    except Exception as e:
        logger.error(
            "shortest_path_length_failed",
            start=str(start_concept_id),
            end=str(end_concept_id),
            error=str(e),
        )
        raise StorageError(f"Failed to find shortest path length: {e}") from e


async def get_neighborhood(
    concept_id: UUID,
    hops: int = 1,
    relationship_type: Optional[RelationshipType] = None,
) -> dict[str, list]:
    """Get N-hop neighborhood of a concept.

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
    pool = await get_connection_pool()

    try:
        async with pool.acquire() as conn:
            await register_vector(conn)  # Required for embedding column
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

            # Get center concept
            center_row = await conn.fetchrow(
                "SELECT * FROM concepts WHERE id = $1", concept_id
            )
            if not center_row:
                raise StorageError(f"Concept not found: {concept_id}")

            center_concept = _row_to_concept(center_row)

            # Recursive CTE for N-hop traversal
            type_filter = (
                "AND cr.relationship_type = $3" if relationship_type else "AND TRUE"
            )

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
                concept_row = await conn.fetchrow(
                    "SELECT * FROM concepts WHERE id = $1", nid
                )
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

    total_score = 0.0
    max_pairs = len(query_concept_ids) * len(chunk_concept_ids)

    try:
        for q_id in query_concept_ids:
            for c_id in chunk_concept_ids:
                path_len = await find_shortest_path_length(q_id, c_id, max_hops)
                if path_len is not None:
                    # Direct link = 1.0, 1-hop = 0.5, 2-hop = 0.33
                    total_score += 1.0 / (path_len + 1)

        return min(total_score / max_pairs, 1.0)

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


async def compute_weighted_graph_score(
    query_concept_ids: list[UUID],
    chunk_concept_ids: list[UUID],
    max_hops: int = 2,
) -> tuple[float, list[str]]:
    """Compute weighted graph score with path explanations.

    Enhanced version of compute_graph_score that:
    1. Uses relationship type weights (REQUIRES > USES > ADDRESSES)
    2. Returns human-readable explanations of scoring paths

    Algorithm:
    1. For each query→chunk concept pair, find shortest path
    2. Score = product of relationship weights along path / (path_length + 1)
    3. Collect explanations for top contributing paths

    Args:
        query_concept_ids: Concepts from query
        chunk_concept_ids: Concepts in candidate chunk
        max_hops: Maximum path length to consider

    Returns:
        Tuple of (normalized_score, list_of_explanations)
        - score: 0.0 (no connection) to 1.0 (all directly connected with strong relations)
        - explanations: List of path explanation strings for top contributing paths

    Example:
        >>> score, explanations = await compute_weighted_graph_score(
        ...     [iv_concept_id],
        ...     [endogeneity_id, unconfoundedness_id]
        ... )
        >>> print(f"Score: {score:.2f}")
        >>> for exp in explanations:
        ...     print(f"  - {exp}")
    """
    if not query_concept_ids or not chunk_concept_ids:
        return 0.0, []

    total_score = 0.0
    max_pairs = len(query_concept_ids) * len(chunk_concept_ids)
    path_scores: list[tuple[float, str]] = []

    try:
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

                    # Score contribution: weight / (path_length + 1)
                    score_contribution = path_weight / (path_len + 1)
                    total_score += score_contribution

                    # Generate explanation
                    explanation = explain_path(path)
                    path_scores.append((score_contribution, explanation))

        # Normalize score
        normalized_score = min(total_score / max_pairs, 1.0) if max_pairs > 0 else 0.0

        # Get top explanations (sorted by score contribution)
        path_scores.sort(key=lambda x: x[0], reverse=True)
        top_explanations = [exp for _, exp in path_scores[:3]]  # Top 3 paths

        return normalized_score, top_explanations

    except Exception as e:
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
        return None, f"No path found between '{start_name}' and '{end_name}' within {max_hops} hops"

    explanation = explain_path(path)
    return path, explanation
