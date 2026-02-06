"""KuzuDB graph store for fast concept relationship traversal.

Phase: KuzuDB Migration (replacing PostgreSQL recursive CTEs)

Provides:
- Embedded graph database for <100ms traversal queries
- Shortest path finding between concepts
- N-hop neighborhood traversal
- Batch graph scoring for search

Architecture:
- PostgreSQL remains authoritative (concepts, embeddings, metadata)
- KuzuDB mirrors graph structure (concepts + relationships only)
- Sync via scripts/sync_kuzu.py (nightly or on-demand)

Performance target: 96s → <5s for graph-boosted search
"""

import asyncio
from pathlib import Path
from typing import Optional
from uuid import UUID

import kuzu

from research_kb_common import get_logger, StorageError
from research_kb_contracts import RelationshipType

logger = get_logger(__name__)

# Default database path (file, not directory - KuzuDB 0.11+ requirement)
DEFAULT_KUZU_PATH = Path.home() / ".research_kb" / "kuzu" / "research_kb.kuzu"

# Singleton connection — NOT thread-safe despite earlier comment.
# Multiple asyncio.to_thread calls can hit the same connection concurrently,
# causing undefined behavior and OOM under load.
_db: Optional[kuzu.Database] = None
_conn: Optional[kuzu.Connection] = None

# Serialize all KuzuDB access through this lock.
# Prevents concurrent asyncio.to_thread(conn.execute, ...) calls from
# hitting the singleton connection simultaneously.
_kuzu_lock: asyncio.Lock = asyncio.Lock()


# Relationship type weights (mirrored from graph_queries.py)
RELATIONSHIP_WEIGHTS: dict[str, float] = {
    "REQUIRES": 1.0,
    "EXTENDS": 0.9,
    "USES": 0.8,
    "ADDRESSES": 0.7,
    "SPECIALIZES": 0.6,
    "GENERALIZES": 0.6,
    "ALTERNATIVE_TO": 0.5,
    "RELATED_TO": 0.4,
}


def get_kuzu_connection(db_path: Optional[Path] = None) -> kuzu.Connection:
    """Get or create KuzuDB connection (singleton).

    Args:
        db_path: Optional path to database directory. Defaults to ~/.research_kb/kuzu.db

    Returns:
        KuzuDB connection object

    Raises:
        StorageError: If database initialization fails
    """
    global _db, _conn

    if _conn is not None:
        return _conn

    path = db_path or DEFAULT_KUZU_PATH

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("initializing_kuzu_db", path=str(path))
        _db = kuzu.Database(str(path))
        _conn = kuzu.Connection(_db)

        # Initialize schema if needed
        _ensure_schema(_conn)

        logger.info("kuzu_db_initialized", path=str(path))
        return _conn

    except Exception as e:
        logger.error("kuzu_init_failed", path=str(path), error=str(e))
        raise StorageError(f"Failed to initialize KuzuDB: {e}") from e


def close_kuzu_connection() -> None:
    """Close KuzuDB connection."""
    global _db, _conn

    if _conn is not None:
        _conn = None
    if _db is not None:
        _db = None

    logger.info("kuzu_connection_closed")


def _ensure_schema(conn: kuzu.Connection) -> None:
    """Ensure KuzuDB schema exists.

    Creates node and edge tables if they don't exist.
    Idempotent - safe to call multiple times.
    """
    # Check if Concept table exists by querying schema
    try:
        result = conn.execute("CALL show_tables() RETURN *")
        df = result.get_as_df()
        tables = set(df["name"].tolist()) if not df.empty else set()

        if "Concept" not in tables:
            logger.info("creating_kuzu_schema")

            # Node table for concepts
            conn.execute("""
                CREATE NODE TABLE Concept(
                    id STRING PRIMARY KEY,
                    name STRING,
                    canonical_name STRING,
                    concept_type STRING
                )
            """)

            # Edge table for relationships
            conn.execute("""
                CREATE REL TABLE RELATES(
                    FROM Concept TO Concept,
                    relationship_type STRING,
                    strength DOUBLE DEFAULT 1.0
                )
            """)

            logger.info("kuzu_schema_created")
        else:
            logger.debug("kuzu_schema_exists")

    except Exception as e:
        logger.error("schema_check_failed", error=str(e))
        raise


def get_relationship_weight(rel_type: str) -> float:
    """Get scoring weight for a relationship type.

    Args:
        rel_type: The relationship type string

    Returns:
        Weight value (0.4-1.0)
    """
    return RELATIONSHIP_WEIGHTS.get(rel_type, 0.5)


# =============================================================================
# Core Graph Queries
# =============================================================================


async def find_shortest_path_kuzu(
    start_id: UUID,
    end_id: UUID,
    max_hops: int = 5,
) -> Optional[list[dict]]:
    """Find shortest path between two concepts using KuzuDB.

    Uses KuzuDB's SHORTEST path pattern for optimal performance.

    Args:
        start_id: Starting concept UUID
        end_id: Target concept UUID
        max_hops: Maximum path length to search

    Returns:
        List of dicts with concept info and relationships, or None if no path.
        Each dict: {"concept_id": str, "name": str, "rel_type": Optional[str]}
    """
    conn = get_kuzu_connection()

    try:
        async with _kuzu_lock:
            # Use KuzuDB's SHORTEST path pattern for optimal BFS
            # The SHORTEST keyword enables optimized shortest path algorithm
            # Use bidirectional edge pattern (-) for undirected traversal
            result = await asyncio.to_thread(
                conn.execute,
                f"""
                MATCH p = (a:Concept)-[r:RELATES* SHORTEST 1..{max_hops}]-(b:Concept)
                WHERE a.id = $start_id AND b.id = $end_id
                RETURN nodes(p) AS nodes, rels(p) AS rels, length(p) AS path_len
                LIMIT 1
                """,
                {"start_id": str(start_id), "end_id": str(end_id)},
            )

        df = result.get_as_df()
        if df.empty:
            return None

        row = df.iloc[0]
        nodes = row["nodes"]
        rels = row["rels"]

        # Build path with relationship info
        path = []
        for i, node in enumerate(nodes):
            entry = {
                "concept_id": node["id"],
                "name": node["name"],
                "canonical_name": node["canonical_name"],
                "concept_type": node["concept_type"],
                "rel_type": rels[i - 1]["relationship_type"] if i > 0 and rels else None,
                "rel_strength": rels[i - 1]["strength"] if i > 0 and rels else None,
            }
            path.append(entry)

        return path

    except Exception as e:
        logger.error(
            "kuzu_shortest_path_failed",
            start=str(start_id),
            end=str(end_id),
            error=str(e),
        )
        return None


async def find_shortest_path_length_kuzu(
    start_id: UUID,
    end_id: UUID,
    max_hops: int = 5,
) -> Optional[int]:
    """Find shortest path length between two concepts.

    Lightweight version that only returns distance.
    Uses KuzuDB's SHORTEST pattern for optimal performance.

    Args:
        start_id: Starting concept UUID
        end_id: Target concept UUID
        max_hops: Maximum path length to search

    Returns:
        Path length (number of edges), or None if no path
    """
    conn = get_kuzu_connection()

    try:
        async with _kuzu_lock:
            # Use bidirectional edge pattern for undirected traversal
            result = await asyncio.to_thread(
                conn.execute,
                f"""
                MATCH p = (a:Concept)-[r:RELATES* SHORTEST 1..{max_hops}]-(b:Concept)
                WHERE a.id = $start_id AND b.id = $end_id
                RETURN length(p) AS path_len
                LIMIT 1
                """,
                {"start_id": str(start_id), "end_id": str(end_id)},
            )

        df = result.get_as_df()
        if df.empty:
            return None

        return int(df.iloc[0]["path_len"])

    except Exception as e:
        logger.error(
            "kuzu_path_length_failed",
            start=str(start_id),
            end=str(end_id),
            error=str(e),
        )
        return None


async def get_neighborhood_kuzu(
    concept_id: UUID,
    hops: int = 2,
    relationship_type: Optional[str] = None,
) -> dict:
    """Get N-hop neighborhood of a concept.

    Args:
        concept_id: Center concept UUID
        hops: Number of hops to traverse
        relationship_type: Optional filter by relationship type

    Returns:
        Dictionary with:
        - 'center_id': The center concept ID
        - 'neighbors': List of neighbor concept dicts
        - 'relationships': List of relationship dicts
    """
    conn = get_kuzu_connection()

    try:
        # Build relationship type filter
        rel_filter = ""
        if relationship_type:
            rel_filter = f"AND r.relationship_type = '{relationship_type}'"

        async with _kuzu_lock:
            # Get neighbors within N hops
            result = await asyncio.to_thread(
                conn.execute,
                f"""
                MATCH (center:Concept)-[r:RELATES*1..{hops}]-(neighbor:Concept)
                WHERE center.id = $concept_id
                  AND neighbor.id <> $concept_id
                  {rel_filter}
                RETURN DISTINCT
                    neighbor.id AS neighbor_id,
                    neighbor.name AS name,
                    neighbor.canonical_name AS canonical_name,
                    neighbor.concept_type AS concept_type
                """,
                {"concept_id": str(concept_id)},
            )

        df = result.get_as_df()
        neighbors = df.to_dict("records") if not df.empty else []

        # Get relationships in neighborhood
        neighbor_ids = [n["neighbor_id"] for n in neighbors]
        all_ids = [str(concept_id)] + neighbor_ids

        async with _kuzu_lock:
            # Get edges within the neighborhood
            rel_result = await asyncio.to_thread(
                conn.execute,
                f"""
                MATCH (a:Concept)-[r:RELATES]->(b:Concept)
                WHERE a.id IN $all_ids AND b.id IN $all_ids
                  {rel_filter.replace('r.', 'r.')}
                RETURN
                    a.id AS source_id,
                    b.id AS target_id,
                    r.relationship_type AS relationship_type,
                    r.strength AS strength
                """,
                {"all_ids": all_ids},
            )

        rel_df = rel_result.get_as_df()
        relationships = rel_df.to_dict("records") if not rel_df.empty else []

        return {
            "center_id": str(concept_id),
            "neighbors": neighbors,
            "relationships": relationships,
        }

    except Exception as e:
        logger.error(
            "kuzu_neighborhood_failed",
            concept_id=str(concept_id),
            error=str(e),
        )
        raise StorageError(f"Failed to get neighborhood: {e}") from e


# =============================================================================
# Batch Graph Scoring (Key Performance Optimization)
# =============================================================================


async def compute_batch_graph_scores(
    query_concept_ids: list[UUID],
    chunk_concept_ids_list: list[list[UUID]],
    max_hops: int = 2,
) -> list[float]:
    """Compute graph scores for multiple chunks in a single batch.

    This is the key optimization: instead of N×M individual queries,
    we issue a single batch query for all pairs.

    Args:
        query_concept_ids: Concepts extracted from query
        chunk_concept_ids_list: List of concept ID lists, one per chunk

    Returns:
        List of normalized graph scores, one per chunk
    """
    if not query_concept_ids:
        return [0.0] * len(chunk_concept_ids_list)

    conn = get_kuzu_connection()

    try:
        # Flatten all chunk concept IDs for batch query
        all_chunk_ids = set()
        for chunk_ids in chunk_concept_ids_list:
            all_chunk_ids.update(str(cid) for cid in chunk_ids)

        if not all_chunk_ids:
            return [0.0] * len(chunk_concept_ids_list)

        # Single batch query: find all shortest paths from query concepts to chunk concepts
        # Using SHORTEST pattern for optimal performance
        query_ids_str = [str(qid) for qid in query_concept_ids]

        async with _kuzu_lock:
            # Use bidirectional edge pattern for undirected traversal
            result = await asyncio.to_thread(
                conn.execute,
                f"""
                MATCH p = (q:Concept)-[r:RELATES* SHORTEST 1..{max_hops}]-(c:Concept)
                WHERE q.id IN $query_ids AND c.id IN $chunk_ids
                RETURN
                    q.id AS query_id,
                    c.id AS chunk_id,
                    length(p) AS shortest_len,
                    rels(p) AS path_rels
                """,
                {"query_ids": query_ids_str, "chunk_ids": list(all_chunk_ids)},
            )

        df = result.get_as_df()

        # Build lookup: (query_id, chunk_id) -> (path_len, rel_types)
        path_lookup = {}
        if not df.empty:
            for _, row in df.iterrows():
                key = (row["query_id"], row["chunk_id"])
                # Extract relationship types from raw rel dicts
                path_rels = row["path_rels"] or []
                rel_types = [r.get("relationship_type", "RELATED_TO") for r in path_rels]
                path_lookup[key] = (row["shortest_len"], rel_types)

        # Compute scores per chunk
        scores = []
        for chunk_ids in chunk_concept_ids_list:
            if not chunk_ids:
                scores.append(0.0)
                continue

            chunk_score = 0.0
            max_pairs = len(query_concept_ids) * len(chunk_ids)

            for qid in query_concept_ids:
                for cid in chunk_ids:
                    key = (str(qid), str(cid))
                    if key in path_lookup:
                        path_len, rel_types = path_lookup[key]

                        # Compute weighted score
                        if path_len == 0:
                            path_weight = 1.0
                        else:
                            path_weight = 1.0
                            if rel_types:
                                for rt in rel_types:
                                    path_weight *= get_relationship_weight(rt)

                        chunk_score += path_weight / (path_len + 1)

            normalized = min(chunk_score / max_pairs, 1.0) if max_pairs > 0 else 0.0
            scores.append(normalized)

        return scores

    except Exception as e:
        logger.error("kuzu_batch_scoring_failed", error=str(e))
        return [0.0] * len(chunk_concept_ids_list)


async def compute_single_graph_score(
    query_concept_ids: list[UUID],
    chunk_concept_ids: list[UUID],
    max_hops: int = 2,
) -> tuple[float, list[str]]:
    """Compute weighted graph score for a single chunk.

    Compatible signature with compute_weighted_graph_score from graph_queries.py.

    Args:
        query_concept_ids: Concepts from query
        chunk_concept_ids: Concepts in candidate chunk
        max_hops: Maximum path length to consider

    Returns:
        Tuple of (normalized_score, list_of_explanations)
    """
    if not query_concept_ids or not chunk_concept_ids:
        return 0.0, []

    conn = get_kuzu_connection()

    try:
        query_ids_str = [str(qid) for qid in query_concept_ids]
        chunk_ids_str = [str(cid) for cid in chunk_concept_ids]

        async with _kuzu_lock:
            # Get shortest paths with relationship details using SHORTEST pattern
            # Note: KuzuDB doesn't support list comprehensions in Cypher, so we fetch
            # raw nodes/rels and process in Python
            # Use bidirectional edge pattern for undirected traversal
            result = await asyncio.to_thread(
                conn.execute,
                f"""
                MATCH p = (q:Concept)-[r:RELATES* SHORTEST 1..{max_hops}]-(c:Concept)
                WHERE q.id IN $query_ids AND c.id IN $chunk_ids
                RETURN
                    q.id AS query_id,
                    c.id AS chunk_id,
                    length(p) AS path_len,
                    nodes(p) AS path_nodes,
                    rels(p) AS path_rels
                """,
                {"query_ids": query_ids_str, "chunk_ids": chunk_ids_str},
            )

        df = result.get_as_df()

        total_score = 0.0
        max_pairs = len(query_concept_ids) * len(chunk_concept_ids)
        path_scores: list[tuple[float, str]] = []
        seen_pairs = set()

        if not df.empty:
            for _, row in df.iterrows():
                pair_key = (row["query_id"], row["chunk_id"])
                if pair_key in seen_pairs:
                    continue  # Only count shortest path per pair
                seen_pairs.add(pair_key)

                path_len = row["path_len"]
                path_nodes = row["path_nodes"] or []
                path_rels = row["path_rels"] or []

                # Extract names and relationship types from raw node/rel dicts
                node_names = [n.get("canonical_name") or n.get("name", "?") for n in path_nodes]
                rel_types = [r.get("relationship_type", "RELATED_TO") for r in path_rels]

                # Compute weighted score
                path_weight = 1.0
                for rt in rel_types:
                    path_weight *= get_relationship_weight(rt)

                score_contrib = path_weight / (path_len + 1)
                total_score += score_contrib

                # Build explanation
                if node_names:
                    parts = [node_names[0]]
                    for i, name in enumerate(node_names[1:], 1):
                        rt = rel_types[i - 1].lower() if i - 1 < len(rel_types) else "related"
                        parts.append(f"→ ({rt}) → {name}")
                    explanation = " ".join(parts)
                    path_scores.append((score_contrib, explanation))

        normalized = min(total_score / max_pairs, 1.0) if max_pairs > 0 else 0.0

        # Get top explanations
        path_scores.sort(key=lambda x: x[0], reverse=True)
        top_explanations = [exp for _, exp in path_scores[:3]]

        return normalized, top_explanations

    except Exception as e:
        logger.error("kuzu_single_score_failed", error=str(e))
        return 0.0, []


# =============================================================================
# Data Management
# =============================================================================


async def clear_all_data() -> int:
    """Clear all data from KuzuDB (for re-sync).

    Returns:
        Number of concepts deleted
    """
    conn = get_kuzu_connection()

    try:
        async with _kuzu_lock:
            # Get count first
            count_result = await asyncio.to_thread(
                conn.execute, "MATCH (c:Concept) RETURN count(c) AS cnt"
            )
            count = int(count_result.get_as_df().iloc[0]["cnt"])

            # Delete all relationships first (required before deleting nodes)
            await asyncio.to_thread(conn.execute, "MATCH ()-[r:RELATES]->() DELETE r")

            # Delete all concepts
            await asyncio.to_thread(conn.execute, "MATCH (c:Concept) DELETE c")

        logger.info("kuzu_data_cleared", concepts_deleted=count)
        return count

    except Exception as e:
        logger.error("kuzu_clear_failed", error=str(e))
        raise StorageError(f"Failed to clear KuzuDB: {e}") from e


async def get_stats() -> dict:
    """Get KuzuDB statistics.

    Returns:
        Dict with concept_count, relationship_count, relationship_types
    """
    conn = get_kuzu_connection()

    try:
        async with _kuzu_lock:
            # Concept count
            c_result = await asyncio.to_thread(
                conn.execute, "MATCH (c:Concept) RETURN count(c) AS cnt"
            )
            concept_count = int(c_result.get_as_df().iloc[0]["cnt"])

            # Relationship count
            r_result = await asyncio.to_thread(
                conn.execute, "MATCH ()-[r:RELATES]->() RETURN count(r) AS cnt"
            )
            rel_count = int(r_result.get_as_df().iloc[0]["cnt"])

            # Relationship type distribution
            type_result = await asyncio.to_thread(
                conn.execute,
                """
                MATCH ()-[r:RELATES]->()
                RETURN r.relationship_type AS type, count(*) AS cnt
                ORDER BY cnt DESC
                """,
            )
        type_df = type_result.get_as_df()
        rel_types = dict(zip(type_df["type"], type_df["cnt"])) if not type_df.empty else {}

        return {
            "concept_count": concept_count,
            "relationship_count": rel_count,
            "relationship_types": rel_types,
        }

    except Exception as e:
        logger.error("kuzu_stats_failed", error=str(e))
        return {"concept_count": 0, "relationship_count": 0, "relationship_types": {}}


async def bulk_insert_concepts(concepts: list[dict]) -> int:
    """Bulk insert concepts into KuzuDB using COPY FROM (fast path).

    Args:
        concepts: List of dicts with id, name, canonical_name, concept_type

    Returns:
        Number of concepts inserted
    """
    import csv
    import tempfile
    import os

    if not concepts:
        return 0

    conn = get_kuzu_connection()

    try:
        # Write to CSV for bulk loading (KuzuDB optimized path)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            csv_path = f.name
            writer = csv.writer(f)
            # KuzuDB expects no header row for COPY FROM by default

            for c in concepts:
                # Escape newlines and tabs in text fields
                name = str(c["name"]).replace("\n", " ").replace("\t", " ")
                canonical = str(c["canonical_name"]).replace("\n", " ").replace("\t", " ")
                writer.writerow([
                    str(c["id"]),
                    name,
                    canonical,
                    c["concept_type"],
                ])

        logger.info("concepts_csv_written", path=csv_path, count=len(concepts))

        async with _kuzu_lock:
            # Use COPY FROM for bulk insert
            await asyncio.to_thread(
                conn.execute,
                f"COPY Concept FROM '{csv_path}' (HEADER=false)"
            )

        # Cleanup CSV
        os.unlink(csv_path)

        logger.info("kuzu_concepts_inserted", count=len(concepts))
        return len(concepts)

    except Exception as e:
        logger.error("kuzu_concept_insert_failed", error=str(e))
        raise StorageError(f"Failed to insert concepts: {e}") from e


async def bulk_insert_relationships(relationships: list[dict]) -> int:
    """Bulk insert relationships into KuzuDB using COPY FROM (fast path).

    Args:
        relationships: List of dicts with source_id, target_id, relationship_type, strength

    Returns:
        Number of relationships inserted
    """
    import csv
    import tempfile
    import os

    if not relationships:
        return 0

    conn = get_kuzu_connection()

    try:
        # Write to CSV for bulk loading
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            csv_path = f.name
            writer = csv.writer(f)

            for r in relationships:
                writer.writerow([
                    str(r["source_id"]),
                    str(r["target_id"]),
                    r["relationship_type"],
                    float(r.get("strength", 1.0)),
                ])

        logger.info("relationships_csv_written", path=csv_path, count=len(relationships))

        async with _kuzu_lock:
            # Use COPY FROM for bulk insert
            await asyncio.to_thread(
                conn.execute,
                f"COPY RELATES FROM '{csv_path}' (HEADER=false)"
            )

        # Cleanup CSV
        os.unlink(csv_path)

        logger.info("kuzu_relationships_inserted", count=len(relationships))
        return len(relationships)

    except Exception as e:
        logger.error("kuzu_rel_insert_failed", error=str(e))
        raise StorageError(f"Failed to insert relationships: {e}") from e
