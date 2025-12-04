"""Neo4j graph synchronization service.

Syncs concepts and relationships from PostgreSQL to Neo4j
for efficient graph traversal queries.
"""

from typing import Any, Optional
from uuid import UUID

from neo4j import AsyncGraphDatabase, AsyncDriver
from research_kb_common import get_logger

logger = get_logger(__name__)


class GraphSyncError(Exception):
    """Error during graph synchronization."""

    pass


class GraphSyncService:
    """Synchronize concepts/relationships from PostgreSQL to Neo4j.

    Neo4j provides efficient graph traversal for:
    - Finding related concepts (N-hop queries)
    - Shortest path between concepts
    - Graph-based search scoring

    PostgreSQL remains the authoritative data store.

    Example:
        >>> sync = GraphSyncService("bolt://localhost:7687")
        >>> await sync.sync_concept(concept)
        >>> path = await sync.find_shortest_path("IV", "unconfoundedness")
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "research_kb_dev",
    ):
        """Initialize graph sync service.

        Args:
            uri: Neo4j Bolt protocol URI
            username: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self._driver: Optional[AsyncDriver] = None

    async def _get_driver(self) -> AsyncDriver:
        """Get or create Neo4j driver."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
        return self._driver

    async def close(self) -> None:
        """Close Neo4j driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def is_available(self) -> bool:
        """Check if Neo4j is available."""
        try:
            driver = await self._get_driver()
            async with driver.session() as session:
                result = await session.run("RETURN 1 AS n")
                record = await result.single()
                return record is not None
        except Exception as e:
            logger.warning("neo4j_unavailable", error=str(e))
            return False

    async def sync_concept(
        self,
        concept_id: UUID,
        name: str,
        canonical_name: str,
        concept_type: str,
        definition: Optional[str] = None,
    ) -> None:
        """Upsert a concept as a Neo4j node.

        Args:
            concept_id: PostgreSQL UUID
            name: Concept display name
            canonical_name: Normalized canonical name
            concept_type: Type (method, assumption, etc.)
            definition: Optional definition text
        """
        driver = await self._get_driver()

        query = """
        MERGE (c:Concept {id: $id})
        SET c.name = $name,
            c.canonical_name = $canonical_name,
            c.concept_type = $concept_type,
            c.definition = $definition,
            c.updated_at = datetime()
        """

        try:
            async with driver.session() as session:
                await session.run(
                    query,
                    id=str(concept_id),
                    name=name,
                    canonical_name=canonical_name,
                    concept_type=concept_type,
                    definition=definition,
                )
                logger.debug("concept_synced", concept_id=str(concept_id))
        except Exception as e:
            logger.error(
                "concept_sync_failed", concept_id=str(concept_id), error=str(e)
            )
            raise GraphSyncError(f"Failed to sync concept: {e}") from e

    async def sync_relationship(
        self,
        relationship_id: UUID,
        source_concept_id: UUID,
        target_concept_id: UUID,
        relationship_type: str,
        strength: float = 1.0,
    ) -> None:
        """Upsert a relationship as a Neo4j edge.

        Args:
            relationship_id: PostgreSQL UUID
            source_concept_id: Source concept UUID
            target_concept_id: Target concept UUID
            relationship_type: Type (REQUIRES, USES, etc.)
            strength: Relationship strength 0.0-1.0
        """
        driver = await self._get_driver()

        # Neo4j doesn't support parameterized relationship types,
        # so we use a workaround with APOC or conditional creation
        query = """
        MATCH (s:Concept {id: $source_id})
        MATCH (t:Concept {id: $target_id})
        MERGE (s)-[r:RELATES_TO {type: $rel_type}]->(t)
        SET r.id = $rel_id,
            r.strength = $strength,
            r.updated_at = datetime()
        """

        try:
            async with driver.session() as session:
                await session.run(
                    query,
                    rel_id=str(relationship_id),
                    source_id=str(source_concept_id),
                    target_id=str(target_concept_id),
                    rel_type=relationship_type,
                    strength=strength,
                )
                logger.debug(
                    "relationship_synced",
                    relationship_id=str(relationship_id),
                    type=relationship_type,
                )
        except Exception as e:
            logger.error(
                "relationship_sync_failed",
                relationship_id=str(relationship_id),
                error=str(e),
            )
            raise GraphSyncError(f"Failed to sync relationship: {e}") from e

    async def find_related_concepts(
        self,
        concept_id: UUID,
        max_hops: int = 2,
    ) -> list[dict[str, Any]]:
        """Find concepts related within N hops.

        Args:
            concept_id: Starting concept UUID
            max_hops: Maximum relationship hops

        Returns:
            List of related concepts with path info
        """
        driver = await self._get_driver()

        query = """
        MATCH path = (start:Concept {id: $id})-[*1..$max_hops]-(related:Concept)
        WHERE related <> start
        RETURN DISTINCT
            related.id AS concept_id,
            related.name AS name,
            related.canonical_name AS canonical_name,
            related.concept_type AS concept_type,
            length(path) AS distance,
            [r IN relationships(path) | r.type] AS relationship_types
        ORDER BY distance ASC
        """

        try:
            async with driver.session() as session:
                result = await session.run(
                    query,
                    id=str(concept_id),
                    max_hops=max_hops,
                )
                records = await result.data()
                return records
        except Exception as e:
            logger.error("find_related_failed", error=str(e))
            return []

    async def find_shortest_path(
        self,
        start_canonical: str,
        end_canonical: str,
        max_hops: int = 5,
    ) -> Optional[dict[str, Any]]:
        """Find shortest path between two concepts.

        Args:
            start_canonical: Starting concept canonical name
            end_canonical: Ending concept canonical name
            max_hops: Maximum path length

        Returns:
            Path info dict or None if no path found
        """
        driver = await self._get_driver()

        query = """
        MATCH path = shortestPath(
            (start:Concept {canonical_name: $start})-[*1..$max_hops]-(end:Concept {canonical_name: $end})
        )
        RETURN
            [n IN nodes(path) | n.name] AS concept_path,
            [r IN relationships(path) | r.type] AS relationship_types,
            length(path) AS path_length
        """

        try:
            async with driver.session() as session:
                result = await session.run(
                    query,
                    start=start_canonical,
                    end=end_canonical,
                    max_hops=max_hops,
                )
                record = await result.single()
                if record:
                    return dict(record)
                return None
        except Exception as e:
            logger.error("find_path_failed", error=str(e))
            return None

    async def compute_graph_score(
        self,
        query_concept_ids: list[UUID],
        chunk_concept_ids: list[UUID],
        max_hops: int = 2,
    ) -> float:
        """Compute graph-based relevance score.

        Score = sum of (1 / (path_length + 1)) for connected pairs
        Normalized by max possible score.

        Args:
            query_concept_ids: Concepts from search query
            chunk_concept_ids: Concepts in a chunk

        Returns:
            Normalized score 0.0-1.0
        """
        if not query_concept_ids or not chunk_concept_ids:
            return 0.0

        driver = await self._get_driver()

        query = """
        UNWIND $query_ids AS q_id
        UNWIND $chunk_ids AS c_id
        MATCH path = shortestPath(
            (q:Concept {id: q_id})-[*1..$max_hops]-(c:Concept {id: c_id})
        )
        RETURN q_id, c_id, length(path) AS path_length
        """

        try:
            async with driver.session() as session:
                result = await session.run(
                    query,
                    query_ids=[str(id) for id in query_concept_ids],
                    chunk_ids=[str(id) for id in chunk_concept_ids],
                    max_hops=max_hops,
                )
                records = await result.data()

            # Compute score
            total_score = 0.0
            max_pairs = len(query_concept_ids) * len(chunk_concept_ids)

            for record in records:
                path_length = record["path_length"]
                # Direct link = 1.0, 1-hop = 0.5, 2-hop = 0.33
                total_score += 1.0 / (path_length + 1)

            return min(total_score / max_pairs, 1.0) if max_pairs > 0 else 0.0

        except Exception as e:
            logger.error("graph_score_failed", error=str(e))
            return 0.0

    async def clear_all(self) -> None:
        """Clear all nodes and relationships (for testing)."""
        driver = await self._get_driver()

        try:
            async with driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
                logger.warning("neo4j_cleared")
        except Exception as e:
            logger.error("clear_failed", error=str(e))
            raise GraphSyncError(f"Failed to clear graph: {e}") from e

    async def get_stats(self) -> dict[str, int]:
        """Get graph statistics."""
        driver = await self._get_driver()

        try:
            async with driver.session() as session:
                # Count nodes
                result = await session.run("MATCH (n:Concept) RETURN count(n) AS count")
                node_count = (await result.single())["count"]

                # Count relationships
                result = await session.run("MATCH ()-[r]->() RETURN count(r) AS count")
                rel_count = (await result.single())["count"]

                return {
                    "concepts": node_count,
                    "relationships": rel_count,
                }
        except Exception as e:
            logger.error("stats_failed", error=str(e))
            return {"concepts": 0, "relationships": 0}

    async def __aenter__(self) -> "GraphSyncService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
