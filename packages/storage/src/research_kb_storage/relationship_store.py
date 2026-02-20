"""RelationshipStore - CRUD operations for concept_relationships table.

Provides:
- Create relationship records
- Retrieve relationships by ID or concept pair
- List relationships for a concept (incoming/outgoing)
- Delete relationships
- Batch operations for extraction pipeline
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import asyncpg
from research_kb_common import StorageError, get_logger
from research_kb_contracts import ConceptRelationship, RelationshipType

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class RelationshipStore:
    """Storage operations for ConceptRelationship entities."""

    @staticmethod
    async def create(
        source_concept_id: UUID,
        target_concept_id: UUID,
        relationship_type: RelationshipType,
        is_directed: bool = True,
        strength: float = 1.0,
        evidence_chunk_ids: Optional[list[UUID]] = None,
        confidence_score: Optional[float] = None,
    ) -> ConceptRelationship:
        """Create a new relationship between concepts.

        Args:
            source_concept_id: Source concept UUID
            target_concept_id: Target concept UUID
            relationship_type: Type of relationship
            is_directed: Whether relationship is directed
            strength: Relationship strength 0.0-1.0
            evidence_chunk_ids: Chunks where relationship was observed
            confidence_score: Extraction confidence 0.0-1.0

        Returns:
            Created ConceptRelationship

        Raises:
            StorageError: If creation fails (e.g., duplicate edge)
        """
        pool = await get_connection_pool()
        rel_id = uuid4()
        now = datetime.now(timezone.utc)

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO concept_relationships (
                        id, source_concept_id, target_concept_id,
                        relationship_type, is_directed, strength,
                        evidence_chunk_ids, confidence_score, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING *
                    """,
                    rel_id,
                    source_concept_id,
                    target_concept_id,
                    relationship_type.value,
                    is_directed,
                    strength,
                    evidence_chunk_ids or [],
                    confidence_score,
                    now,
                )

                logger.info(
                    "relationship_created",
                    relationship_id=str(rel_id),
                    source=str(source_concept_id),
                    target=str(target_concept_id),
                    type=relationship_type.value,
                )

                return _row_to_relationship(row)

        except asyncpg.UniqueViolationError as e:
            logger.error(
                "relationship_creation_failed_duplicate",
                source=str(source_concept_id),
                target=str(target_concept_id),
                type=relationship_type.value,
                error=str(e),
            )
            raise StorageError(
                f"Relationship already exists: {source_concept_id} -[{relationship_type.value}]-> {target_concept_id}"
            ) from e
        except asyncpg.ForeignKeyViolationError as e:
            logger.error(
                "relationship_creation_failed_fk",
                source=str(source_concept_id),
                target=str(target_concept_id),
                error=str(e),
            )
            raise StorageError(
                f"One or both concepts do not exist: {source_concept_id}, {target_concept_id}"
            ) from e
        except Exception as e:
            logger.error("relationship_creation_failed", error=str(e))
            raise StorageError(f"Failed to create relationship: {e}") from e

    @staticmethod
    async def get_by_id(relationship_id: UUID) -> Optional[ConceptRelationship]:
        """Retrieve relationship by ID."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM concept_relationships WHERE id = $1",
                    relationship_id,
                )

                if row is None:
                    return None

                return _row_to_relationship(row)

        except Exception as e:
            logger.error(
                "relationship_get_failed",
                relationship_id=str(relationship_id),
                error=str(e),
            )
            raise StorageError(f"Failed to retrieve relationship: {e}") from e

    @staticmethod
    async def get_by_concepts(
        source_concept_id: UUID,
        target_concept_id: UUID,
        relationship_type: Optional[RelationshipType] = None,
    ) -> Optional[ConceptRelationship]:
        """Retrieve relationship by concept pair and optional type."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                if relationship_type:
                    row = await conn.fetchrow(
                        """
                        SELECT * FROM concept_relationships
                        WHERE source_concept_id = $1
                          AND target_concept_id = $2
                          AND relationship_type = $3
                        """,
                        source_concept_id,
                        target_concept_id,
                        relationship_type.value,
                    )
                else:
                    row = await conn.fetchrow(
                        """
                        SELECT * FROM concept_relationships
                        WHERE source_concept_id = $1
                          AND target_concept_id = $2
                        """,
                        source_concept_id,
                        target_concept_id,
                    )

                if row is None:
                    return None

                return _row_to_relationship(row)

        except Exception as e:
            logger.error("relationship_get_by_concepts_failed", error=str(e))
            raise StorageError(f"Failed to retrieve relationship: {e}") from e

    @staticmethod
    async def list_from_concept(
        concept_id: UUID,
        relationship_type: Optional[RelationshipType] = None,
        limit: int = 100,
    ) -> list[ConceptRelationship]:
        """List outgoing relationships from a concept."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                if relationship_type:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM concept_relationships
                        WHERE source_concept_id = $1
                          AND relationship_type = $2
                        ORDER BY strength DESC
                        LIMIT $3
                        """,
                        concept_id,
                        relationship_type.value,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM concept_relationships
                        WHERE source_concept_id = $1
                        ORDER BY strength DESC
                        LIMIT $2
                        """,
                        concept_id,
                        limit,
                    )

                return [_row_to_relationship(row) for row in rows]

        except Exception as e:
            logger.error(
                "relationship_list_from_failed",
                concept_id=str(concept_id),
                error=str(e),
            )
            raise StorageError(f"Failed to list relationships: {e}") from e

    @staticmethod
    async def list_to_concept(
        concept_id: UUID,
        relationship_type: Optional[RelationshipType] = None,
        limit: int = 100,
    ) -> list[ConceptRelationship]:
        """List incoming relationships to a concept."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                if relationship_type:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM concept_relationships
                        WHERE target_concept_id = $1
                          AND relationship_type = $2
                        ORDER BY strength DESC
                        LIMIT $3
                        """,
                        concept_id,
                        relationship_type.value,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM concept_relationships
                        WHERE target_concept_id = $1
                        ORDER BY strength DESC
                        LIMIT $2
                        """,
                        concept_id,
                        limit,
                    )

                return [_row_to_relationship(row) for row in rows]

        except Exception as e:
            logger.error(
                "relationship_list_to_failed",
                concept_id=str(concept_id),
                error=str(e),
            )
            raise StorageError(f"Failed to list relationships: {e}") from e

    @staticmethod
    async def list_all_for_concept(
        concept_id: UUID,
        limit: int = 100,
    ) -> list[ConceptRelationship]:
        """List all relationships involving a concept (both directions)."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM concept_relationships
                    WHERE source_concept_id = $1
                       OR (target_concept_id = $1 AND NOT is_directed)
                    ORDER BY strength DESC
                    LIMIT $2
                    """,
                    concept_id,
                    limit,
                )

                return [_row_to_relationship(row) for row in rows]

        except Exception as e:
            logger.error(
                "relationship_list_all_failed",
                concept_id=str(concept_id),
                error=str(e),
            )
            raise StorageError(f"Failed to list relationships: {e}") from e

    @staticmethod
    async def delete(relationship_id: UUID) -> bool:
        """Delete a relationship."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM concept_relationships WHERE id = $1",
                    relationship_id,
                )

                deleted = result == "DELETE 1"

                if deleted:
                    logger.info("relationship_deleted", relationship_id=str(relationship_id))
                else:
                    logger.warning(
                        "relationship_not_found_for_delete",
                        relationship_id=str(relationship_id),
                    )

                return deleted

        except Exception as e:
            logger.error(
                "relationship_delete_failed",
                relationship_id=str(relationship_id),
                error=str(e),
            )
            raise StorageError(f"Failed to delete relationship: {e}") from e

    @staticmethod
    async def count() -> int:
        """Count total relationships."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT COUNT(*) FROM concept_relationships")
                return result or 0

        except Exception as e:
            logger.error("relationship_count_failed", error=str(e))
            raise StorageError(f"Failed to count relationships: {e}") from e

    @staticmethod
    async def batch_create(relationships_data: list[dict]) -> list[ConceptRelationship]:
        """Batch create multiple relationships.

        Args:
            relationships_data: List of dicts with relationship fields

        Returns:
            List of created ConceptRelationships
        """
        if not relationships_data:
            return []

        pool = await get_connection_pool()
        now = datetime.now(timezone.utc)
        created_rels = []

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for data in relationships_data:
                        rel_id = uuid4()
                        try:
                            row = await conn.fetchrow(
                                """
                                INSERT INTO concept_relationships (
                                    id, source_concept_id, target_concept_id,
                                    relationship_type, is_directed, strength,
                                    evidence_chunk_ids, confidence_score, created_at
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                                RETURNING *
                                """,
                                rel_id,
                                data["source_concept_id"],
                                data["target_concept_id"],
                                data["relationship_type"],
                                data.get("is_directed", True),
                                data.get("strength", 1.0),
                                data.get("evidence_chunk_ids", []),
                                data.get("confidence_score"),
                                now,
                            )
                            created_rels.append(_row_to_relationship(row))
                        except asyncpg.UniqueViolationError:
                            logger.warning(
                                "relationship_batch_skip_duplicate",
                                source=str(data["source_concept_id"]),
                                target=str(data["target_concept_id"]),
                            )
                        except asyncpg.ForeignKeyViolationError:
                            logger.warning(
                                "relationship_batch_skip_missing_concept",
                                source=str(data["source_concept_id"]),
                                target=str(data["target_concept_id"]),
                            )

                logger.info(
                    "relationships_batch_created",
                    count=len(created_rels),
                )

                return created_rels

        except Exception as e:
            logger.error("relationship_batch_create_failed", error=str(e))
            raise StorageError(f"Failed to batch create relationships: {e}") from e


def _row_to_relationship(row: asyncpg.Record) -> ConceptRelationship:
    """Convert database row to ConceptRelationship model."""
    return ConceptRelationship(
        id=row["id"],
        source_concept_id=row["source_concept_id"],
        target_concept_id=row["target_concept_id"],
        relationship_type=RelationshipType(row["relationship_type"]),
        is_directed=row["is_directed"],
        strength=row["strength"],
        evidence_chunk_ids=row["evidence_chunk_ids"] or [],
        confidence_score=row["confidence_score"],
        created_at=row["created_at"],
    )
