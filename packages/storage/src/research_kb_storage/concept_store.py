"""ConceptStore - CRUD operations for concepts table.

Provides:
- Create concept records (with deduplication by canonical_name)
- Retrieve concepts by ID or canonical name
- Update concept metadata and validation status
- Delete concepts (cascades to relationships and chunk_concepts)
- List concepts with filtering by type/category
- Batch operations for extraction pipeline
"""

import json
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import asyncpg
from pgvector.asyncpg import register_vector
from research_kb_common import StorageError, get_logger
from research_kb_contracts import Concept, ConceptType

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class ConceptStore:
    """Storage operations for Concept entities.

    All operations use the global connection pool.
    """

    @staticmethod
    async def create(
        name: str,
        canonical_name: str,
        concept_type: ConceptType,
        aliases: Optional[list[str]] = None,
        category: Optional[str] = None,
        definition: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        extraction_method: Optional[str] = None,
        confidence_score: Optional[float] = None,
        validated: bool = False,
        metadata: Optional[dict] = None,
    ) -> Concept:
        """Create a new concept record.

        Args:
            name: Display name as it appears in text
            canonical_name: Normalized unique name (must be unique)
            concept_type: Classification (method, assumption, etc.)
            aliases: Alternative names/abbreviations
            category: Subcategory (identification, estimation, testing)
            definition: Concept definition text
            embedding: 1024-dim BGE-large-en-v1.5 embedding
            extraction_method: How concept was extracted
            confidence_score: Extraction confidence 0.0-1.0
            validated: Whether manually reviewed
            metadata: Extensible JSONB metadata

        Returns:
            Created Concept

        Raises:
            StorageError: If creation fails (e.g., duplicate canonical_name)
        """
        pool = await get_connection_pool()
        concept_id = uuid4()
        now = datetime.now(timezone.utc)

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    """
                    INSERT INTO concepts (
                        id, name, canonical_name, aliases, concept_type,
                        category, definition, embedding,
                        extraction_method, confidence_score, validated,
                        metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING *
                    """,
                    concept_id,
                    name,
                    canonical_name,
                    aliases or [],
                    concept_type.value,
                    category,
                    definition,
                    embedding,
                    extraction_method,
                    confidence_score,
                    validated,
                    metadata or {},
                    now,
                )

                logger.info(
                    "concept_created",
                    concept_id=str(concept_id),
                    canonical_name=canonical_name,
                    concept_type=concept_type.value,
                )

                return _row_to_concept(row)

        except asyncpg.UniqueViolationError as e:
            logger.error(
                "concept_creation_failed_duplicate",
                canonical_name=canonical_name,
                error=str(e),
            )
            raise StorageError(
                f"Concept with canonical_name '{canonical_name}' already exists"
            ) from e
        except Exception as e:
            logger.error("concept_creation_failed", error=str(e))
            raise StorageError(f"Failed to create concept: {e}") from e

    @staticmethod
    async def get_by_id(concept_id: UUID) -> Optional[Concept]:
        """Retrieve concept by ID."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    "SELECT * FROM concepts WHERE id = $1",
                    concept_id,
                )

                if row is None:
                    return None

                return _row_to_concept(row)

        except Exception as e:
            logger.error("concept_get_failed", concept_id=str(concept_id), error=str(e))
            raise StorageError(f"Failed to retrieve concept: {e}") from e

    @staticmethod
    async def get_by_canonical_name(canonical_name: str) -> Optional[Concept]:
        """Retrieve concept by canonical name.

        Useful for deduplication during extraction.
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    "SELECT * FROM concepts WHERE canonical_name = $1",
                    canonical_name,
                )

                if row is None:
                    return None

                return _row_to_concept(row)

        except Exception as e:
            logger.error(
                "concept_get_by_name_failed",
                canonical_name=canonical_name,
                error=str(e),
            )
            raise StorageError(f"Failed to retrieve concept: {e}") from e

    @staticmethod
    async def update(
        concept_id: UUID,
        definition: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        validated: Optional[bool] = None,
        metadata: Optional[dict] = None,
    ) -> Concept:
        """Update concept fields.

        Only provided fields are updated.
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                # Build dynamic update
                updates = []
                params = [concept_id]
                param_idx = 2

                if definition is not None:
                    updates.append(f"definition = ${param_idx}")
                    params.append(definition)
                    param_idx += 1

                if embedding is not None:
                    updates.append(f"embedding = ${param_idx}")
                    params.append(embedding)
                    param_idx += 1

                if validated is not None:
                    updates.append(f"validated = ${param_idx}")
                    params.append(validated)
                    param_idx += 1

                if metadata is not None:
                    updates.append(f"metadata = metadata || ${param_idx}")
                    params.append(metadata)
                    param_idx += 1

                if not updates:
                    # Nothing to update, just return current
                    return await ConceptStore.get_by_id(concept_id)

                sql = f"""
                    UPDATE concepts
                    SET {', '.join(updates)}
                    WHERE id = $1
                    RETURNING *
                """

                row = await conn.fetchrow(sql, *params)

                if row is None:
                    raise StorageError(f"Concept not found: {concept_id}")

                logger.info("concept_updated", concept_id=str(concept_id))
                return _row_to_concept(row)

        except StorageError:
            raise
        except Exception as e:
            logger.error(
                "concept_update_failed", concept_id=str(concept_id), error=str(e)
            )
            raise StorageError(f"Failed to update concept: {e}") from e

    @staticmethod
    async def delete(concept_id: UUID) -> bool:
        """Delete concept and all associated relationships."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM concepts WHERE id = $1",
                    concept_id,
                )

                deleted = result == "DELETE 1"

                if deleted:
                    logger.info("concept_deleted", concept_id=str(concept_id))
                else:
                    logger.warning(
                        "concept_not_found_for_delete", concept_id=str(concept_id)
                    )

                return deleted

        except Exception as e:
            logger.error(
                "concept_delete_failed", concept_id=str(concept_id), error=str(e)
            )
            raise StorageError(f"Failed to delete concept: {e}") from e

    @staticmethod
    async def list_by_type(
        concept_type: ConceptType,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Concept]:
        """List concepts by type with pagination."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                rows = await conn.fetch(
                    """
                    SELECT * FROM concepts
                    WHERE concept_type = $1
                    ORDER BY canonical_name ASC
                    LIMIT $2 OFFSET $3
                    """,
                    concept_type.value,
                    limit,
                    offset,
                )

                return [_row_to_concept(row) for row in rows]

        except Exception as e:
            logger.error(
                "concept_list_failed",
                concept_type=concept_type.value,
                error=str(e),
            )
            raise StorageError(f"Failed to list concepts: {e}") from e

    @staticmethod
    async def list_all(limit: int = 50000, offset: int = 0) -> list[Concept]:
        """List all concepts with pagination."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                rows = await conn.fetch(
                    """
                    SELECT * FROM concepts
                    ORDER BY canonical_name ASC
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset,
                )

                return [_row_to_concept(row) for row in rows]

        except Exception as e:
            logger.error("concept_list_all_failed", error=str(e))
            raise StorageError(f"Failed to list concepts: {e}") from e

    @staticmethod
    async def count() -> int:
        """Count total concepts."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT COUNT(*) FROM concepts")
                return result or 0

        except Exception as e:
            logger.error("concept_count_failed", error=str(e))
            raise StorageError(f"Failed to count concepts: {e}") from e

    @staticmethod
    async def batch_create(concepts_data: list[dict]) -> list[Concept]:
        """Batch create multiple concepts (for extraction pipeline).

        Args:
            concepts_data: List of dicts with concept fields

        Returns:
            List of created Concepts
        """
        if not concepts_data:
            return []

        pool = await get_connection_pool()
        now = datetime.now(timezone.utc)
        created_concepts = []

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                async with conn.transaction():
                    for data in concepts_data:
                        concept_id = uuid4()
                        try:
                            row = await conn.fetchrow(
                                """
                                INSERT INTO concepts (
                                    id, name, canonical_name, aliases, concept_type,
                                    category, definition, embedding,
                                    extraction_method, confidence_score, validated,
                                    metadata, created_at
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                                RETURNING *
                                """,
                                concept_id,
                                data["name"],
                                data["canonical_name"],
                                data.get("aliases", []),
                                data["concept_type"],
                                data.get("category"),
                                data.get("definition"),
                                data.get("embedding"),
                                data.get("extraction_method"),
                                data.get("confidence_score"),
                                data.get("validated", False),
                                data.get("metadata", {}),
                                now,
                            )
                            created_concepts.append(_row_to_concept(row))
                        except asyncpg.UniqueViolationError:
                            # Skip duplicates, log warning
                            logger.warning(
                                "concept_batch_skip_duplicate",
                                canonical_name=data["canonical_name"],
                            )

                logger.info(
                    "concepts_batch_created",
                    count=len(created_concepts),
                )

                return created_concepts

        except Exception as e:
            logger.error("concept_batch_create_failed", error=str(e))
            raise StorageError(f"Failed to batch create concepts: {e}") from e

    @staticmethod
    async def search(
        query: str,
        limit: int = 50,
        concept_type: Optional[ConceptType] = None,
    ) -> list[Concept]:
        """Search concepts by name using fuzzy text matching.

        Args:
            query: Search query (matches against name, canonical_name, aliases)
            limit: Maximum results to return
            concept_type: Optional filter by concept type

        Returns:
            List of matching Concepts ordered by relevance
        """
        pool = await get_connection_pool()
        query_pattern = f"%{query.lower()}%"

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                if concept_type:
                    rows = await conn.fetch(
                        """
                        SELECT *,
                               CASE
                                   WHEN LOWER(canonical_name) = $1 THEN 1.0
                                   WHEN LOWER(canonical_name) LIKE $2 THEN 0.9
                                   WHEN LOWER(name) LIKE $2 THEN 0.8
                                   ELSE 0.5
                               END AS relevance
                        FROM concepts
                        WHERE concept_type = $3 AND (
                            LOWER(canonical_name) LIKE $2
                            OR LOWER(name) LIKE $2
                            OR EXISTS (
                                SELECT 1 FROM unnest(aliases) AS alias
                                WHERE LOWER(alias) LIKE $2
                            )
                        )
                        ORDER BY relevance DESC, canonical_name ASC
                        LIMIT $4
                        """,
                        query.lower(),
                        query_pattern,
                        concept_type.value,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT *,
                               CASE
                                   WHEN LOWER(canonical_name) = $1 THEN 1.0
                                   WHEN LOWER(canonical_name) LIKE $2 THEN 0.9
                                   WHEN LOWER(name) LIKE $2 THEN 0.8
                                   ELSE 0.5
                               END AS relevance
                        FROM concepts
                        WHERE LOWER(canonical_name) LIKE $2
                            OR LOWER(name) LIKE $2
                            OR EXISTS (
                                SELECT 1 FROM unnest(aliases) AS alias
                                WHERE LOWER(alias) LIKE $2
                            )
                        ORDER BY relevance DESC, canonical_name ASC
                        LIMIT $3
                        """,
                        query.lower(),
                        query_pattern,
                        limit,
                    )

                return [_row_to_concept(row) for row in rows]

        except Exception as e:
            logger.error("concept_search_failed", query=query[:50], error=str(e))
            raise StorageError(f"Failed to search concepts: {e}") from e

    @staticmethod
    async def get(concept_id: UUID) -> Optional[Concept]:
        """Alias for get_by_id (matches API service expectations)."""
        return await ConceptStore.get_by_id(concept_id)

    @staticmethod
    async def find_similar(
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.8,
    ) -> list[tuple[Concept, float]]:
        """Find concepts similar to an embedding.

        Args:
            embedding: 1024-dim query embedding
            limit: Maximum results
            threshold: Minimum similarity (0.0-1.0)

        Returns:
            List of (Concept, similarity_score) tuples
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                # Convert distance to similarity
                rows = await conn.fetch(
                    """
                    SELECT *,
                           1.0 - (embedding <=> $1::vector(1024)) / 2.0 AS similarity
                    FROM concepts
                    WHERE embedding IS NOT NULL
                      AND 1.0 - (embedding <=> $1::vector(1024)) / 2.0 >= $2
                    ORDER BY embedding <=> $1::vector(1024) ASC
                    LIMIT $3
                    """,
                    embedding,
                    threshold,
                    limit,
                )

                return [(_row_to_concept(row), row["similarity"]) for row in rows]

        except Exception as e:
            logger.error("concept_find_similar_failed", error=str(e))
            raise StorageError(f"Failed to find similar concepts: {e}") from e


def _row_to_concept(row: asyncpg.Record) -> Concept:
    """Convert database row to Concept model."""
    return Concept(
        id=row["id"],
        name=row["name"],
        canonical_name=row["canonical_name"],
        aliases=row["aliases"] or [],
        concept_type=ConceptType(row["concept_type"]),
        category=row["category"],
        definition=row["definition"],
        embedding=list(row["embedding"]) if row["embedding"] is not None else None,
        extraction_method=row["extraction_method"],
        confidence_score=row["confidence_score"],
        validated=row["validated"],
        metadata=row["metadata"] or {},
        created_at=row["created_at"],
    )
