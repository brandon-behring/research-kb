"""ChunkConceptStore - CRUD operations for chunk_concepts junction table.

Provides:
- Link chunks to concepts
- List concepts for a chunk
- List chunks for a concept
- Batch operations for extraction pipeline
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import asyncpg
from research_kb_common import StorageError, get_logger
from research_kb_contracts import ChunkConcept

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class ChunkConceptStore:
    """Storage operations for chunk-concept links."""

    @staticmethod
    async def create(
        chunk_id: UUID,
        concept_id: UUID,
        mention_type: str = "reference",
        relevance_score: Optional[float] = None,
    ) -> ChunkConcept:
        """Link a chunk to a concept.

        Args:
            chunk_id: Chunk UUID
            concept_id: Concept UUID
            mention_type: How concept appears (defines, reference, example)
            relevance_score: Relevance of concept to chunk 0.0-1.0

        Returns:
            Created ChunkConcept

        Raises:
            StorageError: If link already exists or FK violation
        """
        pool = await get_connection_pool()
        now = datetime.now(timezone.utc)

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO chunk_concepts (
                        chunk_id, concept_id, mention_type,
                        relevance_score, created_at
                    ) VALUES ($1, $2, $3, $4, $5)
                    RETURNING *
                    """,
                    chunk_id,
                    concept_id,
                    mention_type,
                    relevance_score,
                    now,
                )

                logger.debug(
                    "chunk_concept_created",
                    chunk_id=str(chunk_id),
                    concept_id=str(concept_id),
                    mention_type=mention_type,
                )

                return _row_to_chunk_concept(row)

        except asyncpg.UniqueViolationError as e:
            logger.warning(
                "chunk_concept_already_exists",
                chunk_id=str(chunk_id),
                concept_id=str(concept_id),
                mention_type=mention_type,
            )
            raise StorageError(
                f"Link already exists: chunk {chunk_id} -> concept {concept_id} ({mention_type})"
            ) from e
        except asyncpg.ForeignKeyViolationError as e:
            logger.error(
                "chunk_concept_fk_error",
                chunk_id=str(chunk_id),
                concept_id=str(concept_id),
                error=str(e),
            )
            raise StorageError(f"Chunk or concept does not exist: {chunk_id}, {concept_id}") from e
        except Exception as e:
            logger.error("chunk_concept_creation_failed", error=str(e))
            raise StorageError(f"Failed to create chunk-concept link: {e}") from e

    @staticmethod
    async def list_concepts_for_chunk(
        chunk_id: UUID,
        mention_type: Optional[str] = None,
    ) -> list[ChunkConcept]:
        """List all concepts linked to a chunk."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                if mention_type:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM chunk_concepts
                        WHERE chunk_id = $1 AND mention_type = $2
                        ORDER BY relevance_score DESC NULLS LAST
                        """,
                        chunk_id,
                        mention_type,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM chunk_concepts
                        WHERE chunk_id = $1
                        ORDER BY relevance_score DESC NULLS LAST
                        """,
                        chunk_id,
                    )

                return [_row_to_chunk_concept(row) for row in rows]

        except Exception as e:
            logger.error(
                "chunk_concept_list_for_chunk_failed",
                chunk_id=str(chunk_id),
                error=str(e),
            )
            raise StorageError(f"Failed to list concepts for chunk: {e}") from e

    @staticmethod
    async def list_chunks_for_concept(
        concept_id: UUID,
        mention_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[ChunkConcept]:
        """List all chunks that mention a concept."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                if mention_type:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM chunk_concepts
                        WHERE concept_id = $1 AND mention_type = $2
                        ORDER BY relevance_score DESC NULLS LAST
                        LIMIT $3
                        """,
                        concept_id,
                        mention_type,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM chunk_concepts
                        WHERE concept_id = $1
                        ORDER BY relevance_score DESC NULLS LAST
                        LIMIT $2
                        """,
                        concept_id,
                        limit,
                    )

                return [_row_to_chunk_concept(row) for row in rows]

        except Exception as e:
            logger.error(
                "chunk_concept_list_for_concept_failed",
                concept_id=str(concept_id),
                error=str(e),
            )
            raise StorageError(f"Failed to list chunks for concept: {e}") from e

    @staticmethod
    async def delete(
        chunk_id: UUID,
        concept_id: UUID,
        mention_type: str = "reference",
    ) -> bool:
        """Delete a chunk-concept link."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM chunk_concepts
                    WHERE chunk_id = $1 AND concept_id = $2 AND mention_type = $3
                    """,
                    chunk_id,
                    concept_id,
                    mention_type,
                )

                deleted = result == "DELETE 1"

                if deleted:
                    logger.debug(
                        "chunk_concept_deleted",
                        chunk_id=str(chunk_id),
                        concept_id=str(concept_id),
                    )

                return deleted

        except Exception as e:
            logger.error(
                "chunk_concept_delete_failed",
                chunk_id=str(chunk_id),
                concept_id=str(concept_id),
                error=str(e),
            )
            raise StorageError(f"Failed to delete chunk-concept link: {e}") from e

    @staticmethod
    async def delete_all_for_chunk(chunk_id: UUID) -> int:
        """Delete all concept links for a chunk."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM chunk_concepts WHERE chunk_id = $1",
                    chunk_id,
                )

                # Parse "DELETE N" result
                count = int(result.split()[-1]) if result else 0

                logger.info(
                    "chunk_concepts_deleted_for_chunk",
                    chunk_id=str(chunk_id),
                    count=count,
                )

                return count

        except Exception as e:
            logger.error(
                "chunk_concept_delete_all_failed",
                chunk_id=str(chunk_id),
                error=str(e),
            )
            raise StorageError(f"Failed to delete chunk-concept links: {e}") from e

    @staticmethod
    async def count_for_concept(concept_id: UUID) -> int:
        """Count chunks mentioning a concept."""
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM chunk_concepts WHERE concept_id = $1",
                    concept_id,
                )
                return result or 0

        except Exception as e:
            logger.error(
                "chunk_concept_count_failed",
                concept_id=str(concept_id),
                error=str(e),
            )
            raise StorageError(f"Failed to count chunk-concept links: {e}") from e

    @staticmethod
    async def batch_create(links_data: list[dict]) -> list[ChunkConcept]:
        """Batch create chunk-concept links.

        Args:
            links_data: List of dicts with link fields:
                - chunk_id: UUID
                - concept_id: UUID
                - mention_type: str (optional, default "reference")
                - relevance_score: float (optional)

        Returns:
            List of created ChunkConcepts
        """
        if not links_data:
            return []

        pool = await get_connection_pool()
        now = datetime.now(timezone.utc)
        created_links = []

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for data in links_data:
                        try:
                            row = await conn.fetchrow(
                                """
                                INSERT INTO chunk_concepts (
                                    chunk_id, concept_id, mention_type,
                                    relevance_score, created_at
                                ) VALUES ($1, $2, $3, $4, $5)
                                RETURNING *
                                """,
                                data["chunk_id"],
                                data["concept_id"],
                                data.get("mention_type", "reference"),
                                data.get("relevance_score"),
                                now,
                            )
                            created_links.append(_row_to_chunk_concept(row))
                        except asyncpg.UniqueViolationError:
                            # Skip duplicates silently
                            pass
                        except asyncpg.ForeignKeyViolationError:
                            logger.warning(
                                "chunk_concept_batch_skip_missing",
                                chunk_id=str(data["chunk_id"]),
                                concept_id=str(data["concept_id"]),
                            )

                logger.info(
                    "chunk_concepts_batch_created",
                    count=len(created_links),
                )

                return created_links

        except Exception as e:
            logger.error("chunk_concept_batch_create_failed", error=str(e))
            raise StorageError(f"Failed to batch create chunk-concept links: {e}") from e

    @staticmethod
    async def get_concept_ids_for_chunks(
        chunk_ids: list[UUID],
    ) -> dict[UUID, list[UUID]]:
        """Get concept IDs for multiple chunks.

        Useful for batch graph scoring in search.

        Args:
            chunk_ids: List of chunk UUIDs

        Returns:
            Dict mapping chunk_id -> list of concept_ids
        """
        if not chunk_ids:
            return {}

        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT chunk_id, concept_id
                    FROM chunk_concepts
                    WHERE chunk_id = ANY($1)
                    """,
                    chunk_ids,
                )

                result: dict[UUID, list[UUID]] = {cid: [] for cid in chunk_ids}
                for row in rows:
                    result[row["chunk_id"]].append(row["concept_id"])

                return result

        except Exception as e:
            logger.error("chunk_concept_get_batch_failed", error=str(e))
            raise StorageError(f"Failed to get concept IDs for chunks: {e}") from e

    @staticmethod
    async def get_concept_info_for_chunks(
        chunk_ids: list[UUID],
    ) -> dict[UUID, list[tuple[UUID, str, float | None]]]:
        """Get concept info for multiple chunks including mention metadata.

        Enhanced version of get_concept_ids_for_chunks that returns mention_type
        and relevance_score for weighted graph scoring.

        Args:
            chunk_ids: List of chunk UUIDs

        Returns:
            Dict mapping chunk_id -> list of (concept_id, mention_type, relevance_score)
            - mention_type: 'defines', 'reference', or 'example'
            - relevance_score: Optional float 0.0-1.0

        Example:
            >>> info = await ChunkConceptStore.get_concept_info_for_chunks([chunk_id])
            >>> for concept_id, mention_type, relevance in info[chunk_id]:
            ...     print(f"{concept_id}: {mention_type} ({relevance})")
        """
        if not chunk_ids:
            return {}

        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT chunk_id, concept_id, mention_type, relevance_score
                    FROM chunk_concepts
                    WHERE chunk_id = ANY($1)
                    """,
                    chunk_ids,
                )

                result: dict[UUID, list[tuple[UUID, str, float | None]]] = {
                    cid: [] for cid in chunk_ids
                }
                for row in rows:
                    result[row["chunk_id"]].append(
                        (
                            row["concept_id"],
                            row["mention_type"] or "reference",
                            row["relevance_score"],
                        )
                    )

                return result

        except Exception as e:
            logger.error("chunk_concept_get_info_batch_failed", error=str(e))
            raise StorageError(f"Failed to get concept info for chunks: {e}") from e


def _row_to_chunk_concept(row: asyncpg.Record) -> ChunkConcept:
    """Convert database row to ChunkConcept model."""
    return ChunkConcept(
        chunk_id=row["chunk_id"],
        concept_id=row["concept_id"],
        mention_type=row["mention_type"],
        relevance_score=row["relevance_score"],
        created_at=row["created_at"],
    )
