"""MethodStore - CRUD operations for methods table.

Provides:
- Create method records with specialized attributes
- Retrieve methods by ID or concept_id
- Update method attributes
- Delete methods
- List all methods with pagination
- Count total methods

Methods table stores specialized attributes for method-type concepts (1:1 relationship).
"""

import json
from typing import Optional
from uuid import UUID, uuid4

import asyncpg
from research_kb_common import StorageError, get_logger
from research_kb_contracts import Method

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class MethodStore:
    """Storage operations for Method entities.

    All operations use the global connection pool.
    Methods are 1:1 with Concept records (concept_id is UNIQUE).
    """

    @staticmethod
    async def create(
        concept_id: UUID,
        required_assumptions: Optional[list[str]] = None,
        problem_types: Optional[list[str]] = None,
        common_estimators: Optional[list[str]] = None,
    ) -> Method:
        """Create a new method record for a concept.

        Args:
            concept_id: UUID of the associated concept (must exist)
            required_assumptions: List of assumption concept names
            problem_types: Problem types this method addresses (ATE, ATT, LATE, etc.)
            common_estimators: Common estimators used (OLS, 2SLS, matching, etc.)

        Returns:
            Created Method

        Raises:
            StorageError: If creation fails (e.g., duplicate concept_id, concept doesn't exist)
        """
        pool = await get_connection_pool()
        method_id = uuid4()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    """
                    INSERT INTO methods (
                        id, concept_id, required_assumptions,
                        problem_types, common_estimators
                    ) VALUES ($1, $2, $3, $4, $5)
                    RETURNING *
                    """,
                    method_id,
                    concept_id,
                    required_assumptions or [],
                    problem_types or [],
                    common_estimators or [],
                )

                logger.info(
                    "method_created",
                    method_id=method_id,
                    concept_id=concept_id,
                )

                return Method(
                    id=row["id"],
                    concept_id=row["concept_id"],
                    required_assumptions=row["required_assumptions"] or [],
                    problem_types=row["problem_types"] or [],
                    common_estimators=row["common_estimators"] or [],
                )

        except asyncpg.UniqueViolationError:
            logger.error(
                "method_duplicate_concept",
                concept_id=concept_id,
            )
            raise StorageError(f"Method already exists for concept_id {concept_id}")
        except asyncpg.ForeignKeyViolationError:
            logger.error(
                "method_concept_not_found",
                concept_id=concept_id,
            )
            raise StorageError(f"Concept not found: {concept_id}")
        except Exception as e:
            logger.error(
                "method_creation_failed",
                concept_id=concept_id,
                error=str(e),
            )
            raise StorageError(f"Failed to create method: {e}")

    @staticmethod
    async def get_by_id(method_id: UUID) -> Optional[Method]:
        """Retrieve method by ID.

        Args:
            method_id: Method UUID

        Returns:
            Method if found, None otherwise
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM methods WHERE id = $1",
                    method_id,
                )

                if not row:
                    return None

                return Method(
                    id=row["id"],
                    concept_id=row["concept_id"],
                    required_assumptions=row["required_assumptions"] or [],
                    problem_types=row["problem_types"] or [],
                    common_estimators=row["common_estimators"] or [],
                )

        except Exception as e:
            logger.error(
                "method_get_by_id_failed",
                method_id=method_id,
                error=str(e),
            )
            raise StorageError(f"Failed to retrieve method: {e}")

    @staticmethod
    async def get_by_concept_id(concept_id: UUID) -> Optional[Method]:
        """Retrieve method by associated concept ID.

        Args:
            concept_id: Concept UUID

        Returns:
            Method if found, None otherwise
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM methods WHERE concept_id = $1",
                    concept_id,
                )

                if not row:
                    return None

                return Method(
                    id=row["id"],
                    concept_id=row["concept_id"],
                    required_assumptions=row["required_assumptions"] or [],
                    problem_types=row["problem_types"] or [],
                    common_estimators=row["common_estimators"] or [],
                )

        except Exception as e:
            logger.error(
                "method_get_by_concept_failed",
                concept_id=concept_id,
                error=str(e),
            )
            raise StorageError(f"Failed to retrieve method: {e}")

    @staticmethod
    async def update(
        method_id: UUID,
        required_assumptions: Optional[list[str]] = None,
        problem_types: Optional[list[str]] = None,
        common_estimators: Optional[list[str]] = None,
    ) -> Method:
        """Update method attributes.

        Only provided fields are updated. None values are ignored.

        Args:
            method_id: Method UUID
            required_assumptions: New required assumptions list
            problem_types: New problem types list
            common_estimators: New common estimators list

        Returns:
            Updated Method

        Raises:
            StorageError: If method not found or update fails
        """
        pool = await get_connection_pool()

        # Build dynamic UPDATE query
        updates = []
        params = [method_id]
        param_idx = 2

        if required_assumptions is not None:
            updates.append(f"required_assumptions = ${param_idx}")
            params.append(required_assumptions)
            param_idx += 1

        if problem_types is not None:
            updates.append(f"problem_types = ${param_idx}")
            params.append(problem_types)
            param_idx += 1

        if common_estimators is not None:
            updates.append(f"common_estimators = ${param_idx}")
            params.append(common_estimators)
            param_idx += 1

        if not updates:
            # No updates requested, return current record
            return await MethodStore.get_by_id(method_id)

        query = f"""
            UPDATE methods
            SET {", ".join(updates)}
            WHERE id = $1
            RETURNING *
        """

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, *params)

                if not row:
                    raise StorageError(f"Method not found: {method_id}")

                logger.info(
                    "method_updated",
                    method_id=method_id,
                    updated_fields=updates,
                )

                return Method(
                    id=row["id"],
                    concept_id=row["concept_id"],
                    required_assumptions=row["required_assumptions"] or [],
                    problem_types=row["problem_types"] or [],
                    common_estimators=row["common_estimators"] or [],
                )

        except Exception as e:
            logger.error(
                "method_update_failed",
                method_id=method_id,
                error=str(e),
            )
            raise StorageError(f"Failed to update method: {e}")

    @staticmethod
    async def delete(method_id: UUID) -> bool:
        """Delete method record.

        Args:
            method_id: Method UUID

        Returns:
            True if deleted, False if not found
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM methods WHERE id = $1",
                    method_id,
                )

                deleted = result.split()[-1] == "1"

                if deleted:
                    logger.info(
                        "method_deleted",
                        method_id=method_id,
                    )

                return deleted

        except Exception as e:
            logger.error(
                "method_deletion_failed",
                method_id=method_id,
                error=str(e),
            )
            raise StorageError(f"Failed to delete method: {e}")

    @staticmethod
    async def list_all(limit: int = 100, offset: int = 0) -> list[Method]:
        """List all methods with pagination.

        Args:
            limit: Maximum number of methods to return
            offset: Number of methods to skip

        Returns:
            List of Method records
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM methods
                    ORDER BY concept_id
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset,
                )

                return [
                    Method(
                        id=row["id"],
                        concept_id=row["concept_id"],
                        required_assumptions=row["required_assumptions"] or [],
                        problem_types=row["problem_types"] or [],
                        common_estimators=row["common_estimators"] or [],
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(
                "method_list_failed",
                limit=limit,
                offset=offset,
                error=str(e),
            )
            raise StorageError(f"Failed to list methods: {e}")

    @staticmethod
    async def count() -> int:
        """Count total number of methods.

        Returns:
            Total method count
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT COUNT(*) FROM methods")
                return result

        except Exception as e:
            logger.error(
                "method_count_failed",
                error=str(e),
            )
            raise StorageError(f"Failed to count methods: {e}")
