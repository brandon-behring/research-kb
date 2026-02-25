"""AssumptionStore - CRUD operations for assumptions table.

Provides:
- Create assumption records with specialized attributes
- Retrieve assumptions by ID or concept_id
- Update assumption attributes
- Delete assumptions
- List all assumptions with pagination
- Count total assumptions

Assumptions table stores specialized attributes for assumption-type concepts (1:1 relationship).
"""

import json
from typing import Any, Optional
from uuid import UUID, uuid4

import asyncpg
from research_kb_common import StorageError, get_logger
from research_kb_contracts import Assumption

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class AssumptionStore:
    """Storage operations for Assumption entities.

    All operations use the global connection pool.
    Assumptions are 1:1 with Concept records (concept_id is UNIQUE).
    """

    @staticmethod
    async def create(
        concept_id: UUID,
        mathematical_statement: Optional[str] = None,
        is_testable: Optional[bool] = None,
        common_tests: Optional[list[str]] = None,
        violation_consequences: Optional[str] = None,
    ) -> Assumption:
        """Create a new assumption record for a concept.

        Args:
            concept_id: UUID of the associated concept (must exist)
            mathematical_statement: Formal mathematical statement
            is_testable: Whether assumption can be empirically tested
            common_tests: Common tests for this assumption (Hausman, Durbin-Wu-Hausman, etc.)
            violation_consequences: Consequences of violating this assumption

        Returns:
            Created Assumption

        Raises:
            StorageError: If creation fails (e.g., duplicate concept_id, concept doesn't exist)
        """
        pool = await get_connection_pool()
        assumption_id = uuid4()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    """
                    INSERT INTO assumptions (
                        id, concept_id, mathematical_statement,
                        is_testable, common_tests, violation_consequences
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING *
                    """,
                    assumption_id,
                    concept_id,
                    mathematical_statement,
                    is_testable,
                    common_tests or [],
                    violation_consequences,
                )

                logger.info(
                    "assumption_created",
                    assumption_id=assumption_id,
                    concept_id=concept_id,
                )

                return Assumption(
                    id=row["id"],
                    concept_id=row["concept_id"],
                    mathematical_statement=row["mathematical_statement"],
                    is_testable=row["is_testable"],
                    common_tests=row["common_tests"] or [],
                    violation_consequences=row["violation_consequences"],
                )

        except asyncpg.UniqueViolationError:
            logger.error(
                "assumption_duplicate_concept",
                concept_id=concept_id,
            )
            raise StorageError(f"Assumption already exists for concept_id {concept_id}")
        except asyncpg.ForeignKeyViolationError:
            logger.error(
                "assumption_concept_not_found",
                concept_id=concept_id,
            )
            raise StorageError(f"Concept not found: {concept_id}")
        except Exception as e:
            logger.error(
                "assumption_creation_failed",
                concept_id=concept_id,
                error=str(e),
            )
            raise StorageError(f"Failed to create assumption: {e}")

    @staticmethod
    async def get_by_id(assumption_id: UUID) -> Optional[Assumption]:
        """Retrieve assumption by ID.

        Args:
            assumption_id: Assumption UUID

        Returns:
            Assumption if found, None otherwise
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM assumptions WHERE id = $1",
                    assumption_id,
                )

                if not row:
                    return None

                return Assumption(
                    id=row["id"],
                    concept_id=row["concept_id"],
                    mathematical_statement=row["mathematical_statement"],
                    is_testable=row["is_testable"],
                    common_tests=row["common_tests"] or [],
                    violation_consequences=row["violation_consequences"],
                )

        except Exception as e:
            logger.error(
                "assumption_get_by_id_failed",
                assumption_id=assumption_id,
                error=str(e),
            )
            raise StorageError(f"Failed to retrieve assumption: {e}")

    @staticmethod
    async def get_by_concept_id(concept_id: UUID) -> Optional[Assumption]:
        """Retrieve assumption by associated concept ID.

        Args:
            concept_id: Concept UUID

        Returns:
            Assumption if found, None otherwise
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM assumptions WHERE concept_id = $1",
                    concept_id,
                )

                if not row:
                    return None

                return Assumption(
                    id=row["id"],
                    concept_id=row["concept_id"],
                    mathematical_statement=row["mathematical_statement"],
                    is_testable=row["is_testable"],
                    common_tests=row["common_tests"] or [],
                    violation_consequences=row["violation_consequences"],
                )

        except Exception as e:
            logger.error(
                "assumption_get_by_concept_failed",
                concept_id=concept_id,
                error=str(e),
            )
            raise StorageError(f"Failed to retrieve assumption: {e}")

    @staticmethod
    async def update(
        assumption_id: UUID,
        mathematical_statement: Optional[str] = None,
        is_testable: Optional[bool] = None,
        common_tests: Optional[list[str]] = None,
        violation_consequences: Optional[str] = None,
    ) -> Assumption:
        """Update assumption attributes.

        Only provided fields are updated. None values are ignored.

        Args:
            assumption_id: Assumption UUID
            mathematical_statement: New mathematical statement
            is_testable: New is_testable flag
            common_tests: New common tests list
            violation_consequences: New violation consequences text

        Returns:
            Updated Assumption

        Raises:
            StorageError: If assumption not found or update fails
        """
        pool = await get_connection_pool()

        # Build dynamic UPDATE query
        updates = []
        params: list[Any] = [assumption_id]
        param_idx = 2

        # Note: We need to distinguish between None (don't update) and explicit value update
        # For optional fields, we check if they were provided as arguments
        if mathematical_statement is not None:
            updates.append(f"mathematical_statement = ${param_idx}")
            params.append(mathematical_statement)
            param_idx += 1

        if is_testable is not None:
            updates.append(f"is_testable = ${param_idx}")
            params.append(is_testable)
            param_idx += 1

        if common_tests is not None:
            updates.append(f"common_tests = ${param_idx}")
            params.append(common_tests)
            param_idx += 1

        if violation_consequences is not None:
            updates.append(f"violation_consequences = ${param_idx}")
            params.append(violation_consequences)
            param_idx += 1

        if not updates:
            # No updates requested, return current record
            result = await AssumptionStore.get_by_id(assumption_id)
            if not result:
                raise StorageError(f"Assumption not found: {assumption_id}")
            return result

        query = f"""
            UPDATE assumptions
            SET {", ".join(updates)}
            WHERE id = $1
            RETURNING *
        """

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, *params)

                if not row:
                    raise StorageError(f"Assumption not found: {assumption_id}")

                logger.info(
                    "assumption_updated",
                    assumption_id=assumption_id,
                    updated_fields=updates,
                )

                return Assumption(
                    id=row["id"],
                    concept_id=row["concept_id"],
                    mathematical_statement=row["mathematical_statement"],
                    is_testable=row["is_testable"],
                    common_tests=row["common_tests"] or [],
                    violation_consequences=row["violation_consequences"],
                )

        except Exception as e:
            logger.error(
                "assumption_update_failed",
                assumption_id=assumption_id,
                error=str(e),
            )
            raise StorageError(f"Failed to update assumption: {e}")

    @staticmethod
    async def delete(assumption_id: UUID) -> bool:
        """Delete assumption record.

        Args:
            assumption_id: Assumption UUID

        Returns:
            True if deleted, False if not found
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM assumptions WHERE id = $1",
                    assumption_id,
                )

                deleted: bool = result.split()[-1] == "1"

                if deleted:
                    logger.info(
                        "assumption_deleted",
                        assumption_id=assumption_id,
                    )

                return deleted

        except Exception as e:
            logger.error(
                "assumption_deletion_failed",
                assumption_id=assumption_id,
                error=str(e),
            )
            raise StorageError(f"Failed to delete assumption: {e}")

    @staticmethod
    async def list_all(limit: int = 100, offset: int = 0) -> list[Assumption]:
        """List all assumptions with pagination.

        Args:
            limit: Maximum number of assumptions to return
            offset: Number of assumptions to skip

        Returns:
            List of Assumption records
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM assumptions
                    ORDER BY concept_id
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset,
                )

                return [
                    Assumption(
                        id=row["id"],
                        concept_id=row["concept_id"],
                        mathematical_statement=row["mathematical_statement"],
                        is_testable=row["is_testable"],
                        common_tests=row["common_tests"] or [],
                        violation_consequences=row["violation_consequences"],
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(
                "assumption_list_failed",
                limit=limit,
                offset=offset,
                error=str(e),
            )
            raise StorageError(f"Failed to list assumptions: {e}")

    @staticmethod
    async def count() -> int:
        """Count total number of assumptions.

        Returns:
            Total assumption count
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT COUNT(*) FROM assumptions")
                return int(result)

        except Exception as e:
            logger.error(
                "assumption_count_failed",
                error=str(e),
            )
            raise StorageError(f"Failed to count assumptions: {e}")
