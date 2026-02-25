"""Shared test fixtures for storage package.

Provides database connection pool management for integration tests.

IMPORTANT: Tests use `research_kb_test` database to protect production data.
The TRUNCATE operations will REFUSE to run against `research_kb`.
"""

import os

import pytest_asyncio
from research_kb_storage import (
    DatabaseConfig,
    get_connection_pool,
    close_connection_pool,
)

# Test database name - NEVER use production database for tests
TEST_DATABASE_NAME = os.environ.get("TEST_DATABASE_NAME", "research_kb_test")
PRODUCTION_DATABASE_NAME = "research_kb"


class ProductionDatabaseError(Exception):
    """Raised when test attempts to modify production database."""

    pass


def _verify_not_production(database_name: str) -> None:
    """Safety check: refuse to run destructive operations on production DB."""
    if database_name == PRODUCTION_DATABASE_NAME:
        raise ProductionDatabaseError(
            f"REFUSING to run test fixture against production database '{PRODUCTION_DATABASE_NAME}'!\n"
            f"Tests must use '{TEST_DATABASE_NAME}' or another test database.\n"
            f"Set TEST_DATABASE_NAME environment variable to override."
        )


@pytest_asyncio.fixture(scope="function")
async def db_pool():
    """Create database connection pool for each test function.

    This fixture:
    - Creates a fresh connection pool for each test
    - Cleans the database before the test runs
    - Closes the pool after the test completes
    - REFUSES to connect to production database

    Using function scope avoids event loop issues with session-scoped pools.
    """
    # Safety check BEFORE connecting
    _verify_not_production(TEST_DATABASE_NAME)

    # Reset global pool state
    await close_connection_pool()

    # Create new pool with TEST database
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database=TEST_DATABASE_NAME,
        user="postgres",
        password="postgres",
    )
    pool = await get_connection_pool(config)

    # Double-check we're not on production
    async with pool.acquire() as conn:
        current_db = await conn.fetchval("SELECT current_database()")
        _verify_not_production(current_db)

        # Truncate all tables in correct order (respecting foreign keys)
        await conn.execute(
            "TRUNCATE TABLE chunk_concepts, concept_relationships, chunks, concepts, sources, citations, methods, assumptions CASCADE"
        )

    yield pool

    # Clean up after test
    await close_connection_pool()


@pytest_asyncio.fixture(scope="function")
async def test_db(db_pool):
    """Alias for db_pool to match test expectations."""
    return db_pool


@pytest_asyncio.fixture
async def test_source(test_db):
    """Create a test source for chunk tests."""
    from research_kb_storage import SourceStore
    from research_kb_contracts import SourceType

    source = await SourceStore.create(
        source_type=SourceType.PAPER,
        title="Test Paper",
        file_hash="testhash123",
        domain_id="causal_inference",
        authors=["Test Author"],
        year=2024,
        file_path="/test/path.pdf",
        metadata={"test": True},
    )

    return source
