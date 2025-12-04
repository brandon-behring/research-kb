"""Shared test fixtures for storage package.

Provides database connection pool management for integration tests.
"""

import pytest_asyncio
from research_kb_storage import (
    DatabaseConfig,
    get_connection_pool,
    close_connection_pool,
)


@pytest_asyncio.fixture(scope="function")
async def db_pool():
    """Create database connection pool for each test function.

    This fixture:
    - Creates a fresh connection pool for each test
    - Cleans the database before the test runs
    - Closes the pool after the test completes

    Using function scope avoids event loop issues with session-scoped pools.
    """
    # Reset global pool state
    await close_connection_pool()

    # Create new pool
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password="postgres",
    )
    pool = await get_connection_pool(config)

    # Clean database before test
    async with pool.acquire() as conn:
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
        authors=["Test Author"],
        year=2024,
        file_path="/test/path.pdf",
        file_hash="testhash123",
        metadata={"test": True},
    )

    return source
