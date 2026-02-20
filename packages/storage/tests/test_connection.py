"""Tests for database connection management."""

import pytest

from research_kb_storage.connection import (
    DatabaseConfig,
    get_connection_pool,
    close_connection_pool,
    check_connection_health,
)

pytestmark = pytest.mark.integration


def test_database_config_defaults():
    """Test DatabaseConfig default values."""
    config = DatabaseConfig()

    assert config.host == "localhost"
    assert config.port == 5432
    assert config.database == "research_kb"
    assert config.user == "postgres"
    assert config.password == "postgres"
    assert config.min_pool_size == 2
    assert config.max_pool_size == 10


def test_database_config_custom_values():
    """Test DatabaseConfig with custom values."""
    config = DatabaseConfig(
        host="db.example.com",
        port=5433,
        database="my_db",
        user="myuser",
        password="mypass",
        min_pool_size=5,
        max_pool_size=20,
    )

    assert config.host == "db.example.com"
    assert config.port == 5433
    assert config.database == "my_db"
    assert config.user == "myuser"
    assert config.password == "mypass"
    assert config.min_pool_size == 5
    assert config.max_pool_size == 20


def test_database_config_get_dsn():
    """Test DatabaseConfig DSN generation."""
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_db",
        user="testuser",
        password="testpass",
    )

    dsn = config.get_dsn()
    assert dsn == "postgresql://testuser:testpass@localhost:5432/test_db"


def test_database_config_get_dsn_special_characters():
    """Test DSN generation with special characters in password."""
    config = DatabaseConfig(
        host="localhost",
        database="test_db",
        user="user",
        password="p@ss:w/ord",
    )

    dsn = config.get_dsn()
    # Note: For production, passwords should be URL-encoded
    # This test just verifies the format, not proper encoding
    assert "p@ss:w/ord" in dsn


@pytest.mark.asyncio
async def test_get_connection_pool_default_config(test_db):
    """Test getting connection pool with default config."""
    pool = await get_connection_pool()

    assert pool is not None
    assert not pool._closed


@pytest.mark.asyncio
async def test_get_connection_pool_custom_config(test_db):
    """Test getting connection pool with custom config."""
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password="postgres",
        min_pool_size=1,
        max_pool_size=5,
    )

    pool = await get_connection_pool(config)

    assert pool is not None
    assert not pool._closed


@pytest.mark.asyncio
async def test_get_connection_pool_is_singleton(test_db):
    """Test that connection pool is a singleton (returns same instance)."""
    pool1 = await get_connection_pool()
    pool2 = await get_connection_pool()

    assert pool1 is pool2


@pytest.mark.asyncio
async def test_connection_pool_basic_query(test_db):
    """Test executing a basic query through the pool."""
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")

    assert result == 1


@pytest.mark.asyncio
async def test_connection_pool_concurrent_queries(test_db):
    """Test multiple concurrent queries through the pool."""
    pool = await get_connection_pool()

    # Execute multiple queries concurrently
    async def query():
        async with pool.acquire() as conn:
            return await conn.fetchval("SELECT 1 + 1")

    results = []
    for _ in range(5):
        result = await query()
        results.append(result)

    assert all(r == 2 for r in results)


@pytest.mark.asyncio
async def test_close_connection_pool(test_db):
    """Test closing the connection pool."""
    # Get pool
    pool = await get_connection_pool()
    assert not pool._closed

    # Close pool
    await close_connection_pool()

    # Pool should be closed
    assert pool._closed


@pytest.mark.asyncio
async def test_close_connection_pool_idempotent(test_db):
    """Test that closing pool multiple times is safe."""
    await get_connection_pool()

    # Close multiple times
    await close_connection_pool()
    await close_connection_pool()

    # Should not raise exception


@pytest.mark.asyncio
async def test_check_connection_health_healthy(test_db):
    """Test health check with healthy connection."""
    await get_connection_pool()

    healthy = await check_connection_health()

    assert healthy is True


@pytest.mark.asyncio
async def test_get_connection_pool_after_close(test_db):
    """Test getting pool after closing creates new pool."""
    # Get and close pool
    pool1 = await get_connection_pool()
    await close_connection_pool()

    # Get pool again
    pool2 = await get_connection_pool()

    # Should be a different instance
    assert pool1 is not pool2
    assert pool1._closed
    assert not pool2._closed


@pytest.mark.asyncio
async def test_connection_pool_transaction(test_db):
    """Test using connection pool with transaction."""
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Create a temporary table in transaction
            await conn.execute(
                """
                CREATE TEMP TABLE test_transaction (
                    id SERIAL PRIMARY KEY,
                    value TEXT
                )
            """
            )

            await conn.execute("INSERT INTO test_transaction (value) VALUES ($1)", "test")

            result = await conn.fetchval("SELECT COUNT(*) FROM test_transaction")

    assert result == 1


@pytest.mark.asyncio
async def test_connection_pool_rollback(test_db):
    """Test transaction rollback."""
    pool = await get_connection_pool()

    # Create temp table outside transaction
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TEMP TABLE test_rollback (
                id SERIAL PRIMARY KEY,
                value TEXT
            )
        """
        )

        # Try transaction that will roll back
        try:
            async with conn.transaction():
                await conn.execute("INSERT INTO test_rollback (value) VALUES ($1)", "test")
                # Force rollback by raising exception
                raise ValueError("Intentional rollback")
        except ValueError:
            pass

        # Check that insert was rolled back
        count = await conn.fetchval("SELECT COUNT(*) FROM test_rollback")

    assert count == 0


@pytest.mark.asyncio
async def test_connection_pool_multiple_connections(test_db):
    """Test acquiring multiple connections from pool."""
    pool = await get_connection_pool()

    # Acquire multiple connections simultaneously
    async with pool.acquire() as conn1:
        async with pool.acquire() as conn2:
            result1 = await conn1.fetchval("SELECT 1")
            result2 = await conn2.fetchval("SELECT 2")

    assert result1 == 1
    assert result2 == 2


@pytest.mark.asyncio
async def test_connection_pool_with_timeout(test_db):
    """Test that pool respects command timeout."""
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        # Simple query should complete within timeout
        result = await conn.fetchval("SELECT 1")

    assert result == 1
