"""Database connection management with asyncpg pooling.

Provides:
- Connection pool configuration
- Pool lifecycle management
- Health checks
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

import asyncpg
from research_kb_common import StorageError, get_logger

logger = get_logger(__name__)


def _get_password_from_env() -> str:
    """Get database password from environment.

    Reads from POSTGRES_PASSWORD environment variable.
    Falls back to 'postgres' for development compatibility.
    """
    return os.environ.get("POSTGRES_PASSWORD", "postgres")


@dataclass
class DatabaseConfig:
    """PostgreSQL connection configuration.

    All connection fields read from environment variables with sensible defaults.
    This makes ``DatabaseConfig()`` work both in local development (defaults)
    and in CI (env vars from GitHub Actions service containers).

    Environment variables:
        POSTGRES_HOST     -> host     (default: "localhost")
        POSTGRES_PORT     -> port     (default: 5432)
        POSTGRES_DB       -> database (default: "research_kb")
        POSTGRES_USER     -> user     (default: "postgres")
        POSTGRES_PASSWORD -> password (default: "postgres")

    Attributes:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        min_pool_size: Minimum connection pool size (default: 2)
        max_pool_size: Maximum connection pool size (default: 10)
        command_timeout: Query timeout in seconds (default: 120.0)
    """

    host: str = None  # type: ignore[assignment]
    port: int = None  # type: ignore[assignment]
    database: str = None  # type: ignore[assignment]
    user: str = None  # type: ignore[assignment]
    password: str = None  # type: ignore[assignment]
    min_pool_size: int = 2
    max_pool_size: int = 10
    command_timeout: float = 120.0  # Increased for large batch inserts

    def __post_init__(self) -> None:
        """Populate connection fields from environment variables when not set."""
        if self.host is None:
            self.host = os.environ.get("POSTGRES_HOST", "localhost")
        if self.port is None:
            self.port = int(os.environ.get("POSTGRES_PORT", "5432"))
        if self.database is None:
            self.database = os.environ.get("POSTGRES_DB", "research_kb")
        if self.user is None:
            self.user = os.environ.get("POSTGRES_USER", "postgres")
        if self.password is None:
            self.password = _get_password_from_env()

    def get_dsn(self) -> str:
        """Get PostgreSQL DSN (Data Source Name).

        Returns:
            Connection string in format: postgresql://user:password@host:port/database
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# Global connection pool (initialized once)
_connection_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()


async def get_connection_pool(config: Optional[DatabaseConfig] = None) -> asyncpg.Pool:
    """Get or create the global connection pool.

    Args:
        config: Database configuration (default: DatabaseConfig())

    Returns:
        asyncpg connection pool

    Raises:
        StorageError: If connection pool creation fails

    Example:
        >>> config = DatabaseConfig(host="localhost", database="research_kb")
        >>> pool = await get_connection_pool(config)
        >>> async with pool.acquire() as conn:
        ...     result = await conn.fetchval("SELECT 1")
    """
    global _connection_pool

    # Fast path: pool already exists (no lock needed)
    if _connection_pool is not None:
        return _connection_pool

    # Slow path: acquire lock and double-check
    async with _pool_lock:
        # Double-check after acquiring lock (another coroutine may have created it)
        if _connection_pool is not None:
            return _connection_pool

        if config is None:
            config = DatabaseConfig()

        try:
            logger.info(
                "creating_connection_pool",
                host=config.host,
                port=config.port,
                database=config.database,
                min_size=config.min_pool_size,
                max_size=config.max_pool_size,
            )

            _connection_pool = await asyncpg.create_pool(
                dsn=config.get_dsn(),
                min_size=config.min_pool_size,
                max_size=config.max_pool_size,
                command_timeout=config.command_timeout,
            )

            logger.info("connection_pool_created", pool_size=config.max_pool_size)
            return _connection_pool

        except Exception as e:
            logger.error("connection_pool_creation_failed", error=str(e))
            raise StorageError(f"Failed to create connection pool: {e}") from e


async def close_connection_pool() -> None:
    """Close the global connection pool.

    Should be called during application shutdown.

    Example:
        >>> await close_connection_pool()
    """
    global _connection_pool

    if _connection_pool is not None:
        logger.info("closing_connection_pool")
        try:
            await _connection_pool.close()
        except Exception as e:
            logger.warning("connection_pool_close_warning", error=str(e))
        finally:
            _connection_pool = None
            logger.info("connection_pool_closed")


async def check_connection_health() -> bool:
    """Check database connection health.

    Returns:
        True if connection is healthy, False otherwise

    Example:
        >>> healthy = await check_connection_health()
        >>> if not healthy:
        ...     logger.error("database_unhealthy")
    """
    try:
        pool = await get_connection_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            return bool(result == 1)
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        return False
