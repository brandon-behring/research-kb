"""Database connection management with asyncpg pooling.

Provides:
- Connection pool configuration
- Pool lifecycle management
- Health checks
"""

import os
from dataclasses import dataclass, field
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

    Attributes:
        host: Database host (default: localhost)
        port: Database port (default: 5432)
        database: Database name (default: research_kb)
        user: Database user (default: postgres)
        password: Database password (default: postgres)
        min_pool_size: Minimum connection pool size (default: 2)
        max_pool_size: Maximum connection pool size (default: 10)
        command_timeout: Query timeout in seconds (default: 120.0)
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "research_kb"
    user: str = "postgres"
    password: str = "postgres"
    min_pool_size: int = 2
    max_pool_size: int = 10
    command_timeout: float = 120.0  # Increased for large batch inserts

    def get_dsn(self) -> str:
        """Get PostgreSQL DSN (Data Source Name).

        Returns:
            Connection string in format: postgresql://user:password@host:port/database
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# Global connection pool (initialized once)
_connection_pool: Optional[asyncpg.Pool] = None


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
            return result == 1
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        return False
