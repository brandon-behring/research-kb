"""SQLite response cache for S2 API.

Caches API responses to:
- Reduce API calls (rate limit friendly)
- Enable offline operation for recently accessed papers
- Speed up repeated lookups

Default TTL: 7 days (configurable)
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import aiosqlite

from s2_client.errors import S2CacheError

# Default cache location
DEFAULT_CACHE_DIR = Path(os.environ.get("S2_CACHE_DIR", Path.home() / ".cache" / "s2_client"))
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


class S2Cache:
    """Async SQLite cache for S2 API responses.

    Stores JSON responses keyed by (endpoint, params) hash.
    Automatic cleanup of expired entries on startup.

    Example:
        >>> cache = S2Cache()
        >>> await cache.initialize()
        >>> cached = await cache.get("paper/12345", {"fields": "title,year"})
        >>> if cached is None:
        ...     response = await fetch_from_api()
        ...     await cache.set("paper/12345", {"fields": "title,year"}, response)

    Attributes:
        cache_dir: Directory for cache database (default: ~/.cache/s2_client/)
        ttl_seconds: Time-to-live for cache entries (default: 7 days)
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        """Initialize cache.

        Args:
            cache_dir: Directory for cache database
            ttl_seconds: Cache entry TTL in seconds
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.ttl_seconds = ttl_seconds
        self._db_path = self.cache_dir / "s2_cache.db"
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize cache database.

        Creates directory and tables if they don't exist.
        Runs cleanup of expired entries.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(self._db_path)

        # Create table if not exists
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
        """
        )

        # Create index on expiration for cleanup
        await self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)
        """
        )

        await self._conn.commit()

        # Cleanup expired entries
        await self._cleanup_expired()

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict | None:
        """Get cached response.

        Args:
            endpoint: API endpoint (e.g., "paper/12345")
            params: Query parameters used in the request

        Returns:
            Cached JSON response or None if not found/expired
        """
        if not self._conn:
            raise S2CacheError("Cache not initialized. Call initialize() first.")

        key = self._make_key(endpoint, params)
        now = time.time()

        async with self._conn.execute(
            "SELECT response FROM cache WHERE key = ? AND expires_at > ?",
            (key, now),
        ) as cursor:
            row = await cursor.fetchone()

        if row:
            return json.loads(row[0])
        return None

    async def set(
        self,
        endpoint: str,
        params: dict[str, Any] | None,
        response: dict,
    ) -> None:
        """Cache a response.

        Args:
            endpoint: API endpoint
            params: Query parameters used in the request
            response: JSON response to cache
        """
        if not self._conn:
            raise S2CacheError("Cache not initialized. Call initialize() first.")

        key = self._make_key(endpoint, params)
        now = time.time()
        expires_at = now + self.ttl_seconds

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO cache (key, endpoint, response, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, endpoint, json.dumps(response), now, expires_at),
        )
        await self._conn.commit()

    async def invalidate(self, endpoint: str, params: dict[str, Any] | None = None) -> None:
        """Invalidate a specific cache entry.

        Args:
            endpoint: API endpoint
            params: Query parameters
        """
        if not self._conn:
            return

        key = self._make_key(endpoint, params)
        await self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        await self._conn.commit()

    async def clear(self) -> None:
        """Clear all cached entries."""
        if not self._conn:
            return

        await self._conn.execute("DELETE FROM cache")
        await self._conn.commit()

    async def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with entry count, size, oldest/newest entries
        """
        if not self._conn:
            return {"error": "Cache not initialized"}

        now = time.time()

        async with self._conn.execute(
            "SELECT COUNT(*) FROM cache WHERE expires_at > ?", (now,)
        ) as cursor:
            row = await cursor.fetchone()
            valid_count = row[0] if row else 0

        async with self._conn.execute("SELECT COUNT(*) FROM cache") as cursor:
            row = await cursor.fetchone()
            total_count = row[0] if row else 0

        # Get size
        size_bytes = self._db_path.stat().st_size if self._db_path.exists() else 0

        return {
            "valid_entries": valid_count,
            "total_entries": total_count,
            "expired_entries": total_count - valid_count,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "ttl_seconds": self.ttl_seconds,
            "cache_path": str(self._db_path),
        }

    def _make_key(self, endpoint: str, params: dict[str, Any] | None) -> str:
        """Generate cache key from endpoint and params.

        Uses SHA256 hash of canonicalized params for consistent keys.
        """
        # Sort params for consistent ordering
        params_str = json.dumps(params or {}, sort_keys=True)
        key_input = f"{endpoint}:{params_str}"
        return hashlib.sha256(key_input.encode()).hexdigest()

    async def _cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed
        """
        if not self._conn:
            return 0

        now = time.time()
        cursor = await self._conn.execute(
            "DELETE FROM cache WHERE expires_at <= ?",
            (now,),
        )
        await self._conn.commit()
        return cursor.rowcount

    async def __aenter__(self) -> "S2Cache":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.close()
