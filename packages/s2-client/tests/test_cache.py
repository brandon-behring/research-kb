"""Tests for S2 SQLite response cache.

Tests cache operations including hits, misses, expiration, and file operations.
Uses tmp_path fixture for isolated file-based tests.
"""

import asyncio
import json
import time

import pytest

from s2_client.cache import DEFAULT_TTL_SECONDS, S2Cache
from s2_client.errors import S2CacheError


# -----------------------------------------------------------------------------
# Initialization Tests
# -----------------------------------------------------------------------------


class TestS2CacheInit:
    """Tests for S2Cache initialization."""

    def test_default_cache_dir(self):
        """Test default cache directory."""
        cache = S2Cache()
        assert cache.cache_dir.name == "s2_client"
        assert cache.cache_dir.parent.name == ".cache"

    def test_custom_cache_dir(self, tmp_path):
        """Test custom cache directory."""
        custom_dir = tmp_path / "custom_cache"
        cache = S2Cache(cache_dir=custom_dir)
        assert cache.cache_dir == custom_dir

    def test_default_ttl(self):
        """Test default TTL is 7 days."""
        cache = S2Cache()
        assert cache.ttl_seconds == DEFAULT_TTL_SECONDS
        assert cache.ttl_seconds == 7 * 24 * 60 * 60

    def test_custom_ttl(self):
        """Test custom TTL."""
        cache = S2Cache(ttl_seconds=3600)
        assert cache.ttl_seconds == 3600

    def test_connection_not_initialized(self, tmp_path):
        """Connection should be None before initialize()."""
        cache = S2Cache(cache_dir=tmp_path)
        assert cache._conn is None


class TestS2CacheInitialize:
    """Tests for cache initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_directory(self, tmp_path):
        """Initialize should create cache directory."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        cache = S2Cache(cache_dir=cache_dir)
        await cache.initialize()

        assert cache_dir.exists()
        await cache.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_database(self, tmp_path):
        """Initialize should create database file."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        db_path = tmp_path / "s2_cache.db"
        assert db_path.exists()

        await cache.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_table(self, tmp_path):
        """Initialize should create cache table."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        # Verify table exists by executing a query
        async with cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cache'"
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "cache"

        await cache.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_index(self, tmp_path):
        """Initialize should create expiration index."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        # Verify index exists
        async with cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_cache_expires'"
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None

        await cache.close()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, tmp_path):
        """Initialize should be idempotent."""
        cache = S2Cache(cache_dir=tmp_path)

        # Initialize twice
        await cache.initialize()
        await cache.initialize()

        # Should work fine
        await cache.set("test", None, {"data": 1})
        result = await cache.get("test", None)
        assert result == {"data": 1}

        await cache.close()


# -----------------------------------------------------------------------------
# Get/Set Tests
# -----------------------------------------------------------------------------


class TestS2CacheGetSet:
    """Tests for cache get and set operations."""

    @pytest.mark.asyncio
    async def test_set_and_get_roundtrip(self, tmp_path):
        """Data should survive cache roundtrip."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        test_data = {"title": "Test Paper", "year": 2024, "authors": ["Alice", "Bob"]}
        await cache.set("paper/123", {"fields": "title,year"}, test_data)

        result = await cache.get("paper/123", {"fields": "title,year"})
        assert result == test_data

        await cache.close()

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, tmp_path):
        """Get should return None for non-existent key."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        result = await cache.get("nonexistent/key", None)
        assert result is None

        await cache.close()

    @pytest.mark.asyncio
    async def test_get_with_different_params(self, tmp_path):
        """Different params should produce different cache keys."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        await cache.set("paper/123", {"fields": "title"}, {"title": "Title Only"})
        await cache.set("paper/123", {"fields": "all"}, {"title": "Full", "year": 2024})

        result1 = await cache.get("paper/123", {"fields": "title"})
        result2 = await cache.get("paper/123", {"fields": "all"})

        assert result1 == {"title": "Title Only"}
        assert result2 == {"title": "Full", "year": 2024}

        await cache.close()

    @pytest.mark.asyncio
    async def test_set_overwrites_existing(self, tmp_path):
        """Set should overwrite existing cache entry."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        await cache.set("key", None, {"version": 1})
        await cache.set("key", None, {"version": 2})

        result = await cache.get("key", None)
        assert result == {"version": 2}

        await cache.close()

    @pytest.mark.asyncio
    async def test_get_without_initialize_raises(self, tmp_path):
        """Get without initialize should raise S2CacheError."""
        cache = S2Cache(cache_dir=tmp_path)

        with pytest.raises(S2CacheError, match="not initialized"):
            await cache.get("test", None)

    @pytest.mark.asyncio
    async def test_set_without_initialize_raises(self, tmp_path):
        """Set without initialize should raise S2CacheError."""
        cache = S2Cache(cache_dir=tmp_path)

        with pytest.raises(S2CacheError, match="not initialized"):
            await cache.set("test", None, {"data": 1})

    @pytest.mark.asyncio
    async def test_get_with_none_params(self, tmp_path):
        """Get with None params should work."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        await cache.set("test", None, {"data": "value"})
        result = await cache.get("test", None)

        assert result == {"data": "value"}

        await cache.close()

    @pytest.mark.asyncio
    async def test_get_with_empty_params(self, tmp_path):
        """Get with empty params dict should work."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        await cache.set("test", {}, {"data": "value"})
        result = await cache.get("test", {})

        assert result == {"data": "value"}

        await cache.close()

    @pytest.mark.asyncio
    async def test_params_order_does_not_matter(self, tmp_path):
        """Params order should not affect cache key."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        # Set with one order
        await cache.set("test", {"a": 1, "b": 2}, {"data": "value"})

        # Get with different order
        result = await cache.get("test", {"b": 2, "a": 1})

        assert result == {"data": "value"}

        await cache.close()


# -----------------------------------------------------------------------------
# Expiration Tests
# -----------------------------------------------------------------------------


class TestS2CacheExpiration:
    """Tests for cache expiration behavior."""

    @pytest.mark.asyncio
    async def test_expired_entries_not_returned(self, tmp_path):
        """Expired entries should not be returned by get."""
        # Use very short TTL
        cache = S2Cache(cache_dir=tmp_path, ttl_seconds=1)
        await cache.initialize()

        await cache.set("test", None, {"data": "value"})

        # Immediate get should succeed
        result = await cache.get("test", None)
        assert result == {"data": "value"}

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Should return None after expiration
        result = await cache.get("test", None)
        assert result is None

        await cache.close()

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired(self, tmp_path):
        """Cleanup should remove expired entries."""
        cache = S2Cache(cache_dir=tmp_path, ttl_seconds=1)
        await cache.initialize()

        await cache.set("key1", None, {"data": 1})
        await cache.set("key2", None, {"data": 2})

        # Check both exist
        stats = await cache.stats()
        assert stats["valid_entries"] == 2

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Manual cleanup
        removed = await cache._cleanup_expired()
        assert removed == 2

        stats = await cache.stats()
        assert stats["valid_entries"] == 0

        await cache.close()

    @pytest.mark.asyncio
    async def test_initialize_cleans_expired(self, tmp_path):
        """Initialize should clean up expired entries."""
        # Create cache and add entry
        cache1 = S2Cache(cache_dir=tmp_path, ttl_seconds=1)
        await cache1.initialize()
        await cache1.set("old", None, {"data": "old"})
        await cache1.close()

        # Wait for expiration
        await asyncio.sleep(1.5)

        # New cache instance - should cleanup on init
        cache2 = S2Cache(cache_dir=tmp_path, ttl_seconds=3600)
        await cache2.initialize()

        # Old entry should be gone
        result = await cache2.get("old", None)
        assert result is None

        await cache2.close()


# -----------------------------------------------------------------------------
# Invalidate Tests
# -----------------------------------------------------------------------------


class TestS2CacheInvalidate:
    """Tests for cache invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_removes_entry(self, tmp_path):
        """Invalidate should remove specific entry."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        await cache.set("key1", {"p": 1}, {"data": 1})
        await cache.set("key2", {"p": 2}, {"data": 2})

        await cache.invalidate("key1", {"p": 1})

        assert await cache.get("key1", {"p": 1}) is None
        assert await cache.get("key2", {"p": 2}) == {"data": 2}

        await cache.close()

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_no_error(self, tmp_path):
        """Invalidating nonexistent key should not raise."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        # Should not raise
        await cache.invalidate("nonexistent", None)

        await cache.close()

    @pytest.mark.asyncio
    async def test_invalidate_without_connection(self, tmp_path):
        """Invalidate without connection should be no-op."""
        cache = S2Cache(cache_dir=tmp_path)
        # Don't initialize

        # Should not raise
        await cache.invalidate("test", None)


# -----------------------------------------------------------------------------
# Clear Tests
# -----------------------------------------------------------------------------


class TestS2CacheClear:
    """Tests for cache clearing."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_entries(self, tmp_path):
        """Clear should remove all cached entries."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        # Add multiple entries
        for i in range(5):
            await cache.set(f"key{i}", None, {"data": i})

        stats = await cache.stats()
        assert stats["valid_entries"] == 5

        await cache.clear()

        stats = await cache.stats()
        assert stats["valid_entries"] == 0

        await cache.close()

    @pytest.mark.asyncio
    async def test_clear_without_connection(self, tmp_path):
        """Clear without connection should be no-op."""
        cache = S2Cache(cache_dir=tmp_path)
        # Don't initialize

        # Should not raise
        await cache.clear()


# -----------------------------------------------------------------------------
# Stats Tests
# -----------------------------------------------------------------------------


class TestS2CacheStats:
    """Tests for cache statistics."""

    @pytest.mark.asyncio
    async def test_stats_empty_cache(self, tmp_path):
        """Stats should report zeros for empty cache."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        stats = await cache.stats()

        assert stats["valid_entries"] == 0
        assert stats["total_entries"] == 0
        assert stats["expired_entries"] == 0

        await cache.close()

    @pytest.mark.asyncio
    async def test_stats_with_entries(self, tmp_path):
        """Stats should reflect cache state."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        await cache.set("key1", None, {"data": 1})
        await cache.set("key2", None, {"data": 2})
        await cache.set("key3", None, {"data": 3})

        stats = await cache.stats()

        assert stats["valid_entries"] == 3
        assert stats["total_entries"] == 3
        assert stats["expired_entries"] == 0

        await cache.close()

    @pytest.mark.asyncio
    async def test_stats_includes_size(self, tmp_path):
        """Stats should include database size."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        await cache.set("test", None, {"data": "x" * 1000})

        stats = await cache.stats()

        assert "size_bytes" in stats
        assert stats["size_bytes"] > 0
        assert "size_mb" in stats

        await cache.close()

    @pytest.mark.asyncio
    async def test_stats_includes_ttl(self, tmp_path):
        """Stats should include TTL setting."""
        cache = S2Cache(cache_dir=tmp_path, ttl_seconds=3600)
        await cache.initialize()

        stats = await cache.stats()

        assert stats["ttl_seconds"] == 3600

        await cache.close()

    @pytest.mark.asyncio
    async def test_stats_includes_path(self, tmp_path):
        """Stats should include cache path."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        stats = await cache.stats()

        assert "cache_path" in stats
        assert "s2_cache.db" in stats["cache_path"]

        await cache.close()

    @pytest.mark.asyncio
    async def test_stats_without_initialization(self, tmp_path):
        """Stats without initialization should return error."""
        cache = S2Cache(cache_dir=tmp_path)

        stats = await cache.stats()

        assert "error" in stats
        assert "not initialized" in stats["error"]

    @pytest.mark.asyncio
    async def test_stats_tracks_expired_entries(self, tmp_path):
        """Stats should differentiate valid and expired entries."""
        cache = S2Cache(cache_dir=tmp_path, ttl_seconds=1)
        await cache.initialize()

        await cache.set("key1", None, {"data": 1})

        # Add entry then wait for expiration
        await asyncio.sleep(1.5)

        # Add fresh entry
        cache.ttl_seconds = 3600  # Change TTL for new entry
        await cache.set("key2", None, {"data": 2})

        stats = await cache.stats()

        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 1

        await cache.close()


# -----------------------------------------------------------------------------
# Key Generation Tests
# -----------------------------------------------------------------------------


class TestS2CacheKeyGeneration:
    """Tests for cache key generation."""

    @pytest.mark.asyncio
    async def test_key_is_sha256_hash(self, tmp_path):
        """Key should be SHA256 hash."""
        cache = S2Cache(cache_dir=tmp_path)

        key = cache._make_key("test", {"param": "value"})

        # SHA256 produces 64 hex characters
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    @pytest.mark.asyncio
    async def test_same_inputs_same_key(self, tmp_path):
        """Same inputs should produce same key."""
        cache = S2Cache(cache_dir=tmp_path)

        key1 = cache._make_key("endpoint", {"a": 1, "b": 2})
        key2 = cache._make_key("endpoint", {"a": 1, "b": 2})

        assert key1 == key2

    @pytest.mark.asyncio
    async def test_different_endpoints_different_keys(self, tmp_path):
        """Different endpoints should produce different keys."""
        cache = S2Cache(cache_dir=tmp_path)

        key1 = cache._make_key("endpoint1", {"a": 1})
        key2 = cache._make_key("endpoint2", {"a": 1})

        assert key1 != key2

    @pytest.mark.asyncio
    async def test_different_params_different_keys(self, tmp_path):
        """Different params should produce different keys."""
        cache = S2Cache(cache_dir=tmp_path)

        key1 = cache._make_key("endpoint", {"a": 1})
        key2 = cache._make_key("endpoint", {"a": 2})

        assert key1 != key2

    @pytest.mark.asyncio
    async def test_param_order_does_not_affect_key(self, tmp_path):
        """Param order should not affect key (sorted)."""
        cache = S2Cache(cache_dir=tmp_path)

        key1 = cache._make_key("endpoint", {"z": 1, "a": 2})
        key2 = cache._make_key("endpoint", {"a": 2, "z": 1})

        assert key1 == key2


# -----------------------------------------------------------------------------
# Context Manager Tests
# -----------------------------------------------------------------------------


class TestS2CacheContextManager:
    """Tests for async context manager interface."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes(self, tmp_path):
        """Context manager should initialize cache."""
        async with S2Cache(cache_dir=tmp_path) as cache:
            assert cache._conn is not None

    @pytest.mark.asyncio
    async def test_context_manager_closes(self, tmp_path):
        """Context manager should close on exit."""
        cache = S2Cache(cache_dir=tmp_path)

        async with cache:
            pass

        assert cache._conn is None

    @pytest.mark.asyncio
    async def test_context_manager_allows_operations(self, tmp_path):
        """Context manager should allow cache operations."""
        async with S2Cache(cache_dir=tmp_path) as cache:
            await cache.set("test", None, {"value": 42})
            result = await cache.get("test", None)
            assert result == {"value": 42}

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exception(self, tmp_path):
        """Context manager should close even on exception."""
        cache = S2Cache(cache_dir=tmp_path)

        with pytest.raises(ValueError):
            async with cache:
                raise ValueError("Test error")

        assert cache._conn is None


# -----------------------------------------------------------------------------
# Close Tests
# -----------------------------------------------------------------------------


class TestS2CacheClose:
    """Tests for cache closing."""

    @pytest.mark.asyncio
    async def test_close_sets_conn_none(self, tmp_path):
        """Close should set connection to None."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        assert cache._conn is not None

        await cache.close()

        assert cache._conn is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, tmp_path):
        """Close should be idempotent."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        await cache.close()
        await cache.close()  # Should not raise

        assert cache._conn is None

    @pytest.mark.asyncio
    async def test_close_without_initialize(self, tmp_path):
        """Close without initialize should not raise."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.close()  # Should not raise


# -----------------------------------------------------------------------------
# Concurrent Access Tests
# -----------------------------------------------------------------------------


class TestS2CacheConcurrency:
    """Tests for concurrent cache access."""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, tmp_path):
        """Concurrent reads should work correctly."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        await cache.set("test", None, {"data": "value"})

        # Launch multiple concurrent reads
        tasks = [cache.get("test", None) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(r == {"data": "value"} for r in results)

        await cache.close()

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, tmp_path):
        """Concurrent writes should not corrupt data."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        async def write_entry(i: int):
            await cache.set(f"key{i}", None, {"index": i})

        # Launch multiple concurrent writes
        tasks = [write_entry(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # All entries should exist
        for i in range(20):
            result = await cache.get(f"key{i}", None)
            assert result == {"index": i}

        await cache.close()


# -----------------------------------------------------------------------------
# Complex Data Tests
# -----------------------------------------------------------------------------


class TestS2CacheComplexData:
    """Tests for caching complex data structures."""

    @pytest.mark.asyncio
    async def test_nested_dict(self, tmp_path):
        """Cache should handle nested dictionaries."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        data = {
            "paper": {
                "title": "Test",
                "authors": [
                    {"name": "Alice", "affiliation": {"org": "MIT"}},
                    {"name": "Bob", "affiliation": {"org": "Stanford"}},
                ],
            },
            "metadata": {"version": 1},
        }

        await cache.set("complex", None, data)
        result = await cache.get("complex", None)

        assert result == data

        await cache.close()

    @pytest.mark.asyncio
    async def test_large_response(self, tmp_path):
        """Cache should handle large responses."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        # Create large data
        data = {"papers": [{"title": f"Paper {i}", "abstract": "x" * 1000} for i in range(100)]}

        await cache.set("large", None, data)
        result = await cache.get("large", None)

        assert result == data
        assert len(result["papers"]) == 100

        await cache.close()

    @pytest.mark.asyncio
    async def test_unicode_content(self, tmp_path):
        """Cache should handle Unicode content."""
        cache = S2Cache(cache_dir=tmp_path)
        await cache.initialize()

        data = {
            "title": "Econometrica",
            "chinese": "机器学习",
            "emoji": "📊",
            "math": "∫ f(x)dx",
        }

        await cache.set("unicode", None, data)
        result = await cache.get("unicode", None)

        assert result == data

        await cache.close()
