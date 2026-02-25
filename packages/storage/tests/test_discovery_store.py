"""Tests for DiscoveryStore — audit logging for S2 discovery operations.

Tests cover:
- log_discovery: insert with all fields, minimal fields, metadata handling
- get_recent: ordering, domain filtering, method filtering, limit
- get_stats: aggregation, domain filtering, day windowing, method breakdown
- Error propagation as StorageError
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from research_kb_common import StorageError
from research_kb_storage.discovery_store import (
    DiscoveryMethod,
    DiscoveryStore,
)

pytestmark = pytest.mark.unit


def _make_mock_pool(conn_mock):
    """Create a mock connection pool wrapping the given connection mock."""
    pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn_mock)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool


class TestLogDiscovery:
    """Tests for DiscoveryStore.log_discovery()."""

    @patch("research_kb_storage.discovery_store.get_connection_pool", new_callable=AsyncMock)
    async def test_log_discovery_returns_uuid(self, mock_get_pool):
        """log_discovery returns a UUID for the created row."""
        expected_id = str(uuid4())
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=expected_id)
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await DiscoveryStore.log_discovery(
            discovery_method=DiscoveryMethod.KEYWORD_SEARCH,
            domain_id="causal_inference",
            query="double machine learning",
            papers_found=50,
            papers_ingested=12,
        )

        assert isinstance(result, UUID)
        assert str(result) == expected_id

    @patch("research_kb_storage.discovery_store.get_connection_pool", new_callable=AsyncMock)
    async def test_log_discovery_passes_all_params(self, mock_get_pool):
        """log_discovery forwards all parameters to the SQL INSERT."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=str(uuid4()))
        mock_get_pool.return_value = _make_mock_pool(conn)

        await DiscoveryStore.log_discovery(
            discovery_method=DiscoveryMethod.AUTHOR_SEARCH,
            domain_id="time_series",
            query="Hyndman",
            papers_found=100,
            papers_acquired=80,
            papers_ingested=60,
            papers_skipped=30,
            papers_failed=10,
            duration_seconds=45.2,
            metadata={"year_from": 2020},
        )

        conn.fetchval.assert_called_once()
        args = conn.fetchval.call_args[0]
        # SQL is first arg, then positional params
        assert args[1] == DiscoveryMethod.AUTHOR_SEARCH
        assert args[2] == "Hyndman"
        assert args[3] == "time_series"
        assert args[4] == 100  # papers_found
        assert args[5] == 80  # papers_acquired
        assert args[6] == 60  # papers_ingested
        assert args[7] == 30  # papers_skipped
        assert args[8] == 10  # papers_failed
        assert args[9] == pytest.approx(45.2)
        assert args[10] == {"year_from": 2020}
        assert isinstance(args[11], datetime)

    @patch("research_kb_storage.discovery_store.get_connection_pool", new_callable=AsyncMock)
    async def test_log_discovery_defaults_metadata_to_empty_dict(self, mock_get_pool):
        """When metadata is None, it defaults to {}."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=str(uuid4()))
        mock_get_pool.return_value = _make_mock_pool(conn)

        await DiscoveryStore.log_discovery(
            discovery_method=DiscoveryMethod.TOPIC_BATCH,
            domain_id="statistics",
        )

        args = conn.fetchval.call_args[0]
        assert args[10] == {}  # metadata defaults to {}

    @patch("research_kb_storage.discovery_store.get_connection_pool", new_callable=AsyncMock)
    async def test_log_discovery_wraps_errors_as_storage_error(self, mock_get_pool):
        """Database errors are wrapped in StorageError."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(side_effect=RuntimeError("connection lost"))
        mock_get_pool.return_value = _make_mock_pool(conn)

        with pytest.raises(StorageError, match="Failed to log discovery"):
            await DiscoveryStore.log_discovery(
                discovery_method=DiscoveryMethod.KEYWORD_SEARCH,
                domain_id="causal_inference",
            )


class TestGetRecent:
    """Tests for DiscoveryStore.get_recent()."""

    @patch("research_kb_storage.discovery_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_recent_returns_list_of_dicts(self, mock_get_pool):
        """get_recent returns list of dict rows."""
        mock_row = MagicMock()
        mock_row.__iter__ = MagicMock(return_value=iter([("key", "value")]))
        mock_row.keys = MagicMock(return_value=["discovery_method", "papers_ingested"])
        mock_row.__getitem__ = lambda self, k: {
            "discovery_method": "keyword_search",
            "papers_ingested": 10,
        }[k]

        conn = AsyncMock()
        conn.fetch = AsyncMock(
            return_value=[{"discovery_method": "keyword_search", "papers_ingested": 10}]
        )
        mock_get_pool.return_value = _make_mock_pool(conn)

        results = await DiscoveryStore.get_recent(limit=5)

        assert len(results) == 1
        assert results[0]["discovery_method"] == "keyword_search"

    @patch("research_kb_storage.discovery_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_recent_builds_domain_filter(self, mock_get_pool):
        """Passing domain_id adds WHERE clause."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_get_pool.return_value = _make_mock_pool(conn)

        await DiscoveryStore.get_recent(limit=10, domain_id="causal_inference")

        sql = conn.fetch.call_args[0][0]
        assert "domain_id = $1" in sql
        params = conn.fetch.call_args[0][1:]
        assert "causal_inference" in params

    @patch("research_kb_storage.discovery_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_recent_builds_method_filter(self, mock_get_pool):
        """Passing discovery_method adds WHERE clause."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_get_pool.return_value = _make_mock_pool(conn)

        await DiscoveryStore.get_recent(
            limit=10,
            discovery_method=DiscoveryMethod.CITATION_TRAVERSE,
        )

        sql = conn.fetch.call_args[0][0]
        assert "discovery_method" in sql

    @patch("research_kb_storage.discovery_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_recent_wraps_errors(self, mock_get_pool):
        """Database errors wrapped in StorageError."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=RuntimeError("timeout"))
        mock_get_pool.return_value = _make_mock_pool(conn)

        with pytest.raises(StorageError, match="Failed to get recent"):
            await DiscoveryStore.get_recent()


class TestGetStats:
    """Tests for DiscoveryStore.get_stats()."""

    @patch("research_kb_storage.discovery_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_stats_returns_aggregate_dict(self, mock_get_pool):
        """get_stats returns correct structure."""
        agg_row = {
            "total_discoveries": 15,
            "total_papers_found": 500,
            "total_papers_acquired": 300,
            "total_papers_ingested": 200,
            "total_papers_skipped": 100,
            "total_papers_failed": 10,
            "avg_duration_seconds": 32.5,
        }
        method_rows = [
            {"discovery_method": "keyword_search", "count": 10, "papers_ingested": 150},
            {"discovery_method": "author_search", "count": 5, "papers_ingested": 50},
        ]

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=agg_row)
        conn.fetch = AsyncMock(return_value=method_rows)
        mock_get_pool.return_value = _make_mock_pool(conn)

        stats = await DiscoveryStore.get_stats(days=7)

        assert stats["total_discoveries"] == 15
        assert stats["total_papers_ingested"] == 200
        assert stats["avg_duration_seconds"] == pytest.approx(32.5)
        assert stats["days"] == 7
        assert stats["by_method"]["keyword_search"]["count"] == 10
        assert stats["by_method"]["author_search"]["papers_ingested"] == 50

    @patch("research_kb_storage.discovery_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_stats_with_domain_filter(self, mock_get_pool):
        """Passing domain_id adds filter to both queries."""
        agg_row = {
            "total_discoveries": 0,
            "total_papers_found": 0,
            "total_papers_acquired": 0,
            "total_papers_ingested": 0,
            "total_papers_skipped": 0,
            "total_papers_failed": 0,
            "avg_duration_seconds": 0,
        }
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=agg_row)
        conn.fetch = AsyncMock(return_value=[])
        mock_get_pool.return_value = _make_mock_pool(conn)

        await DiscoveryStore.get_stats(domain_id="causal_inference", days=30)

        # Both queries should include domain filter
        agg_sql = conn.fetchrow.call_args[0][0]
        assert "domain_id = $2" in agg_sql
        method_sql = conn.fetch.call_args[0][0]
        assert "domain_id = $2" in method_sql

    @patch("research_kb_storage.discovery_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_stats_wraps_errors(self, mock_get_pool):
        """Database errors wrapped in StorageError."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=RuntimeError("db down"))
        mock_get_pool.return_value = _make_mock_pool(conn)

        with pytest.raises(StorageError, match="Failed to get discovery stats"):
            await DiscoveryStore.get_stats()


class TestDiscoveryMethodConstants:
    """Tests for DiscoveryMethod constants."""

    def test_method_constants_are_strings(self):
        """All discovery methods are non-empty strings."""
        methods = [
            DiscoveryMethod.KEYWORD_SEARCH,
            DiscoveryMethod.AUTHOR_SEARCH,
            DiscoveryMethod.CITATION_TRAVERSE,
            DiscoveryMethod.TOPIC_BATCH,
        ]
        for method in methods:
            assert isinstance(method, str)
            assert len(method) > 0

    def test_method_constants_are_unique(self):
        """All discovery methods are distinct."""
        methods = [
            DiscoveryMethod.KEYWORD_SEARCH,
            DiscoveryMethod.AUTHOR_SEARCH,
            DiscoveryMethod.CITATION_TRAVERSE,
            DiscoveryMethod.TOPIC_BATCH,
        ]
        assert len(set(methods)) == len(methods)
