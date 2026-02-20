"""Tests for daemon request handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_kb_daemon.handler import (
    METHODS,
    _result_to_dict,
    dispatch,
    handle_health,
    handle_search,
    handle_stats,
)

pytestmark = pytest.mark.unit


class TestResultToDict:
    """Tests for _result_to_dict helper."""

    def test_converts_search_result(self):
        """Test converting SearchResult to dict."""
        # Create mock result
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-123"
        mock_chunk.content = "Test content"
        mock_chunk.metadata = {"page_number": 5, "section_header": "Introduction"}

        mock_source = MagicMock()
        mock_source.id = "source-456"
        mock_source.title = "Test Paper"
        mock_source.authors = ["Author A", "Author B"]
        mock_source.year = 2024

        mock_result = MagicMock()
        mock_result.chunk = mock_chunk
        mock_result.source = mock_source
        mock_result.fts_score = 0.8
        mock_result.vector_score = 0.9
        mock_result.graph_score = 0.1
        mock_result.citation_score = 0.05
        mock_result.combined_score = 0.85

        result = _result_to_dict(mock_result)

        assert result["chunk_id"] == "chunk-123"
        assert result["source_id"] == "source-456"
        assert result["content"] == "Test content"
        assert result["source_title"] == "Test Paper"
        assert result["source_authors"] == ["Author A", "Author B"]
        assert result["source_year"] == 2024
        assert result["page_number"] == 5
        assert result["section_header"] == "Introduction"
        assert result["fts_score"] == 0.8
        assert result["combined_score"] == 0.85


class TestDispatch:
    """Tests for method dispatch."""

    def test_methods_registered(self):
        """Test all expected methods are registered."""
        assert "search" in METHODS
        assert "health" in METHODS
        assert "stats" in METHODS

    @pytest.mark.asyncio
    async def test_dispatch_calls_handler(self):
        """Test dispatch routes to correct handler."""
        mock_health = AsyncMock(return_value={"status": "healthy"})
        with patch("research_kb_daemon.handler.METHODS", {"health": mock_health}):
            result = await dispatch("health", {})
            mock_health.assert_called_once_with({})

    @pytest.mark.asyncio
    async def test_dispatch_unknown_method(self):
        """Test dispatch raises for unknown method."""
        with pytest.raises(ValueError, match="Method not found"):
            await dispatch("unknown_method", {})


class TestHandleSearch:
    """Tests for search handler."""

    @pytest.mark.asyncio
    async def test_search_requires_query(self):
        """Test search requires query parameter."""
        with pytest.raises(ValueError, match="Missing required parameter: query"):
            await handle_search({})

    @pytest.mark.asyncio
    async def test_search_with_defaults(self, mock_embed_response):
        """Test search with default parameters."""
        mock_embed_client = AsyncMock()
        mock_embed_client.embed_query = AsyncMock(return_value=[0.1] * 1024)

        mock_results = []  # Empty results for simplicity

        # Default search uses search_hybrid (not v2) when use_graph=False and use_citations=False
        with (
            patch(
                "research_kb_daemon.handler.get_embed_client",
                return_value=mock_embed_client,
            ),
            patch(
                "research_kb_daemon.handler.search_hybrid", new_callable=AsyncMock
            ) as mock_search,
        ):
            mock_search.return_value = mock_results

            result = await handle_search({"query": "test query"})

            # Verify embed_query called
            mock_embed_client.embed_query.assert_called_once_with("test query")

            # Verify search called with correct params
            mock_search.assert_called_once()
            search_query = mock_search.call_args[0][0]
            assert search_query.text == "test query"
            assert search_query.limit == 10
            # Default balanced weights (normalized)
            assert search_query.fts_weight == pytest.approx(0.3, rel=0.01)
            assert search_query.vector_weight == pytest.approx(0.7, rel=0.01)

    @pytest.mark.asyncio
    async def test_search_context_types(self, mock_embed_response):
        """Test search respects context type presets."""
        mock_embed_client = AsyncMock()
        mock_embed_client.embed_query = AsyncMock(return_value=[0.1] * 1024)

        # Default search uses search_hybrid (not v2) when use_graph=False and use_citations=False
        with (
            patch(
                "research_kb_daemon.handler.get_embed_client",
                return_value=mock_embed_client,
            ),
            patch(
                "research_kb_daemon.handler.search_hybrid", new_callable=AsyncMock
            ) as mock_search,
        ):
            mock_search.return_value = []

            # Test building context
            await handle_search({"query": "test", "context_type": "building"})
            search_query = mock_search.call_args[0][0]
            assert search_query.fts_weight == pytest.approx(0.2, rel=0.01)
            assert search_query.vector_weight == pytest.approx(0.8, rel=0.01)

            # Test auditing context
            await handle_search({"query": "test", "context_type": "auditing"})
            search_query = mock_search.call_args[0][0]
            assert search_query.fts_weight == pytest.approx(0.5, rel=0.01)
            assert search_query.vector_weight == pytest.approx(0.5, rel=0.01)


class TestHandleHealth:
    """Tests for health handler."""

    @pytest.mark.asyncio
    async def test_health_returns_status(self):
        """Test health returns expected structure."""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        mock_embed_client = MagicMock()
        mock_embed_client.health_check = AsyncMock(return_value={"status": "healthy"})

        with (
            patch(
                "research_kb_daemon.handler.get_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "research_kb_daemon.handler.get_embed_client",
                return_value=mock_embed_client,
            ),
        ):
            result = await handle_health({})

            assert "status" in result
            assert "uptime_seconds" in result
            assert "database" in result
            assert "embed_server" in result


class TestHandleStats:
    """Tests for stats handler."""

    @pytest.mark.asyncio
    async def test_stats_returns_counts(self):
        """Test stats returns database counts."""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=[100, 5000, 10000, 500, 20000])
        mock_conn.fetch = AsyncMock(
            return_value=[
                {"source_type": "paper", "count": 60},
                {"source_type": "textbook", "count": 40},
            ]
        )
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        with patch(
            "research_kb_daemon.handler.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            result = await handle_stats({})

            assert result["sources"] == 100
            assert result["chunks"] == 5000
            assert result["concepts"] == 10000
            assert result["citations"] == 500
            assert result["relationships"] == 20000
            assert result["source_types"]["paper"] == 60
            assert result["source_types"]["textbook"] == 40
