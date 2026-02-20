"""Tests for ResearchKBClient API client.

Tests HTTP client functionality with mocked httpx responses.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx


# Import the module under test
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "packages/dashboard/src"))

from research_kb_dashboard.api_client import ResearchKBClient, get_api_client

pytestmark = pytest.mark.unit


class TestClientLifecycle:
    """Tests for client creation and lifecycle."""

    def test_default_base_url(self):
        """Client uses default base URL when not specified."""
        with patch.dict("os.environ", {}, clear=True):
            client = ResearchKBClient()
            assert client.base_url == "http://localhost:8000"

    def test_custom_base_url(self):
        """Client accepts custom base URL."""
        client = ResearchKBClient(base_url="http://custom:9000")
        assert client.base_url == "http://custom:9000"

    def test_env_var_base_url(self):
        """Client reads base URL from environment variable."""
        with patch.dict("os.environ", {"RESEARCH_KB_API_URL": "http://from-env:8080"}):
            client = ResearchKBClient()
            assert client.base_url == "http://from-env:8080"

    def test_custom_timeout(self):
        """Client accepts custom timeout."""
        client = ResearchKBClient(timeout=60.0)
        assert client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_get_client_creates_singleton(self):
        """_get_client creates and reuses httpx client."""
        client = ResearchKBClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value = mock_instance

            # First call creates client
            result1 = await client._get_client()
            assert mock_client_class.called
            assert result1 == mock_instance

            # Second call reuses client
            mock_client_class.reset_mock()
            result2 = await client._get_client()
            assert not mock_client_class.called
            assert result2 == mock_instance

    @pytest.mark.asyncio
    async def test_close_client(self):
        """close() properly closes the HTTP client."""
        client = ResearchKBClient()
        mock_http = AsyncMock()
        client._client = mock_http

        await client.close()

        mock_http.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self):
        """close() handles uninitialized client gracefully."""
        client = ResearchKBClient()
        assert client._client is None

        # Should not raise
        await client.close()


class TestGetStats:
    """Tests for get_stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self):
        """get_stats returns parsed JSON response."""
        client = ResearchKBClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "sources": 294,
            "chunks": 142962,
            "concepts": 283714,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.get_stats()

        mock_http.get.assert_called_once_with("/stats")
        mock_response.raise_for_status.assert_called_once()
        assert result["sources"] == 294
        assert result["chunks"] == 142962

    @pytest.mark.asyncio
    async def test_get_stats_connection_error(self):
        """get_stats raises on connection errors."""
        client = ResearchKBClient()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        client._client = mock_http

        with pytest.raises(httpx.ConnectError):
            await client.get_stats()


class TestHealthCheck:
    """Tests for health_check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """health_check returns health status."""
        client = ResearchKBClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "healthy",
            "database": "ok",
            "embedding_model": "loaded",
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.health_check()

        mock_http.get.assert_called_once_with("/health")
        assert result["status"] == "healthy"


class TestSearch:
    """Tests for search endpoint."""

    @pytest.mark.asyncio
    async def test_search_with_defaults(self):
        """search sends correct default parameters."""
        client = ResearchKBClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": "instrumental variables",
            "results": [],
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        await client.search("instrumental variables")

        call_args = mock_http.post.call_args
        assert call_args[0][0] == "/search"

        payload = call_args[1]["json"]
        assert payload["query"] == "instrumental variables"
        assert payload["limit"] == 20
        assert payload["context_type"] == "balanced"
        assert payload["use_graph"] is True
        assert payload["use_rerank"] is True

    @pytest.mark.asyncio
    async def test_search_with_all_params(self):
        """search sends all custom parameters."""
        client = ResearchKBClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"query": "test", "results": []}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        await client.search(
            query="DML",
            limit=50,
            context_type="building",
            source_filter="PAPER",
            use_graph=False,
            graph_weight=0.1,
            use_rerank=False,
            use_expand=False,
        )

        payload = mock_http.post.call_args[1]["json"]
        assert payload["query"] == "DML"
        assert payload["limit"] == 50
        assert payload["context_type"] == "building"
        assert payload["source_filter"] == "PAPER"
        assert payload["use_graph"] is False
        assert payload["graph_weight"] == 0.1
        assert payload["use_rerank"] is False
        assert payload["use_expand"] is False

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """search handles empty results gracefully."""
        client = ResearchKBClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": "nonexistent xyz",
            "results": [],
            "total_count": 0,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.search("nonexistent xyz")

        assert result["results"] == []
        assert result["total_count"] == 0


class TestSources:
    """Tests for source-related endpoints."""

    @pytest.mark.asyncio
    async def test_list_sources_default(self):
        """list_sources uses default parameters."""
        client = ResearchKBClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "sources": [{"id": str(uuid4()), "title": "Test Paper"}],
            "total": 1,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.list_sources()

        call_args = mock_http.get.call_args
        assert call_args[0][0] == "/sources"

        params = call_args[1]["params"]
        assert params["limit"] == 100
        assert params["offset"] == 0
        assert "source_type" not in params

    @pytest.mark.asyncio
    async def test_list_sources_with_filter(self):
        """list_sources includes source_type filter."""
        client = ResearchKBClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"sources": [], "total": 0}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        await client.list_sources(limit=50, offset=10, source_type="PAPER")

        params = mock_http.get.call_args[1]["params"]
        assert params["limit"] == 50
        assert params["offset"] == 10
        assert params["source_type"] == "PAPER"

    @pytest.mark.asyncio
    async def test_get_source(self):
        """get_source fetches source by ID."""
        client = ResearchKBClient()
        source_id = str(uuid4())

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": source_id,
            "title": "Test Paper",
            "chunks": [],
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.get_source(source_id)

        mock_http.get.assert_called_once_with(f"/sources/{source_id}")
        assert result["id"] == source_id

    @pytest.mark.asyncio
    async def test_get_source_citations(self):
        """get_source_citations fetches citation data."""
        client = ResearchKBClient()
        source_id = str(uuid4())

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "citing_sources": [{"id": str(uuid4())}],
            "cited_sources": [],
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.get_source_citations(source_id)

        mock_http.get.assert_called_once_with(f"/sources/{source_id}/citations")
        assert len(result["citing_sources"]) == 1


class TestConcepts:
    """Tests for concept-related endpoints."""

    @pytest.mark.asyncio
    async def test_list_concepts_default(self):
        """list_concepts uses default parameters."""
        client = ResearchKBClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "concepts": [],
            "total": 0,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        await client.list_concepts()

        params = mock_http.get.call_args[1]["params"]
        assert params["limit"] == 100
        assert "query" not in params
        assert "concept_type" not in params

    @pytest.mark.asyncio
    async def test_list_concepts_with_search(self):
        """list_concepts includes search query."""
        client = ResearchKBClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"concepts": [], "total": 0}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        await client.list_concepts(query="IV", limit=50, concept_type="METHOD")

        params = mock_http.get.call_args[1]["params"]
        assert params["query"] == "IV"
        assert params["limit"] == 50
        assert params["concept_type"] == "METHOD"


class TestGraph:
    """Tests for graph-related endpoints."""

    @pytest.mark.asyncio
    async def test_get_graph_neighborhood(self):
        """get_graph_neighborhood fetches concept neighborhood."""
        client = ResearchKBClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "center": {"id": "123", "name": "DML"},
            "nodes": [],
            "edges": [],
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.get_graph_neighborhood("DML", hops=2, limit=50)

        call_args = mock_http.get.call_args
        assert "/graph/neighborhood/DML" in call_args[0][0]
        assert call_args[1]["params"]["hops"] == 2
        assert call_args[1]["params"]["limit"] == 50

    @pytest.mark.asyncio
    async def test_get_graph_path(self):
        """get_graph_path fetches shortest path between concepts."""
        client = ResearchKBClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "from_concept": "DML",
            "to_concept": "IV",
            "path": ["DML", "LATE", "IV"],
            "path_length": 2,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.get_graph_path("DML", "IV")

        mock_http.get.assert_called_once_with("/graph/path/DML/IV")
        assert result["path_length"] == 2


class TestConvenienceFunction:
    """Tests for get_api_client convenience function."""

    @pytest.mark.asyncio
    async def test_get_api_client(self):
        """get_api_client returns new client instance."""
        client = await get_api_client()

        assert isinstance(client, ResearchKBClient)
        assert client._client is None  # Not yet initialized
