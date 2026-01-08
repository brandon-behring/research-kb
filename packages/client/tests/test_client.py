"""Tests for research-kb client SDK."""

import json
import os
import socket
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from research_kb_client import (
    ConnectionError,
    DaemonClient,
    ResearchKBError,
    SearchResponse,
    TimeoutError,
    get_methodology_context,
    search_or_none,
)


class TestDaemonClient:
    """Tests for DaemonClient."""

    def test_default_socket_path_uses_user(self):
        """Socket path includes current user."""
        client = DaemonClient()
        user = os.environ.get("USER", "unknown")
        assert user in client.socket_path

    def test_custom_socket_path(self):
        """Can override socket path."""
        client = DaemonClient(socket_path="/custom/path.sock")
        assert client.socket_path == "/custom/path.sock"

    def test_is_available_false_when_no_socket_and_no_cli(self):
        """is_available returns False when both socket and CLI don't exist."""
        client = DaemonClient(
            socket_path="/nonexistent/socket.sock",
            cli_path="/nonexistent/cli",
        )
        assert not client.is_available()

    def test_search_empty_query_raises(self):
        """Empty query raises ValueError."""
        client = DaemonClient()
        with pytest.raises(ValueError, match="cannot be empty"):
            client.search("")

    def test_search_whitespace_query_raises(self):
        """Whitespace-only query raises ValueError."""
        client = DaemonClient()
        with pytest.raises(ValueError, match="cannot be empty"):
            client.search("   ")


class TestSearchResponse:
    """Tests for search response parsing."""

    def test_parse_minimal_response(self):
        """Parse response with minimal fields."""
        response = SearchResponse(
            results=[],
            query="test",
        )
        assert response.query == "test"
        assert response.results == []
        assert response.expanded_query is None

    def test_parse_full_response(self):
        """Parse response with all fields."""
        from research_kb_client import SearchResult, SearchResultChunk, SearchResultSource

        result = SearchResult(
            source=SearchResultSource(
                id="abc123",
                title="Test Paper",
                authors=["Author A", "Author B"],
                year=2020,
                source_type="paper",
            ),
            chunk=SearchResultChunk(
                id="chunk123",
                content="Test content here",
                page_number=42,
                section_header="Methods",
            ),
            score=0.85,
            fts_score=0.7,
            vector_score=0.9,
            concepts=["IV", "estimation"],
        )

        response = SearchResponse(
            results=[result],
            query="instrumental variables",
            expanded_query="instrumental variables estimation",
            total_count=100,
        )

        assert len(response.results) == 1
        assert response.results[0].source.title == "Test Paper"
        assert response.results[0].chunk.page_number == 42
        assert response.results[0].score == 0.85


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_search_or_none_returns_none_when_unavailable(self):
        """search_or_none returns None when client unavailable."""
        with patch.object(DaemonClient, "is_available", return_value=False):
            result = search_or_none("test query")
            assert result is None

    def test_get_methodology_context_empty_when_unavailable(self):
        """get_methodology_context returns empty string when unavailable."""
        with patch.object(DaemonClient, "is_available", return_value=False):
            result = get_methodology_context("test topic")
            assert result == ""


class TestMockDaemon:
    """Tests with mocked daemon socket."""

    @pytest.fixture
    def mock_socket_path(self):
        """Create temp socket path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test.sock")

    def test_ping_with_mock_server(self, mock_socket_path):
        """Test ping against mock server."""
        # Start mock server
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(mock_socket_path)
        server.listen(1)
        server.settimeout(1.0)

        def serve():
            conn, _ = server.accept()
            data = conn.recv(1024)
            request = json.loads(data.decode())
            # JSON-RPC 2.0 response format
            response = json.dumps({
                "jsonrpc": "2.0",
                "result": {"status": "healthy", "uptime_seconds": 100},
                "id": request.get("id", 1),
            })
            conn.sendall(response.encode())
            conn.close()

        import threading
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        # Test client
        client = DaemonClient(socket_path=mock_socket_path, daemon_timeout=1.0)
        assert client.ping()

        server.close()

    def test_connection_error_when_socket_gone(self, mock_socket_path):
        """ConnectionError when socket doesn't exist."""
        client = DaemonClient(socket_path=mock_socket_path)
        assert not client.ping()


@pytest.mark.integration
class TestLiveIntegration:
    """Integration tests against live daemon.

    Only run when daemon is actually available.
    """

    @pytest.fixture
    def live_client(self):
        """Get client connected to live daemon."""
        client = DaemonClient()
        if not client.is_available():
            pytest.skip("research-kb daemon not available")
        return client

    def test_health_check(self, live_client):
        """Health check returns valid response."""
        health = live_client.health()
        assert health.status in ["healthy", "degraded"]
        assert health.database in ["healthy", "unhealthy", "unknown"]

    def test_stats(self, live_client):
        """Stats returns counts."""
        stats = live_client.stats()
        assert stats.sources > 0
        assert stats.chunks > 0

    def test_search(self, live_client):
        """Search returns results."""
        response = live_client.search("instrumental variables", limit=3)
        assert len(response.results) <= 3
        assert response.query == "instrumental variables"
        if response.results:
            assert response.results[0].score > 0
