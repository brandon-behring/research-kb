"""Tests for research-kb client SDK."""

import json
import os
import socket
import tempfile
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

from research_kb_client import (
    ConnectionError,
    DaemonClient,
    ResearchKBError,
    SearchResponse,
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
        from research_kb_client import (
            SearchResult,
            SearchResultChunk,
            SearchResultSource,
        )

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
            response = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "result": {"status": "healthy", "uptime_seconds": 100},
                    "id": request.get("id", 1),
                }
            )
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


class TestProtocolCompatibility:
    """Tests verifying the daemon protocol matches what bridges expect.

    These tests ensure that the JSON-RPC 2.0 protocol used by DaemonClient
    is compatible with external bridge integrations.

    The mock servers handle multiple connections because DaemonClient.search()
    calls _is_daemon_available() (health check) before the actual search.
    """

    @pytest.fixture
    def mock_socket_path(self, tmp_path):
        """Create temp socket path."""
        yield str(tmp_path / "test.sock")

    def _start_mock_server(self, socket_path, responses, captured_requests):
        """Start a mock daemon server that handles multiple connections.

        Args:
            socket_path: Unix socket path to bind
            responses: Dict mapping method -> response result.
                       If callable, called with (request) -> result.
            captured_requests: List to append received requests to.
        """
        import threading

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(socket_path)
        srv.listen(5)
        srv.settimeout(5.0)
        stop = threading.Event()

        def serve():
            while not stop.is_set():
                try:
                    conn, _ = srv.accept()
                    conn.settimeout(2.0)
                    data = b""
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk

                    request = json.loads(data.decode())
                    captured_requests.append(request)

                    method = request.get("method", "")
                    result = responses.get(method, [])
                    if callable(result):
                        result = result(request)

                    response = json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "result": result,
                            "id": request.get("id", 1),
                        }
                    )
                    conn.sendall(response.encode())
                    conn.close()
                except socket.timeout:
                    continue
                except Exception:
                    break

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        return srv, stop, thread

    def test_request_format_matches_bridge(self, mock_socket_path):
        """Client _send_request produces JSON-RPC 2.0 compatible with daemon server.py."""
        captured = []
        srv, stop, thread = self._start_mock_server(
            mock_socket_path,
            responses={
                "health": {"status": "healthy", "uptime_seconds": 100},
                "search": [],
            },
            captured_requests=captured,
        )

        try:
            client = DaemonClient(socket_path=mock_socket_path, daemon_timeout=2.0)
            try:
                client.search("test query", limit=3)
            except Exception:
                pass  # May fail parsing empty results

            # Find the search request (not the health check)
            search_reqs = [r for r in captured if r.get("method") == "search"]
            assert (
                len(search_reqs) >= 1
            ), f"Expected search request, got methods: {[r.get('method') for r in captured]}"

            req = search_reqs[0]
            # Verify JSON-RPC 2.0 structure required by server.py:122
            assert req.get("jsonrpc") == "2.0", "Must include jsonrpc version"
            assert req.get("method") == "search", "Method must be search"
            assert isinstance(req.get("params"), dict), "params must be dict"
            assert "id" in req, "Must include request id"
            assert req["params"].get("query") == "test query"
        finally:
            stop.set()
            srv.close()

    def test_response_format_flat_results(self, mock_socket_path):
        """Daemon returns flat result dicts via JSON-RPC 2.0 result field.

        The daemon's _result_to_dict produces flat keys:
            source_title, vector_score, content, etc.

        Note: _parse_search_response expects nested source/chunk sub-dicts
        (designed for CLI output). Flat daemon results lose source_title/content
        but vector_score/combined_score at top level are preserved.
        External bridges should use from_daemon_result() which handles flat
        format correctly — this is the recommended integration path.
        """
        # This is the actual format daemon's _result_to_dict produces
        flat_result = {
            "chunk_id": "chunk-001",
            "source_id": "src-001",
            "content": "IV estimation requires relevance and exclusion",
            "source_title": "Econometric Analysis",
            "source_authors": ["Wooldridge, J."],
            "source_year": 2010,
            "page_number": 100,
            "vector_score": 0.85,
            "combined_score": 0.92,
        }

        captured = []
        srv, stop, thread = self._start_mock_server(
            mock_socket_path,
            responses={
                "health": {"status": "healthy", "uptime_seconds": 100},
                "search": [flat_result],
            },
            captured_requests=captured,
        )

        try:
            client = DaemonClient(socket_path=mock_socket_path, daemon_timeout=2.0)
            response = client.search("instrumental variables", limit=1)

            # Client parses flat results — vector_score at top level is preserved
            assert len(response.results) >= 1
            result = response.results[0]
            assert result.vector_score == 0.85
            assert result.score == 0.92  # combined_score at top level
            # Note: source.title falls back to "Unknown" because flat format
            # doesn't have nested source.title — this is a known limitation
            # of _parse_search_response (designed for CLI nested format).
            # ResearchKBBridge.from_daemon_result() handles this correctly.
        finally:
            stop.set()
            srv.close()

    def test_error_response_format(self, mock_socket_path):
        """JSON-RPC 2.0 error format is correctly handled."""
        import threading

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(mock_socket_path)
        srv.listen(5)
        srv.settimeout(5.0)

        request_count = 0

        def serve():
            nonlocal request_count
            while True:
                try:
                    conn, _ = srv.accept()
                    conn.settimeout(2.0)
                    data = b""
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    request = json.loads(data.decode())
                    request_count += 1
                    method = request.get("method", "")

                    if method == "health":
                        # Health OK so it tries daemon for search
                        response = json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "result": {"status": "healthy"},
                                "id": request.get("id", 1),
                            }
                        )
                    else:
                        # Error for search
                        response = json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "error": {"code": -32600, "message": "Invalid Request"},
                                "id": request.get("id", 1),
                            }
                        )
                    conn.sendall(response.encode())
                    conn.close()
                except socket.timeout:
                    continue
                except Exception:
                    break

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        try:
            client = DaemonClient(
                socket_path=mock_socket_path,
                daemon_timeout=2.0,
                cli_path="/nonexistent/cli",  # No CLI fallback
            )
            with pytest.raises((ResearchKBError, ConnectionError)):
                client.search("test")
        finally:
            srv.close()


@pytest.mark.integration
@pytest.mark.requires_embedding
class TestLiveIntegration:
    """Integration tests against live daemon.

    Only run when daemon is actually available.
    Requires embedding server + populated database.
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
