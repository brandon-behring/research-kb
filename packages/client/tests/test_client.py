"""Tests for research-kb client SDK."""

import json
import os
import socket
import tempfile
import threading
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

from research_kb_client import (
    ConceptInfo,
    ConceptNeighborhood,
    ConceptRelationship,
    ConnectionError,
    DaemonClient,
    HealthStatus,
    ResearchKBError,
    SearchResponse,
    SearchResult,
    SearchResultChunk,
    SearchResultSource,
    StatsResponse,
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


# ---------------------------------------------------------------------------
# Model Validation Tests
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for response model construction and validation."""

    def test_health_status_defaults(self):
        """HealthStatus defaults for optional fields."""
        health = HealthStatus(status="healthy")
        assert health.database == "unknown"
        assert health.embed_server == "unknown"
        assert health.rerank_server is None
        assert health.uptime_seconds is None

    def test_health_status_full(self):
        """HealthStatus with all fields."""
        health = HealthStatus(
            status="healthy",
            database="healthy",
            embed_server="healthy",
            rerank_server="healthy",
            uptime_seconds=3600.0,
        )
        assert health.uptime_seconds == 3600.0
        assert health.rerank_server == "healthy"

    def test_stats_response_defaults(self):
        """StatsResponse defaults for optional fields."""
        stats = StatsResponse(sources=10, chunks=100, concepts=50, relationships=75)
        assert stats.citations == 0
        assert stats.chunk_concepts == 0

    def test_stats_response_full(self):
        """StatsResponse with all fields."""
        stats = StatsResponse(
            sources=495,
            chunks=228000,
            concepts=312000,
            relationships=744000,
            citations=10758,
            chunk_concepts=600000,
        )
        assert stats.sources == 495
        assert stats.chunk_concepts == 600000

    def test_search_result_source_defaults(self):
        """SearchResultSource defaults."""
        source = SearchResultSource(id="abc", title="Test")
        assert source.authors is None
        assert source.year is None
        assert source.source_type == "unknown"

    def test_search_result_chunk_defaults(self):
        """SearchResultChunk optional fields."""
        chunk = SearchResultChunk(id="c1", content="text")
        assert chunk.page_number is None
        assert chunk.section_header is None

    def test_search_result_score_fields(self):
        """SearchResult optional score fields."""
        result = SearchResult(
            source=SearchResultSource(id="s1", title="Paper"),
            chunk=SearchResultChunk(id="c1", content="text"),
            score=0.85,
        )
        assert result.fts_score is None
        assert result.vector_score is None
        assert result.graph_score is None
        assert result.citation_score is None
        assert result.concepts == []

    def test_search_response_defaults(self):
        """SearchResponse optional fields."""
        resp = SearchResponse(results=[], query="test")
        assert resp.expanded_query is None
        assert resp.total_count is None

    def test_concept_info(self):
        """ConceptInfo construction."""
        concept = ConceptInfo(
            id="uuid1",
            name="IV",
            concept_type="METHOD",
            description="Instrumental variables",
        )
        assert concept.name == "IV"
        assert concept.concept_type == "METHOD"

    def test_concept_relationship(self):
        """ConceptRelationship construction."""
        rel = ConceptRelationship(
            source_id="a",
            target_id="b",
            relationship_type="REQUIRES",
        )
        assert rel.relationship_type == "REQUIRES"
        assert rel.source_name is None

    def test_concept_neighborhood(self):
        """ConceptNeighborhood construction."""
        center = ConceptInfo(id="c1", name="DML", concept_type="METHOD")
        connected = ConceptInfo(id="c2", name="Cross-Fitting", concept_type="TECHNIQUE")
        rel = ConceptRelationship(source_id="c1", target_id="c2", relationship_type="USES")
        neighborhood = ConceptNeighborhood(
            center=center,
            connected_concepts=[connected],
            relationships=[rel],
            hops=2,
        )
        assert len(neighborhood.connected_concepts) == 1
        assert neighborhood.hops == 2


# ---------------------------------------------------------------------------
# Client Connection Tests
# ---------------------------------------------------------------------------


class TestClientConnection:
    """Tests for DaemonClient connection logic."""

    def test_default_cli_path(self):
        """Default CLI path includes research-kb."""
        client = DaemonClient()
        assert "research-kb" in client.cli_path

    def test_custom_timeout(self):
        """Can set custom daemon and CLI timeouts."""
        client = DaemonClient(daemon_timeout=5.0, cli_timeout=30.0)
        assert client.daemon_timeout == 5.0
        assert client.cli_timeout == 30.0

    def test_is_available_with_cli(self, tmp_path):
        """is_available returns True when CLI exists and is executable."""
        # Create a fake CLI executable
        cli = tmp_path / "research-kb"
        cli.write_text("#!/bin/bash\n")
        cli.chmod(0o755)

        client = DaemonClient(
            socket_path="/nonexistent/socket.sock",
            cli_path=str(cli),
        )
        assert client.is_available()

    def test_is_cli_available_nonexistent(self):
        """_is_cli_available returns False for nonexistent path."""
        client = DaemonClient(cli_path="/nonexistent/research-kb")
        assert not client._is_cli_available()

    def test_is_daemon_available_no_socket(self, tmp_path):
        """_is_daemon_available returns False when socket file doesn't exist."""
        client = DaemonClient(socket_path=str(tmp_path / "nonexistent.sock"))
        assert not client._is_daemon_available()

    def test_env_var_socket_path(self):
        """Socket path respects RESEARCH_KB_SOCKET_PATH env var."""
        with patch.dict(os.environ, {"RESEARCH_KB_SOCKET_PATH": "/custom/socket.sock"}):
            from research_kb_client.socket_client import _default_socket_path

            assert _default_socket_path() == "/custom/socket.sock"

    def test_env_var_cli_path(self):
        """CLI path respects RESEARCH_KB_CLI_PATH env var."""
        with patch.dict(os.environ, {"RESEARCH_KB_CLI_PATH": "/custom/research-kb"}):
            from research_kb_client.socket_client import _default_cli_path

            assert _default_cli_path() == "/custom/research-kb"


# ---------------------------------------------------------------------------
# Search Validation Tests
# ---------------------------------------------------------------------------


class TestSearchValidation:
    """Tests for search input validation and response parsing."""

    def test_search_none_query_raises(self):
        """None query raises ValueError."""
        client = DaemonClient()
        with pytest.raises(ValueError, match="cannot be empty"):
            client.search(None)

    def test_search_unavailable_raises_connection_error(self):
        """Search raises ConnectionError when both daemon and CLI unavailable."""
        client = DaemonClient(
            socket_path="/nonexistent/socket.sock",
            cli_path="/nonexistent/cli",
        )
        with pytest.raises(ConnectionError):
            client.search("test query")

    def test_parse_search_response_empty(self):
        """_parse_search_response handles empty results."""
        client = DaemonClient()
        response = client._parse_search_response({"results": []}, "test")
        assert response.query == "test"
        assert response.results == []

    def test_parse_search_response_with_expanded_query(self):
        """_parse_search_response preserves expanded_query."""
        client = DaemonClient()
        data = {
            "results": [],
            "expanded_query": "instrumental variables estimation",
            "total_count": 42,
        }
        response = client._parse_search_response(data, "IV")
        assert response.expanded_query == "instrumental variables estimation"
        assert response.total_count == 42

    def test_parse_search_response_authors_string(self):
        """_parse_search_response handles authors as comma-separated string."""
        client = DaemonClient()
        data = {
            "results": [
                {
                    "source": {
                        "id": "s1",
                        "title": "Test",
                        "authors": "Angrist, J., Imbens, G.",
                    },
                    "chunk": {"id": "c1", "content": "text"},
                    "combined_score": 0.8,
                }
            ]
        }
        response = client._parse_search_response(data, "query")
        assert response.results[0].source.authors == ["Angrist", "J.", "Imbens", "G."]

    def test_parse_search_response_score_fallback(self):
        """_parse_search_response uses combined_score as fallback for score."""
        client = DaemonClient()
        data = {
            "results": [
                {
                    "source": {"id": "s1", "title": "Test"},
                    "chunk": {"id": "c1", "content": "text"},
                    "combined_score": 0.92,
                }
            ]
        }
        response = client._parse_search_response(data, "query")
        assert response.results[0].score == 0.92

    def test_parse_search_response_content_truncation(self):
        """_parse_search_response truncates content to 500 chars."""
        client = DaemonClient()
        long_content = "x" * 1000
        data = {
            "results": [
                {
                    "source": {"id": "s1", "title": "Test"},
                    "chunk": {"id": "c1", "content": long_content},
                    "score": 0.5,
                }
            ]
        }
        response = client._parse_search_response(data, "query")
        assert len(response.results[0].chunk.content) == 500


# ---------------------------------------------------------------------------
# Health/Stats Parsing Tests
# ---------------------------------------------------------------------------


class TestHealthStatsParsing:
    """Tests for health and stats response parsing."""

    @pytest.fixture
    def mock_socket_path(self, tmp_path):
        """Create temp socket path."""
        yield str(tmp_path / "test.sock")

    def _start_mock_server(self, socket_path, responses):
        """Start a mock daemon server that handles multiple connections."""
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
                    method = request.get("method", "")
                    result = responses.get(method, {})
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
        return srv, stop

    def test_health_nested_status(self, mock_socket_path):
        """Health parses nested database status dicts."""
        srv, stop = self._start_mock_server(
            mock_socket_path,
            {
                "health": {
                    "status": "healthy",
                    "database": {"status": "healthy", "latency_ms": 5},
                    "embed_server": {"status": "healthy"},
                    "rerank_server": {"status": "available"},
                    "uptime_seconds": 3600,
                },
            },
        )
        try:
            client = DaemonClient(socket_path=mock_socket_path, daemon_timeout=2.0)
            health = client.health()
            assert health.status == "healthy"
            assert health.database == "healthy"
            assert health.embed_server == "healthy"
            assert health.rerank_server == "available"
            assert health.uptime_seconds == 3600
        finally:
            stop.set()
            srv.close()

    def test_stats_response(self, mock_socket_path):
        """Stats parses all fields from daemon response."""
        srv, stop = self._start_mock_server(
            mock_socket_path,
            {
                "health": {"status": "healthy"},
                "stats": {
                    "sources": 495,
                    "chunks": 228000,
                    "concepts": 312000,
                    "relationships": 744000,
                    "citations": 10758,
                    "chunk_concepts": 600000,
                },
            },
        )
        try:
            client = DaemonClient(socket_path=mock_socket_path, daemon_timeout=2.0)
            stats = client.stats()
            assert stats.sources == 495
            assert stats.chunks == 228000
            assert stats.concepts == 312000
            assert stats.relationships == 744000
            assert stats.citations == 10758
            assert stats.chunk_concepts == 600000
        finally:
            stop.set()
            srv.close()

    def test_health_empty_response(self, mock_socket_path):
        """Health handles None/empty response gracefully."""
        srv, stop = self._start_mock_server(
            mock_socket_path,
            {
                "health": None,
            },
        )
        try:
            client = DaemonClient(socket_path=mock_socket_path, daemon_timeout=2.0)
            health = client.health()
            assert health.status == "unknown"
            assert health.database == "unknown"
        finally:
            stop.set()
            srv.close()

    def test_stats_empty_response(self, mock_socket_path):
        """Stats handles None/empty response gracefully."""
        srv, stop = self._start_mock_server(
            mock_socket_path,
            {
                "health": {"status": "healthy"},
                "stats": None,
            },
        )
        try:
            client = DaemonClient(socket_path=mock_socket_path, daemon_timeout=2.0)
            stats = client.stats()
            assert stats.sources == 0
            assert stats.chunks == 0
        finally:
            stop.set()
            srv.close()


# ---------------------------------------------------------------------------
# Convenience Functions Extended Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctionsExtended:
    """Extended tests for convenience functions."""

    def test_search_or_none_catches_exception(self):
        """search_or_none returns None on exception."""
        with patch.object(DaemonClient, "is_available", return_value=True):
            with patch.object(DaemonClient, "search", side_effect=RuntimeError("boom")):
                result = search_or_none("test query")
                assert result is None

    def test_get_methodology_context_formats_results(self):
        """get_methodology_context returns formatted markdown."""
        mock_response = SearchResponse(
            results=[
                SearchResult(
                    source=SearchResultSource(
                        id="s1",
                        title="IV Methods Paper",
                        authors=["Angrist", "Imbens"],
                        year=2020,
                    ),
                    chunk=SearchResultChunk(
                        id="c1",
                        content="IV estimation provides causal identification.",
                    ),
                    score=0.85,
                    concepts=["IV", "estimation"],
                ),
            ],
            query="instrumental variables",
        )
        with patch.object(DaemonClient, "is_available", return_value=True):
            with patch.object(DaemonClient, "search", return_value=mock_response):
                result = get_methodology_context("IV")

        assert "## Research Context" in result
        assert "IV Methods Paper" in result
        assert "Angrist" in result
        assert "Score: 0.85" in result
        assert "IV, estimation" in result

    def test_get_methodology_context_no_results(self):
        """get_methodology_context returns empty string for no results."""
        mock_response = SearchResponse(results=[], query="nothing")
        with patch.object(DaemonClient, "is_available", return_value=True):
            with patch.object(DaemonClient, "search", return_value=mock_response):
                result = get_methodology_context("nothing")
                assert result == ""
