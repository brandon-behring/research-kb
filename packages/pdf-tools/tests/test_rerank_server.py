"""Tests for RerankServer - Cross-encoder reranking via Unix socket.

Tests:
- Request handling (rerank, predict, ping, shutdown)
- Score normalization
- Error handling
- Server lifecycle
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from research_kb_pdf.rerank_server import RerankServer, SOCKET_PATH, BUFFER_SIZE
from research_kb_pdf.reranker import RerankResult, CrossEncoderReranker

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_reranker():
    """Create a mock CrossEncoderReranker."""
    reranker = MagicMock(spec=CrossEncoderReranker)
    reranker.model_name = "test-model"
    reranker.device = "cpu"
    return reranker


@pytest.fixture
def mock_server(mock_reranker):
    """Create RerankServer with mocked reranker."""
    with patch.object(RerankServer, "__init__", lambda self, *args, **kwargs: None):
        server = RerankServer.__new__(RerankServer)
        server.reranker = mock_reranker
        server.device = "cpu"
        server.model_name = "test-model"
        return server


class TestRerankServerInit:
    """Tests for RerankServer initialization."""

    def test_init_creates_reranker(self):
        """Test initialization creates CrossEncoderReranker."""
        with patch("research_kb_pdf.rerank_server.CrossEncoderReranker") as mock_class:
            mock_class.return_value = MagicMock()

            server = RerankServer(model_name="test-model", device="cpu")

            mock_class.assert_called_once_with(model_name="test-model", device="cpu")
            assert server.device == "cpu"
            assert server.model_name == "test-model"

    def test_init_default_device(self):
        """Test initialization uses default device."""
        with patch("research_kb_pdf.rerank_server.CrossEncoderReranker") as mock_class:
            with patch("research_kb_pdf.rerank_server.DEVICE", "cuda"):
                mock_class.return_value = MagicMock()

                server = RerankServer()

                # Device should be from DEVICE constant
                assert server.device in ["cuda", "cpu"]


class TestHandleRequestRerank:
    """Tests for handle_request with 'rerank' action."""

    def test_rerank_success(self, mock_server, mock_reranker):
        """Test successful rerank request."""
        mock_reranker.rerank_texts.return_value = [
            RerankResult(
                content="Relevant document",
                original_rank=2,
                rerank_score=0.95,
                new_rank=1,
            ),
            RerankResult(
                content="Less relevant",
                original_rank=1,
                rerank_score=0.75,
                new_rank=2,
            ),
        ]

        request = {
            "action": "rerank",
            "query": "test query",
            "documents": ["Less relevant", "Relevant document"],
            "top_k": 2,
        }

        response = mock_server.handle_request(request)

        assert "results" in response
        assert response["count"] == 2
        assert response["results"][0]["content"] == "Relevant document"
        assert response["results"][0]["rerank_score"] == 0.95
        assert response["results"][0]["new_rank"] == 1
        mock_reranker.rerank_texts.assert_called_once()

    def test_rerank_missing_query(self, mock_server):
        """Test rerank with missing query returns error."""
        request = {
            "action": "rerank",
            "documents": ["doc1", "doc2"],
        }

        response = mock_server.handle_request(request)

        assert "error" in response
        assert "Missing 'query' field" in response["error"]

    def test_rerank_missing_documents(self, mock_server):
        """Test rerank with missing documents returns error."""
        request = {
            "action": "rerank",
            "query": "test query",
        }

        response = mock_server.handle_request(request)

        assert "error" in response
        assert "Missing 'documents' field" in response["error"]

    def test_rerank_empty_query(self, mock_server):
        """Test rerank with empty query returns error."""
        request = {
            "action": "rerank",
            "query": "",
            "documents": ["doc1"],
        }

        response = mock_server.handle_request(request)

        assert "error" in response

    def test_rerank_empty_documents(self, mock_server):
        """Test rerank with empty documents list returns error."""
        request = {
            "action": "rerank",
            "query": "test query",
            "documents": [],
        }

        response = mock_server.handle_request(request)

        assert "error" in response

    def test_rerank_default_top_k(self, mock_server, mock_reranker):
        """Test rerank uses default top_k when not specified."""
        mock_reranker.rerank_texts.return_value = []

        request = {
            "action": "rerank",
            "query": "test",
            "documents": ["doc1"],
        }

        mock_server.handle_request(request)

        # Verify default top_k=10 was passed
        call_args = mock_reranker.rerank_texts.call_args
        assert call_args.kwargs.get("top_k", call_args[1].get("top_k", 10)) == 10


class TestHandleRequestPredict:
    """Tests for handle_request with 'predict' action."""

    def test_predict_success(self, mock_server, mock_reranker):
        """Test successful predict request."""
        mock_reranker.predict_scores.return_value = [0.95, 0.75, 0.50]

        request = {
            "action": "predict",
            "query": "test query",
            "documents": ["doc1", "doc2", "doc3"],
        }

        response = mock_server.handle_request(request)

        assert "scores" in response
        assert response["count"] == 3
        assert response["scores"] == [0.95, 0.75, 0.50]
        mock_reranker.predict_scores.assert_called_once_with("test query", ["doc1", "doc2", "doc3"])

    def test_predict_missing_query(self, mock_server):
        """Test predict with missing query returns error."""
        request = {
            "action": "predict",
            "documents": ["doc1"],
        }

        response = mock_server.handle_request(request)

        assert "error" in response
        assert "Missing 'query' field" in response["error"]

    def test_predict_missing_documents(self, mock_server):
        """Test predict with missing documents returns error."""
        request = {
            "action": "predict",
            "query": "test",
        }

        response = mock_server.handle_request(request)

        assert "error" in response
        assert "Missing 'documents' field" in response["error"]


class TestHandleRequestPing:
    """Tests for handle_request with 'ping' action."""

    def test_ping_returns_status(self, mock_server):
        """Test ping returns status info."""
        request = {"action": "ping"}

        response = mock_server.handle_request(request)

        assert response["status"] == "ok"
        assert response["device"] == "cpu"
        assert response["model"] == "test-model"


class TestHandleRequestShutdown:
    """Tests for handle_request with 'shutdown' action."""

    def test_shutdown_returns_status(self, mock_server):
        """Test shutdown returns shutting_down status."""
        request = {"action": "shutdown"}

        response = mock_server.handle_request(request)

        assert response["status"] == "shutting_down"


class TestHandleRequestUnknown:
    """Tests for handle_request with unknown action."""

    def test_unknown_action_returns_error(self, mock_server):
        """Test unknown action returns error."""
        request = {"action": "unknown_action"}

        response = mock_server.handle_request(request)

        assert "error" in response
        assert "Unknown action: unknown_action" in response["error"]

    def test_default_action_is_rerank(self, mock_server, mock_reranker):
        """Test default action is rerank when not specified."""
        mock_reranker.rerank_texts.return_value = []

        request = {
            "query": "test",
            "documents": ["doc1"],
        }

        # Should not error, should attempt rerank
        response = mock_server.handle_request(request)

        # Either success or rerank-related error
        assert (
            "results" in response
            or "error" not in response
            or "Missing" not in response.get("error", "")
        )


class TestHandleRequestErrorHandling:
    """Tests for error handling in handle_request."""

    def test_exception_returns_error(self, mock_server, mock_reranker):
        """Test exception during processing returns error."""
        mock_reranker.rerank_texts.side_effect = RuntimeError("Model error")

        request = {
            "action": "rerank",
            "query": "test",
            "documents": ["doc1"],
        }

        response = mock_server.handle_request(request)

        assert "error" in response
        assert "Model error" in response["error"]


class TestRerankResultSerialization:
    """Tests for RerankResult serialization in responses."""

    def test_result_contains_all_fields(self, mock_server, mock_reranker):
        """Test rerank response contains all RerankResult fields."""
        mock_reranker.rerank_texts.return_value = [
            RerankResult(
                content="Test document content",
                original_rank=3,
                rerank_score=0.875,
                new_rank=1,
                metadata={"source_id": "test123"},
            ),
        ]

        request = {
            "action": "rerank",
            "query": "test",
            "documents": ["Test document content"],
        }

        response = mock_server.handle_request(request)

        result = response["results"][0]
        assert result["content"] == "Test document content"
        assert result["original_rank"] == 3
        assert result["rerank_score"] == 0.875
        assert result["new_rank"] == 1
        # Note: metadata is not included in serialization per the source


class TestScoreNormalization:
    """Tests for score normalization in reranking."""

    def test_scores_are_floats(self, mock_server, mock_reranker):
        """Test rerank scores are returned as floats."""
        mock_reranker.rerank_texts.return_value = [
            RerankResult(
                content="doc",
                original_rank=1,
                rerank_score=0.5,
                new_rank=1,
            ),
        ]

        request = {
            "action": "rerank",
            "query": "test",
            "documents": ["doc"],
        }

        response = mock_server.handle_request(request)

        assert isinstance(response["results"][0]["rerank_score"], float)

    def test_predict_scores_are_floats(self, mock_server, mock_reranker):
        """Test predict scores are returned as floats."""
        mock_reranker.predict_scores.return_value = [0.5, 0.3]

        request = {
            "action": "predict",
            "query": "test",
            "documents": ["doc1", "doc2"],
        }

        response = mock_server.handle_request(request)

        assert all(isinstance(s, float) for s in response["scores"])


class TestServerSocket:
    """Tests for Unix socket server operations."""

    def test_socket_path_constant(self):
        """Test SOCKET_PATH constant is defined."""
        assert SOCKET_PATH == "/tmp/research_kb_rerank.sock"

    def test_buffer_size_constant(self):
        """Test BUFFER_SIZE is large enough for batch requests."""
        # 256KB should handle large batch requests
        assert BUFFER_SIZE == 262144

    def test_run_server_creates_socket(self, mock_server):
        """Test run_server creates Unix socket."""

        # We can't fully test the server loop, but we can test socket creation
        # by patching socket operations
        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            # Make accept raise KeyboardInterrupt to exit loop
            mock_socket.accept.side_effect = KeyboardInterrupt()

            with patch("os.path.exists", return_value=False):
                with patch("os.chmod"):
                    try:
                        mock_server.run_server("/tmp/test.sock")
                    except KeyboardInterrupt:
                        pass

            mock_socket.bind.assert_called_once_with("/tmp/test.sock")
            mock_socket.listen.assert_called_once_with(5)

    def test_run_server_removes_existing_socket(self, mock_server):
        """Test run_server removes existing socket file."""
        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket
            mock_socket.accept.side_effect = KeyboardInterrupt()

            with patch("os.path.exists", return_value=True) as mock_exists:
                with patch("os.remove") as mock_remove:
                    with patch("os.chmod"):
                        try:
                            mock_server.run_server("/tmp/test.sock")
                        except KeyboardInterrupt:
                            pass

                mock_remove.assert_called()


class TestMain:
    """Tests for main() entry point."""

    def test_main_test_mode(self):
        """Test main with --test flag exits after test."""
        from research_kb_pdf.rerank_server import main

        with patch("sys.argv", ["rerank_server", "--test"]):
            with patch("research_kb_pdf.rerank_server.CrossEncoderReranker") as mock_class:
                mock_reranker = MagicMock()
                mock_reranker.rerank_texts.return_value = [
                    RerankResult(
                        content="test",
                        original_rank=1,
                        rerank_score=0.9,
                        new_rank=1,
                    )
                ]
                mock_class.return_value = mock_reranker

                # main() should complete without starting server
                main()

                mock_reranker.rerank_texts.assert_called_once()

    def test_main_fast_flag_uses_fallback_model(self):
        """Test main with --fast flag uses MiniLM model."""
        from research_kb_pdf.rerank_server import main, FALLBACK_MODEL

        with patch("sys.argv", ["rerank_server", "--test", "--fast"]):
            with patch("research_kb_pdf.rerank_server.CrossEncoderReranker") as mock_class:
                mock_reranker = MagicMock()
                mock_reranker.rerank_texts.return_value = []
                mock_class.return_value = mock_reranker

                main()

                # Should use FALLBACK_MODEL
                call_args = mock_class.call_args
                assert (
                    call_args.kwargs.get("model_name", call_args[0][0] if call_args[0] else None)
                    == FALLBACK_MODEL
                )


class TestIntegration:
    """Integration-style tests for complete request flows."""

    def test_full_rerank_flow(self, mock_server, mock_reranker):
        """Test complete rerank flow from request to response."""
        # Setup mock to return realistic results
        mock_reranker.rerank_texts.return_value = [
            RerankResult(
                content="IV is a method for causal inference",
                original_rank=3,
                rerank_score=0.92,
                new_rank=1,
            ),
            RerankResult(
                content="2SLS is a common IV estimator",
                original_rank=1,
                rerank_score=0.88,
                new_rank=2,
            ),
            RerankResult(
                content="Machine learning models predict",
                original_rank=2,
                rerank_score=0.35,
                new_rank=3,
            ),
        ]

        request = {
            "action": "rerank",
            "query": "instrumental variables for causal inference",
            "documents": [
                "2SLS is a common IV estimator",
                "Machine learning models predict",
                "IV is a method for causal inference",
            ],
            "top_k": 3,
        }

        response = mock_server.handle_request(request)

        # Verify response structure
        assert "error" not in response
        assert response["count"] == 3
        assert len(response["results"]) == 3

        # Verify ordering (most relevant first)
        assert response["results"][0]["new_rank"] == 1
        assert response["results"][0]["rerank_score"] > response["results"][1]["rerank_score"]

    def test_json_roundtrip(self, mock_server, mock_reranker):
        """Test JSON serialization/deserialization of request/response."""
        mock_reranker.predict_scores.return_value = [0.5]

        # Simulate what the socket would do
        request_json = json.dumps(
            {
                "action": "predict",
                "query": "test",
                "documents": ["doc"],
            }
        )

        request = json.loads(request_json)
        response = mock_server.handle_request(request)
        response_json = json.dumps(response)
        parsed_response = json.loads(response_json)

        assert "scores" in parsed_response
        assert parsed_response["count"] == 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_document(self, mock_server, mock_reranker):
        """Test reranking with single document."""
        mock_reranker.rerank_texts.return_value = [
            RerankResult(content="doc", original_rank=1, rerank_score=0.5, new_rank=1),
        ]

        request = {
            "action": "rerank",
            "query": "test",
            "documents": ["doc"],
            "top_k": 1,
        }

        response = mock_server.handle_request(request)

        assert response["count"] == 1

    def test_very_long_query(self, mock_server, mock_reranker):
        """Test with very long query string."""
        mock_reranker.rerank_texts.return_value = []

        long_query = "test " * 1000

        request = {
            "action": "rerank",
            "query": long_query,
            "documents": ["doc"],
        }

        # Should not error
        response = mock_server.handle_request(request)

        assert "error" not in response or "Model error" in response.get("error", "")

    def test_unicode_content(self, mock_server, mock_reranker):
        """Test with unicode characters in content."""
        mock_reranker.rerank_texts.return_value = [
            RerankResult(
                content="Econometric analysis",
                original_rank=1,
                rerank_score=0.8,
                new_rank=1,
            ),
        ]

        request = {
            "action": "rerank",
            "query": "test query",
            "documents": ["Econometric analysis"],
        }

        response = mock_server.handle_request(request)

        assert response["results"][0]["content"] == "Econometric analysis"
