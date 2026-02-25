"""Contract tests for EmbeddingClient — validates request/response shapes.

These tests mock the Unix socket at the protocol level using monkeypatch,
verifying that EmbeddingClient:
- Sends correct JSON request shapes
- Parses response formats correctly
- Handles batch splitting at batch_size boundary
- Handles error/timeout/connection failure scenarios
- Validates 1024-dimensional embedding vectors

Uses socket mocking (not respx, since this is a Unix socket client).
"""

import json
import socket
from unittest.mock import MagicMock, patch

import pytest

from research_kb_pdf.embedding_client import EmbeddingClient

pytestmark = pytest.mark.unit


def _mock_socket_response(response_dict: dict):
    """Create a mock socket that returns a JSON response.

    Returns a mock socket class that can be used as socket.socket replacement.
    """
    response_bytes = json.dumps(response_dict).encode("utf-8")

    mock_sock = MagicMock()
    mock_sock.settimeout = MagicMock()
    mock_sock.connect = MagicMock()
    mock_sock.sendall = MagicMock()
    mock_sock.shutdown = MagicMock()
    mock_sock.close = MagicMock()

    # recv returns data once, then empty bytes
    mock_sock.recv = MagicMock(side_effect=[response_bytes, b""])

    return mock_sock


class TestPing:
    """Tests for EmbeddingClient.ping() contract."""

    @patch("research_kb_pdf.embedding_client.socket.socket")
    def test_ping_sends_correct_request(self, mock_socket_cls):
        """ping sends {"action": "ping"} to server."""
        mock_sock = _mock_socket_response(
            {"status": "ok", "device": "cuda", "model": "bge-large", "dim": 1024}
        )
        mock_socket_cls.return_value = mock_sock

        client = EmbeddingClient("/tmp/test.sock")
        result = client.ping()

        # Verify request shape
        sent_bytes = mock_sock.sendall.call_args[0][0]
        sent_json = json.loads(sent_bytes.decode("utf-8"))
        assert sent_json == {"action": "ping"}

        # Verify response parsing
        assert result["status"] == "ok"
        assert result["dim"] == 1024

    @patch("research_kb_pdf.embedding_client.socket.socket")
    def test_ping_error_response_raises_value_error(self, mock_socket_cls):
        """Server error response raises ValueError."""
        mock_sock = _mock_socket_response({"error": "model not loaded"})
        mock_socket_cls.return_value = mock_sock

        client = EmbeddingClient("/tmp/test.sock")

        with pytest.raises(ValueError, match="Embedding server error"):
            client.ping()


class TestEmbed:
    """Tests for EmbeddingClient.embed() contract."""

    @patch("research_kb_pdf.embedding_client.socket.socket")
    def test_embed_sends_text_action(self, mock_socket_cls):
        """embed sends {"action": "embed", "text": ...} request."""
        embedding = [0.1] * 1024
        mock_sock = _mock_socket_response({"embedding": embedding})
        mock_socket_cls.return_value = mock_sock

        client = EmbeddingClient("/tmp/test.sock")
        result = client.embed("Introduction to causality")

        # Verify request shape
        sent_bytes = mock_sock.sendall.call_args[0][0]
        sent_json = json.loads(sent_bytes.decode("utf-8"))
        assert sent_json["action"] == "embed"
        assert sent_json["text"] == "Introduction to causality"

        # Verify response: 1024-dim vector
        assert len(result) == 1024
        assert result[0] == pytest.approx(0.1)

    @patch("research_kb_pdf.embedding_client.socket.socket")
    def test_embed_query_sends_query_action(self, mock_socket_cls):
        """embed_query sends {"action": "embed_query", "text": ...}."""
        mock_sock = _mock_socket_response({"embedding": [0.2] * 1024})
        mock_socket_cls.return_value = mock_sock

        client = EmbeddingClient("/tmp/test.sock")
        result = client.embed_query("instrumental variables")

        sent_bytes = mock_sock.sendall.call_args[0][0]
        sent_json = json.loads(sent_bytes.decode("utf-8"))
        assert sent_json["action"] == "embed_query"
        assert sent_json["text"] == "instrumental variables"
        assert len(result) == 1024


class TestEmbedBatch:
    """Tests for EmbeddingClient.embed_batch() contract."""

    @patch("research_kb_pdf.embedding_client.socket.socket")
    def test_small_batch_single_request(self, mock_socket_cls):
        """Batch <= batch_size sends single request."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = [[0.1] * 1024 for _ in texts]
        mock_sock = _mock_socket_response({"embeddings": embeddings})
        mock_socket_cls.return_value = mock_sock

        client = EmbeddingClient("/tmp/test.sock")
        result = client.embed_batch(texts, batch_size=32)

        # Verify request shape
        sent_bytes = mock_sock.sendall.call_args[0][0]
        sent_json = json.loads(sent_bytes.decode("utf-8"))
        assert sent_json["action"] == "embed_batch"
        assert sent_json["texts"] == texts
        assert sent_json["batch_size"] == 32

        # Verify response
        assert len(result) == 3
        assert all(len(e) == 1024 for e in result)

    @patch("research_kb_pdf.embedding_client.socket.socket")
    def test_large_batch_splits_into_chunks(self, mock_socket_cls):
        """Batch > batch_size splits into multiple socket requests."""
        batch_size = 2
        texts = ["Text 1", "Text 2", "Text 3"]  # 3 texts, batch_size=2 → 2 requests

        call_count = 0

        def mock_socket_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First call: batch of 2
                n = min(2, 3 - (call_count - 1) * 2)
                embeddings = [[0.1] * 1024 for _ in range(n)]
            else:
                # Second call: batch of 1
                embeddings = [[0.1] * 1024]
            return _mock_socket_response({"embeddings": embeddings})

        mock_socket_cls.side_effect = mock_socket_factory

        client = EmbeddingClient("/tmp/test.sock")
        result = client.embed_batch(texts, batch_size=batch_size)

        assert len(result) == 3
        assert call_count == 2  # 2 socket connections for 3 texts with batch_size=2


class TestEmbedQueryBatch:
    """Tests for EmbeddingClient.embed_query_batch() contract."""

    @patch("research_kb_pdf.embedding_client.socket.socket")
    def test_query_batch_sends_correct_action(self, mock_socket_cls):
        """embed_query_batch sends {"action": "embed_query_batch", ...}."""
        texts = ["IV", "DML"]
        mock_sock = _mock_socket_response({"embeddings": [[0.1] * 1024, [0.2] * 1024]})
        mock_socket_cls.return_value = mock_sock

        client = EmbeddingClient("/tmp/test.sock")
        result = client.embed_query_batch(texts)

        sent_bytes = mock_sock.sendall.call_args[0][0]
        sent_json = json.loads(sent_bytes.decode("utf-8"))
        assert sent_json["action"] == "embed_query_batch"
        assert sent_json["texts"] == ["IV", "DML"]
        assert len(result) == 2


class TestErrorHandling:
    """Tests for error scenarios."""

    def test_connection_error_on_missing_socket(self):
        """FileNotFoundError (missing socket) raises ConnectionError."""
        client = EmbeddingClient("/tmp/nonexistent_socket_12345.sock")

        with pytest.raises(ConnectionError, match="not running"):
            client.ping()

    @patch("research_kb_pdf.embedding_client.socket.socket")
    def test_timeout_raises_timeout_error(self, mock_socket_cls):
        """Socket timeout raises TimeoutError."""
        mock_sock = MagicMock()
        mock_sock.settimeout = MagicMock()
        mock_sock.connect = MagicMock()
        mock_sock.sendall = MagicMock()
        mock_sock.shutdown = MagicMock()
        mock_sock.recv = MagicMock(side_effect=socket.timeout("timed out"))
        mock_sock.close = MagicMock()
        mock_socket_cls.return_value = mock_sock

        client = EmbeddingClient("/tmp/test.sock", timeout=1.0)

        with pytest.raises(TimeoutError, match="timed out"):
            client.embed("test")

    @patch("research_kb_pdf.embedding_client.socket.socket")
    def test_server_error_raises_value_error(self, mock_socket_cls):
        """Server-side error response raises ValueError."""
        mock_sock = _mock_socket_response({"error": "internal error"})
        mock_socket_cls.return_value = mock_sock

        client = EmbeddingClient("/tmp/test.sock")

        with pytest.raises(ValueError, match="Embedding server error"):
            client.embed("test")


class TestEmbedChunks:
    """Tests for EmbeddingClient.embed_chunks() contract."""

    @patch("research_kb_pdf.embedding_client.socket.socket")
    def test_embed_chunks_extracts_content(self, mock_socket_cls):
        """embed_chunks extracts .content from TextChunk objects."""
        from unittest.mock import MagicMock as MM

        chunks = [MM(content="chunk 1 text"), MM(content="chunk 2 text")]
        mock_sock = _mock_socket_response({"embeddings": [[0.1] * 1024, [0.2] * 1024]})
        mock_socket_cls.return_value = mock_sock

        client = EmbeddingClient("/tmp/test.sock")
        result = client.embed_chunks(chunks)

        assert len(result) == 2
        # Verify texts sent are chunk contents
        sent_bytes = mock_sock.sendall.call_args[0][0]
        sent_json = json.loads(sent_bytes.decode("utf-8"))
        assert sent_json["texts"] == ["chunk 1 text", "chunk 2 text"]
