"""Contract tests for EmbeddingServer — validates the server-side API contract.

Tests the EmbeddingServer class interface (embed, embed_query, embed_batch)
without loading the actual model. Validates:
- Method signatures and return types
- Action routing in handle_request
- Response format contract
- Error handling for invalid inputs

The model itself is mocked (SentenceTransformer is heavy).
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestEmbeddingServerContract:
    """Tests for EmbeddingServer class methods.

    These test the server's API contract by mocking the model.
    """

    @patch("research_kb_pdf.embed_server.SentenceTransformer")
    @patch("research_kb_pdf.embed_server.torch")
    def test_embed_returns_list_of_floats(self, mock_torch, mock_st):
        """embed() returns list[float] of correct dimension."""
        import numpy as np

        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 1024])
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st.return_value = mock_model

        from research_kb_pdf.embed_server import EmbeddingServer

        server = EmbeddingServer(model_name="test", device="cpu", revision=None)
        result = server.embed("Hello world")

        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(v, float) for v in result)

    @patch("research_kb_pdf.embed_server.SentenceTransformer")
    @patch("research_kb_pdf.embed_server.torch")
    def test_embed_query_adds_instruction_prefix(self, mock_torch, mock_st):
        """embed_query() prefixes text with BGE query instruction."""
        import numpy as np

        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 1024])
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st.return_value = mock_model

        from research_kb_pdf.embed_server import EmbeddingServer, QUERY_INSTRUCTION

        server = EmbeddingServer(model_name="test", device="cpu", revision=None)
        result = server.embed_query("IV assumptions")

        # Verify the model.encode was called with prefixed text
        call_args = mock_model.encode.call_args[0][0]
        expected_text = QUERY_INSTRUCTION + "IV assumptions"
        assert call_args == [expected_text]

        assert len(result) == 1024

    @patch("research_kb_pdf.embed_server.SentenceTransformer")
    @patch("research_kb_pdf.embed_server.torch")
    def test_embed_batch_returns_list_of_embeddings(self, mock_torch, mock_st):
        """embed_batch() returns list[list[float]] matching input length."""
        import numpy as np

        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 1024, [0.2] * 1024])
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st.return_value = mock_model

        from research_kb_pdf.embed_server import EmbeddingServer

        server = EmbeddingServer(model_name="test", device="cpu", revision=None)
        result = server.embed_batch(["text 1", "text 2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(len(e) == 1024 for e in result)

    @patch("research_kb_pdf.embed_server.SentenceTransformer")
    @patch("research_kb_pdf.embed_server.torch")
    def test_server_stores_device_and_model_info(self, mock_torch, mock_st):
        """Server stores device and model metadata."""
        import numpy as np

        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 1024])
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st.return_value = mock_model

        from research_kb_pdf.embed_server import EmbeddingServer

        server = EmbeddingServer(model_name="test-model", device="cpu", revision="abc123")

        assert server.device == "cpu"
        assert server.model_name == "test-model"
        assert server.revision == "abc123"


class TestHandleRequest:
    """Tests for request routing (handle_request function)."""

    @patch("research_kb_pdf.embed_server.SentenceTransformer")
    @patch("research_kb_pdf.embed_server.torch")
    def test_ping_action(self, mock_torch, mock_st):
        """'ping' action returns status info."""
        import numpy as np

        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 1024])
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st.return_value = mock_model

        from research_kb_pdf.embed_server import EmbeddingServer

        server = EmbeddingServer(model_name="test", device="cpu", revision=None)

        # handle_request dispatches based on "action" key
        response = server.handle_request({"action": "ping"})

        assert response["status"] == "ok"
        assert response["dim"] == 1024

    @patch("research_kb_pdf.embed_server.SentenceTransformer")
    @patch("research_kb_pdf.embed_server.torch")
    def test_embed_action(self, mock_torch, mock_st):
        """'embed' action returns embedding."""
        import numpy as np

        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 1024])
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st.return_value = mock_model

        from research_kb_pdf.embed_server import EmbeddingServer

        server = EmbeddingServer(model_name="test", device="cpu", revision=None)
        response = server.handle_request({"action": "embed", "text": "test"})

        assert "embedding" in response
        assert len(response["embedding"]) == 1024

    @patch("research_kb_pdf.embed_server.SentenceTransformer")
    @patch("research_kb_pdf.embed_server.torch")
    def test_unknown_action_returns_error(self, mock_torch, mock_st):
        """Unknown action returns error response."""
        import numpy as np

        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 1024])
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st.return_value = mock_model

        from research_kb_pdf.embed_server import EmbeddingServer

        server = EmbeddingServer(model_name="test", device="cpu", revision=None)
        response = server.handle_request({"action": "nonexistent"})

        assert "error" in response
