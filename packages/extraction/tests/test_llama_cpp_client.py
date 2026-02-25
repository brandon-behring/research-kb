"""Tests for LlamaCpp client for direct GPU inference."""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from research_kb_extraction.models import ChunkExtraction

pytestmark = pytest.mark.unit


class TestLlamaCppClientInit:
    """Tests for LlamaCppClient initialization."""

    def test_default_config(self):
        """Test default configuration."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        client = LlamaCppClient()

        assert client.n_ctx == 2048
        assert client.n_gpu_layers == 20
        assert client.temperature == 0.1
        assert client.max_tokens == 4096
        assert client.n_batch == 512
        assert client.verbose is False
        assert client._llm is None  # Lazy loading

    def test_custom_config(self):
        """Test custom configuration."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        client = LlamaCppClient(
            model_path="/custom/path/model.gguf",
            n_ctx=4096,
            n_gpu_layers=32,
            temperature=0.3,
            max_tokens=8192,
            n_batch=256,
            verbose=True,
        )

        assert client.model_path == Path("/custom/path/model.gguf")
        assert client.n_ctx == 4096
        assert client.n_gpu_layers == 32
        assert client.temperature == 0.3
        assert client.max_tokens == 8192
        assert client.n_batch == 256
        assert client.verbose is True

    def test_model_name_from_path(self):
        """Test model name is extracted from path stem."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        client = LlamaCppClient(model_path="/models/my-model-Q4_K_M.gguf")
        assert client._model_name == "my-model-Q4_K_M"

    def test_extraction_method_property(self):
        """Test extraction_method returns correct identifier."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        client = LlamaCppClient(model_path="/models/test-model.gguf")
        assert client.extraction_method == "llamacpp:test-model"


class TestLlamaCppAvailability:
    """Tests for availability checks."""

    async def test_is_available_model_exists(self):
        """Test is_available returns True when model file exists."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        with patch.object(Path, "exists", return_value=True):
            with patch.dict("sys.modules", {"llama_cpp": MagicMock()}):
                client = LlamaCppClient(model_path="/models/exists.gguf")
                result = await client.is_available()
                assert result is True

    async def test_is_available_model_not_found(self):
        """Test is_available returns False when model not found."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        with patch.object(Path, "exists", return_value=False):
            client = LlamaCppClient(model_path="/models/missing.gguf")
            result = await client.is_available()
            assert result is False

    async def test_is_available_import_error(self):
        """Test is_available returns False when llama_cpp not installed."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        with patch.object(Path, "exists", return_value=True):
            # Simulate import error
            import sys

            original_modules = sys.modules.copy()
            sys.modules["llama_cpp"] = None

            try:
                client = LlamaCppClient(model_path="/models/exists.gguf")
                # is_available tries to import llama_cpp and catches ImportError
                # Need to make the import fail
                with patch.dict("sys.modules", {"llama_cpp": None}, clear=False):

                    def raise_import(*args, **kwargs):
                        raise ImportError("No module")

                    with patch("builtins.__import__", side_effect=raise_import):
                        result = await client.is_available()
                        assert result is False
            finally:
                sys.modules.clear()
                sys.modules.update(original_modules)


class TestLlamaCppModelLoading:
    """Tests for lazy model loading."""

    def test_lazy_load_on_first_use(self):
        """Test model is not loaded until first use."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        client = LlamaCppClient()
        assert client._llm is None  # Not loaded yet

    def test_load_model_creates_llm(self):
        """Test _load_model creates Llama instance."""
        from research_kb_extraction.llama_cpp_client import (
            LlamaCppClient,
        )

        with patch.object(Path, "exists", return_value=True):
            mock_llama_class = MagicMock()
            mock_llama_instance = MagicMock()
            mock_llama_class.return_value = mock_llama_instance

            with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
                client = LlamaCppClient(model_path="/models/test.gguf")
                client._load_model()

                mock_llama_class.assert_called_once()
                assert client._llm is mock_llama_instance

    def test_load_model_missing_file_raises_error(self):
        """Test _load_model raises error when model file missing."""
        from research_kb_extraction.llama_cpp_client import (
            LlamaCppClient,
            LlamaCppError,
        )

        mock_llama_module = MagicMock()
        with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
            with patch.object(Path, "exists", return_value=False):
                client = LlamaCppClient(model_path="/models/missing.gguf")

                with pytest.raises(LlamaCppError) as exc_info:
                    client._load_model()

                assert "Model not found" in str(exc_info.value)

    def test_load_model_import_error_raises(self):
        """Test _load_model raises error when llama_cpp not installed."""
        from research_kb_extraction.llama_cpp_client import (
            LlamaCppClient,
            LlamaCppError,
        )

        client = LlamaCppClient(model_path="/models/test.gguf")

        # Make the import fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "llama_cpp":
                raise ImportError("Not installed")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(LlamaCppError) as exc_info:
                client._load_model()

            assert "llama-cpp-python not installed" in str(exc_info.value)


class TestLlamaCppGenerate:
    """Tests for text generation."""

    def test_generate_basic(self):
        """Test basic generation with JSON mode."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"result": "test"}'}}]
        }

        client = LlamaCppClient()
        client._llm = mock_llm

        result = client.generate("Test prompt", json_mode=True)

        assert result == '{"result": "test"}'
        mock_llm.create_chat_completion.assert_called_once()

    def test_generate_without_json_mode(self):
        """Test generation without JSON mode."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Plain text response"}}]
        }

        client = LlamaCppClient()
        client._llm = mock_llm

        result = client.generate("Test prompt", json_mode=False)

        assert result == "Plain text response"
        call_kwargs = mock_llm.create_chat_completion.call_args.kwargs
        assert "response_format" not in call_kwargs

    def test_generate_with_system_prompt(self):
        """Test generation includes system prompt in messages."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "{}"}}]}

        client = LlamaCppClient()
        client._llm = mock_llm

        client.generate("User prompt", system="Custom system", json_mode=True)

        call_kwargs = mock_llm.create_chat_completion.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Custom system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User prompt"

    def test_generate_error_raises_llama_cpp_error(self):
        """Test generation errors are wrapped in LlamaCppError."""
        from research_kb_extraction.llama_cpp_client import (
            LlamaCppClient,
            LlamaCppError,
        )

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.side_effect = RuntimeError("CUDA error")

        client = LlamaCppClient()
        client._llm = mock_llm

        with pytest.raises(LlamaCppError) as exc_info:
            client.generate("Test")

        assert "CUDA error" in str(exc_info.value)


class TestLlamaCppExtraction:
    """Tests for concept extraction."""

    async def test_extract_concepts_success(self):
        """Test successful concept extraction."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        extraction_json = {
            "concepts": [
                {
                    "name": "double machine learning",
                    "concept_type": "method",
                    "definition": "A debiasing technique",
                    "aliases": ["DML"],
                    "confidence": 0.9,
                }
            ],
            "relationships": [
                {
                    "source_concept": "double machine learning",
                    "target_concept": "cross-fitting",
                    "relationship_type": "USES",
                    "confidence": 0.85,
                }
            ],
        }

        client = LlamaCppClient()

        with patch.object(client, "generate", return_value=json.dumps(extraction_json)):
            result = await client.extract_concepts("Text about DML", domain_id="causal_inference")

            assert isinstance(result, ChunkExtraction)
            assert len(result.concepts) == 1
            assert result.concepts[0].name == "double machine learning"
            assert len(result.relationships) == 1

    async def test_extract_concepts_invalid_json(self):
        """Test handling of invalid JSON returns empty extraction."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        client = LlamaCppClient()

        with patch.object(client, "generate", return_value="Not valid JSON"):
            result = await client.extract_concepts("Sample text", domain_id="causal_inference")

            # Should return empty extraction, not raise
            assert isinstance(result, ChunkExtraction)
            assert len(result.concepts) == 0

    async def test_extract_concepts_validation_error(self):
        """Test handling of schema validation errors."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        invalid_json = {"concepts": [{"wrong_field": "value"}], "relationships": []}

        client = LlamaCppClient()

        with patch.object(client, "generate", return_value=json.dumps(invalid_json)):
            result = await client.extract_concepts("Sample text", domain_id="causal_inference")

            # Should return empty extraction, not raise
            assert isinstance(result, ChunkExtraction)
            assert len(result.concepts) == 0


class TestLlamaCppContextManager:
    """Tests for async context manager."""

    async def test_context_manager_preloads_model(self):
        """Test context manager preloads model on entry."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        client = LlamaCppClient()

        with patch.object(client, "_load_model") as mock_load:
            async with client:
                mock_load.assert_called_once()

    async def test_context_manager_closes_on_exit(self):
        """Test context manager calls close on exit."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        client = LlamaCppClient()

        with patch.object(client, "_load_model"):
            with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
                async with client:
                    pass

                mock_close.assert_called_once()

    async def test_close_releases_model(self):
        """Test close() releases model resources."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        mock_llm = MagicMock()
        client = LlamaCppClient()
        client._llm = mock_llm

        await client.close()

        assert client._llm is None


class TestLlamaCppChatFormat:
    """Tests for Llama 3.1 chat prompt formatting."""

    def test_format_chat_prompt(self):
        """Test Llama 3.1 chat format."""
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        client = LlamaCppClient()
        result = client._format_chat_prompt("You are helpful.", "Hello!")

        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "You are helpful." in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "Hello!" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result


# Import AsyncMock for context manager test
from unittest.mock import AsyncMock
