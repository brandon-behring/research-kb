"""Tests for Ollama client."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from research_kb_common import ExtractionValidationError
from research_kb_extraction.ollama_client import OllamaClient, OllamaError
from research_kb_extraction.models import ChunkExtraction

pytestmark = pytest.mark.unit


class TestOllamaClientInit:
    """Tests for OllamaClient initialization."""

    def test_default_config(self):
        """Test default configuration."""
        client = OllamaClient()

        assert client.model == "llama3.1:8b"
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 120.0
        assert client.temperature == 0.1
        assert client.num_ctx == 4096

    def test_custom_config(self):
        """Test custom configuration."""
        client = OllamaClient(
            model="hermes3:8b",
            base_url="http://custom:8080",
            timeout=60.0,
            temperature=0.5,
            num_ctx=8192,
        )

        assert client.model == "hermes3:8b"
        assert client.base_url == "http://custom:8080"
        assert client.timeout == 60.0
        assert client.temperature == 0.5
        assert client.num_ctx == 8192

    def test_trailing_slash_removed(self):
        """Test trailing slash is removed from base URL."""
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"


class TestOllamaAvailability:
    """Tests for availability checks."""

    async def test_is_available_success(self):
        """Test is_available returns True when server responds."""
        client = OllamaClient()

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            result = await client.is_available()

            assert result is True
            mock_http.get.assert_called_once_with("/api/tags")

    async def test_is_available_failure(self):
        """Test is_available returns False on error."""
        client = OllamaClient()

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_get.return_value = mock_http

            result = await client.is_available()

            assert result is False

    async def test_is_model_loaded(self):
        """Test is_model_loaded checks for model."""
        client = OllamaClient(model="llama3.1:8b")

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.1:8b"},
                    {"name": "hermes3:8b"},
                ]
            }
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            result = await client.is_model_loaded()

            assert result is True


class TestOllamaGenerate:
    """Tests for text generation."""

    async def test_generate_basic(self):
        """Test basic generation."""
        client = OllamaClient()

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"response": "Hello world"}
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            result = await client.generate("Say hello")

            assert result == "Hello world"

    async def test_generate_with_system_prompt(self):
        """Test generation with system prompt."""
        client = OllamaClient()

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"response": "Response"}
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            await client.generate("User prompt", system="System prompt")

            # Check system was included in payload
            call_args = mock_http.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["system"] == "System prompt"

    async def test_generate_json_mode(self):
        """Test JSON mode sets format."""
        client = OllamaClient()

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"response": "{}"}
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            await client.generate("Get JSON", json_mode=True)

            call_args = mock_http.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["format"] == "json"


class TestOllamaExtraction:
    """Tests for concept extraction."""

    async def test_extract_concepts_success(self):
        """Test successful concept extraction."""
        client = OllamaClient()

        extraction_json = {
            "concepts": [
                {
                    "name": "instrumental variables",
                    "concept_type": "method",
                    "definition": "A method for causal inference",
                    "aliases": ["IV"],
                    "confidence": 0.9,
                }
            ],
            "relationships": [],
        }

        with patch.object(client, "generate") as mock_generate:
            mock_generate.return_value = json.dumps(extraction_json)

            result = await client.extract_concepts(
                "Sample text about IV", domain_id="causal_inference"
            )

            assert isinstance(result, ChunkExtraction)
            assert len(result.concepts) == 1
            assert result.concepts[0].name == "instrumental variables"

    async def test_extract_concepts_invalid_json(self):
        """Test handling of invalid JSON response raises ExtractionValidationError."""
        client = OllamaClient()

        with patch.object(client, "generate") as mock_generate:
            mock_generate.return_value = "This is not JSON"

            # Should raise ExtractionValidationError on parse failure (fail fast)
            with pytest.raises(ExtractionValidationError) as exc_info:
                await client.extract_concepts("Sample text", domain_id="causal_inference")

            assert "Failed to parse JSON response" in str(exc_info.value)

    async def test_extract_concepts_empty_response(self):
        """Test handling of empty extraction."""
        client = OllamaClient()

        with patch.object(client, "generate") as mock_generate:
            mock_generate.return_value = json.dumps(
                {
                    "concepts": [],
                    "relationships": [],
                }
            )

            result = await client.extract_concepts("No concepts here", domain_id="causal_inference")

            assert isinstance(result, ChunkExtraction)
            assert len(result.concepts) == 0

    async def test_extract_batch(self):
        """Test batch extraction."""
        client = OllamaClient()

        extraction_json = json.dumps(
            {
                "concepts": [{"name": "test", "concept_type": "method", "confidence": 0.8}],
                "relationships": [],
            }
        )

        with patch.object(client, "generate") as mock_generate:
            mock_generate.return_value = extraction_json

            chunks = ["chunk 1", "chunk 2", "chunk 3"]
            results = await client.extract_batch(chunks, domain_id="causal_inference")

            assert len(results) == 3
            assert mock_generate.call_count == 3


class TestOllamaContextManager:
    """Tests for async context manager."""

    async def test_context_manager(self):
        """Test async context manager usage."""
        async with OllamaClient() as client:
            assert client is not None
            assert client.model == "llama3.1:8b"

        # Client should be closed after exiting context
        assert client._client is None


class TestOllamaErrors:
    """Tests for error handling."""

    async def test_http_error_raises_ollama_error(self):
        """Test HTTP errors are wrapped in OllamaError."""
        import httpx

        client = OllamaClient()

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Server error", request=MagicMock(), response=mock_response
                )
            )
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            with pytest.raises(OllamaError) as exc_info:
                await client.generate("test")

            assert "HTTP error" in str(exc_info.value)
