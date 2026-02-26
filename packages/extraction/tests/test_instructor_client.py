"""Tests for Instructor-based Ollama client with automatic validation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from research_kb_extraction.models import ChunkExtraction

pytestmark = [pytest.mark.unit, pytest.mark.requires_instructor]


class TestInstructorClientInit:
    """Tests for InstructorOllamaClient initialization."""

    def test_default_config(self):
        """Test default configuration."""
        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MagicMock()

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient()

            assert client.model == "llama3.1:8b"
            assert client.base_url == "http://localhost:11434"
            assert client.timeout == 120.0
            assert client.temperature == 0.1
            assert client.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MagicMock()

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient(
                model="hermes3:8b",
                base_url="http://custom:8080",
                timeout=60.0,
                temperature=0.5,
                max_retries=5,
            )

            assert client.model == "hermes3:8b"
            assert client.base_url == "http://custom:8080"
            assert client.timeout == 60.0
            assert client.temperature == 0.5
            assert client.max_retries == 5

    def test_instructor_provider_called_correctly(self):
        """Test instructor.from_provider is called with correct args."""
        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MagicMock()

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient(
                model="llama3.1:8b",
                base_url="http://localhost:11434",
                timeout=120.0,
            )

            # Should be called twice (sync and async clients)
            assert mock_provider.call_count == 2

            # First call: sync client
            call_args_1 = mock_provider.call_args_list[0]
            assert call_args_1.args[0] == "ollama/llama3.1:8b"
            assert call_args_1.kwargs["base_url"] == "http://localhost:11434/v1"
            assert call_args_1.kwargs["timeout"] == 120.0

            # Second call: async client
            call_args_2 = mock_provider.call_args_list[1]
            assert call_args_2.kwargs["async_client"] is True

    def test_extraction_method_property(self):
        """Test extraction_method returns correct identifier."""
        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MagicMock()

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient(model="llama3.1:8b")
            assert client.extraction_method == "instructor:llama3.1:8b"


class TestInstructorAvailability:
    """Tests for availability checks."""

    async def test_is_available_success(self):
        """Test is_available returns True when Ollama responds."""
        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MagicMock()

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient()

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await client.is_available()

                assert result is True

    async def test_is_available_failure(self):
        """Test is_available returns False on connection error."""
        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MagicMock()

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient()

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await client.is_available()

                assert result is False


class TestInstructorExtraction:
    """Tests for concept extraction with automatic validation."""

    async def test_extract_concepts_success(self):
        """Test successful concept extraction."""
        with patch("instructor.from_provider") as mock_provider:
            mock_async_client = AsyncMock()

            # Create a proper ChunkExtraction with real ExtractedConcept
            from research_kb_extraction.models import ExtractedConcept

            expected_extraction = ChunkExtraction(
                concepts=[
                    ExtractedConcept(
                        name="instrumental variables",
                        concept_type="method",
                        definition="A causal method",
                        aliases=["IV"],
                        confidence=0.9,
                    )
                ],
                relationships=[],
            )
            mock_async_client.create = AsyncMock(return_value=expected_extraction)

            # Mock returns different clients for sync/async
            mock_provider.side_effect = [MagicMock(), mock_async_client]

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient()
            result = await client.extract_concepts("Text about IV", domain_id="causal_inference")

            assert isinstance(result, ChunkExtraction)
            mock_async_client.create.assert_called_once()

    async def test_extract_concepts_uses_max_retries(self):
        """Test extraction uses configured max_retries."""
        with patch("instructor.from_provider") as mock_provider:
            mock_async_client = AsyncMock()
            mock_async_client.create = AsyncMock(return_value=ChunkExtraction())

            mock_provider.side_effect = [MagicMock(), mock_async_client]

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient(max_retries=5)
            await client.extract_concepts("Test text", domain_id="causal_inference")

            call_kwargs = mock_async_client.create.call_args.kwargs
            assert call_kwargs["max_retries"] == 5

    async def test_extract_concepts_uses_temperature(self):
        """Test extraction uses configured temperature."""
        with patch("instructor.from_provider") as mock_provider:
            mock_async_client = AsyncMock()
            mock_async_client.create = AsyncMock(return_value=ChunkExtraction())

            mock_provider.side_effect = [MagicMock(), mock_async_client]

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient(temperature=0.7)
            await client.extract_concepts("Test text", domain_id="causal_inference")

            call_kwargs = mock_async_client.create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.7

    async def test_extract_concepts_failure_returns_empty(self):
        """Test that extraction failures return empty ChunkExtraction."""
        with patch("instructor.from_provider") as mock_provider:
            mock_async_client = AsyncMock()
            mock_async_client.create = AsyncMock(
                side_effect=Exception("Validation failed after retries")
            )

            mock_provider.side_effect = [MagicMock(), mock_async_client]

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient()
            result = await client.extract_concepts("Test text", domain_id="causal_inference")

            # Should return empty extraction, not raise
            assert isinstance(result, ChunkExtraction)
            assert len(result.concepts) == 0
            assert len(result.relationships) == 0

    async def test_extract_concepts_passes_response_model(self):
        """Test extraction passes ChunkExtraction as response_model."""
        with patch("instructor.from_provider") as mock_provider:
            mock_async_client = AsyncMock()
            mock_async_client.create = AsyncMock(return_value=ChunkExtraction())

            mock_provider.side_effect = [MagicMock(), mock_async_client]

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient()
            await client.extract_concepts("Test text", domain_id="causal_inference")

            call_kwargs = mock_async_client.create.call_args.kwargs
            assert call_kwargs["response_model"] is ChunkExtraction


class TestInstructorContextManager:
    """Tests for async context manager."""

    async def test_context_manager_usage(self):
        """Test async context manager works correctly."""
        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MagicMock()

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            async with InstructorOllamaClient() as client:
                assert client is not None
                assert client.model == "llama3.1:8b"

    async def test_close_is_noop(self):
        """Test close() completes without error."""
        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MagicMock()

            from research_kb_extraction.instructor_client import InstructorOllamaClient

            client = InstructorOllamaClient()
            await client.close()  # Should not raise
