"""Tests for Anthropic Claude API client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import os

from research_kb_extraction.models import ChunkExtraction


class TestAnthropicClientInit:
    """Tests for AnthropicClient initialization."""

    def test_default_config_with_env_key(self):
        """Test default configuration with API key from environment."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}):
            with patch("anthropic.Anthropic"):
                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient()

                assert client.model_name == "haiku"
                assert client.model_id == "claude-3-haiku-20240307"
                assert client.temperature == 0.1
                assert client.max_tokens == 4096

    def test_custom_config(self):
        """Test custom configuration."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient(
                    model="sonnet",
                    temperature=0.5,
                    max_tokens=8192,
                )

                assert client.model_name == "sonnet"
                assert client.model_id == "claude-sonnet-4-20250514"
                assert client.temperature == 0.5
                assert client.max_tokens == 8192

    def test_model_name_mapping(self):
        """Test all model name mappings work correctly."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from research_kb_extraction.anthropic_client import AnthropicClient

                # Test standard mappings
                for name, expected_id in [
                    ("haiku", "claude-3-haiku-20240307"),
                    ("haiku-3.5", "claude-3-5-haiku-20241022"),
                    ("haiku-4.5", "claude-haiku-4-5-20251001"),
                    ("sonnet", "claude-sonnet-4-20250514"),
                    ("opus", "claude-opus-4-5-20251101"),
                ]:
                    client = AnthropicClient(model=name)
                    assert client.model_id == expected_id, f"Failed for {name}"

    def test_full_model_id_passthrough(self):
        """Test that full model IDs are passed through unchanged."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient(model="claude-3-opus-20240229")
                assert client.model_id == "claude-3-opus-20240229"

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ANTHROPIC_API_KEY if present
            os.environ.pop("ANTHROPIC_API_KEY", None)

            with patch("anthropic.Anthropic"):
                from research_kb_extraction.anthropic_client import AnthropicClient

                with pytest.raises(ValueError) as exc_info:
                    AnthropicClient()

                assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_api_key_whitespace_stripped(self):
        """Test that API key whitespace is stripped."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "  test-key  \n"}):
            with patch("anthropic.Anthropic") as mock_anthropic:
                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient()

                # Verify stripped key was used
                mock_anthropic.assert_called_once_with(api_key="test-key")

    def test_explicit_api_key_overrides_env(self):
        """Test that explicit API key overrides environment variable."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            with patch("anthropic.Anthropic") as mock_anthropic:
                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient(api_key="explicit-key")

                mock_anthropic.assert_called_once_with(api_key="explicit-key")


class TestAnthropicAvailability:
    """Tests for availability checks."""

    @pytest.mark.asyncio
    async def test_is_available_success(self):
        """Test is_available returns True when API responds."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_client = MagicMock()
                mock_client.messages.count_tokens = MagicMock(return_value={"tokens": 5})
                mock_anthropic.return_value = mock_client

                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient()
                result = await client.is_available()

                assert result is True

    @pytest.mark.asyncio
    async def test_is_available_failure(self):
        """Test is_available returns False on API error."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_client = MagicMock()
                mock_client.messages.count_tokens = MagicMock(
                    side_effect=Exception("API Error")
                )
                mock_anthropic.return_value = mock_client

                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient()
                result = await client.is_available()

                assert result is False


class TestAnthropicExtraction:
    """Tests for concept extraction."""

    @pytest.mark.asyncio
    async def test_extract_concepts_success(self):
        """Test successful concept extraction via tool_use."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_anthropic:
                # Mock tool_use response
                mock_block = MagicMock()
                mock_block.type = "tool_use"
                mock_block.name = "extract_concepts"
                mock_block.input = {
                    "concepts": [
                        {
                            "name": "instrumental variables",
                            "concept_type": "method",
                            "definition": "A causal method",
                            "aliases": ["IV"],
                            "confidence": 0.95,
                        }
                    ],
                    "relationships": [],
                }

                mock_message = MagicMock()
                mock_message.content = [mock_block]

                mock_client = MagicMock()
                mock_client.messages.create = MagicMock(return_value=mock_message)
                mock_anthropic.return_value = mock_client

                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient()
                result = await client.extract_concepts("Sample text about IV")

                assert isinstance(result, ChunkExtraction)
                assert len(result.concepts) == 1
                assert result.concepts[0].name == "instrumental variables"
                assert result.concepts[0].concept_type == "method"

    @pytest.mark.asyncio
    async def test_extract_concepts_no_tool_use(self):
        """Test fallback when no tool_use in response."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_anthropic:
                # Mock response without tool_use
                mock_block = MagicMock()
                mock_block.type = "text"
                mock_block.text = "No concepts found"

                mock_message = MagicMock()
                mock_message.content = [mock_block]

                mock_client = MagicMock()
                mock_client.messages.create = MagicMock(return_value=mock_message)
                mock_anthropic.return_value = mock_client

                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient()
                result = await client.extract_concepts("Sample text")

                assert isinstance(result, ChunkExtraction)
                assert len(result.concepts) == 0

    @pytest.mark.asyncio
    async def test_extract_concepts_validation_error_returns_empty(self):
        """Test that validation errors return empty extraction (graceful degradation)."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_anthropic:
                # Mock tool_use with invalid data
                mock_block = MagicMock()
                mock_block.type = "tool_use"
                mock_block.name = "extract_concepts"
                mock_block.input = {
                    "concepts": [{"invalid": "schema"}],  # Missing required fields
                    "relationships": [],
                }

                mock_message = MagicMock()
                mock_message.content = [mock_block]

                mock_client = MagicMock()
                mock_client.messages.create = MagicMock(return_value=mock_message)
                mock_anthropic.return_value = mock_client

                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient()
                result = await client.extract_concepts("Sample text")

                # Should return empty extraction, not raise
                assert isinstance(result, ChunkExtraction)
                assert len(result.concepts) == 0

    @pytest.mark.asyncio
    async def test_extract_concepts_api_error(self):
        """Test that API errors are wrapped in AnthropicError."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_client = MagicMock()
                mock_client.messages.create = MagicMock(
                    side_effect=Exception("Rate limited")
                )
                mock_anthropic.return_value = mock_client

                from research_kb_extraction.anthropic_client import (
                    AnthropicClient,
                    AnthropicError,
                )

                client = AnthropicClient()

                with pytest.raises(AnthropicError) as exc_info:
                    await client.extract_concepts("Sample text")

                assert "Rate limited" in str(exc_info.value)


class TestAnthropicInterfaceCompliance:
    """Tests for LLMClient interface compliance."""

    def test_extraction_method_property(self):
        """Test extraction_method returns correct identifier."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient(model="haiku")
                assert client.extraction_method == "anthropic:haiku"

                client = AnthropicClient(model="sonnet")
                assert client.extraction_method == "anthropic:sonnet"

    @pytest.mark.asyncio
    async def test_get_model_info(self):
        """Test get_model_info returns configuration."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient(model="haiku", temperature=0.2)
                info = await client.get_model_info()

                assert info["model_name"] == "haiku"
                assert info["model_id"] == "claude-3-haiku-20240307"
                assert info["temperature"] == 0.2
                assert info["backend"] == "anthropic"

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        """Test close() completes without error."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from research_kb_extraction.anthropic_client import AnthropicClient

                client = AnthropicClient()
                await client.close()  # Should not raise


class TestAnthropicContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Test async context manager works correctly."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from research_kb_extraction.anthropic_client import AnthropicClient

                async with AnthropicClient() as client:
                    assert client is not None
                    assert client.model_name == "haiku"
