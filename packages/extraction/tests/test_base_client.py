"""Tests for LLMClient abstract base class.

Tests:
- Abstract method requirements
- Default extract_batch implementation
- Async context manager protocol
- Concrete implementation compliance
"""

import pytest
from abc import ABC

from research_kb_extraction.base_client import LLMClient
from research_kb_extraction.models import ChunkExtraction, ExtractedConcept

pytestmark = [pytest.mark.requires_ollama, pytest.mark.unit]


class TestLLMClientInterface:
    """Tests for LLMClient abstract interface."""

    def test_is_abstract_base_class(self):
        """Test LLMClient is an ABC and cannot be instantiated."""
        assert issubclass(LLMClient, ABC)

        with pytest.raises(TypeError, match="abstract"):
            LLMClient()

    def test_abstract_methods_defined(self):
        """Test all required abstract methods are defined."""
        abstract_methods = LLMClient.__abstractmethods__

        assert "extract_concepts" in abstract_methods
        assert "is_available" in abstract_methods
        assert "close" in abstract_methods
        assert "extraction_method" in abstract_methods

    def test_extraction_method_is_property(self):
        """Test extraction_method is an abstract property."""
        # In Python's ABC, abstract properties are included in __abstractmethods__
        assert "extraction_method" in LLMClient.__abstractmethods__


class ConcreteClient(LLMClient):
    """Minimal concrete implementation for testing."""

    def __init__(self, model_name: str = "test:model"):
        self._model_name = model_name
        self._closed = False
        self._available = True
        self._extract_result = ChunkExtraction(concepts=[], relationships=[])

    async def extract_concepts(
        self, chunk: str, domain_id: str = "causal_inference", prompt_type: str = "full"
    ) -> ChunkExtraction:
        """Extract concepts from text."""
        return self._extract_result

    async def is_available(self) -> bool:
        """Check if backend is available."""
        return self._available

    async def close(self) -> None:
        """Close the client."""
        self._closed = True

    @property
    def extraction_method(self) -> str:
        """Return extraction method identifier."""
        return self._model_name


class TestConcreteImplementation:
    """Tests for concrete implementation of LLMClient."""

    def test_can_instantiate_concrete_class(self):
        """Test concrete implementation can be instantiated."""
        client = ConcreteClient()

        assert client is not None
        assert isinstance(client, LLMClient)

    def test_extraction_method_property(self):
        """Test extraction_method property returns correct value."""
        client = ConcreteClient(model_name="ollama:llama3.1:8b")

        assert client.extraction_method == "ollama:llama3.1:8b"

    async def test_extract_concepts_returns_chunk_extraction(self):
        """Test extract_concepts returns ChunkExtraction."""
        client = ConcreteClient()

        result = await client.extract_concepts("Test chunk text", domain_id="causal_inference")

        assert isinstance(result, ChunkExtraction)

    async def test_is_available_returns_bool(self):
        """Test is_available returns boolean."""
        client = ConcreteClient()

        result = await client.is_available()

        assert isinstance(result, bool)
        assert result is True

    async def test_close_is_callable(self):
        """Test close method can be called."""
        client = ConcreteClient()

        await client.close()

        assert client._closed is True


class TestExtractBatch:
    """Tests for default extract_batch implementation."""

    async def test_extract_batch_processes_all_chunks(self):
        """Test extract_batch processes all chunks."""
        client = ConcreteClient()
        client._extract_result = ChunkExtraction(
            concepts=[ExtractedConcept(name="test", concept_type="method", confidence=0.9)],
            relationships=[],
        )

        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        results = await client.extract_batch(chunks, domain_id="causal_inference")

        assert len(results) == 3
        assert all(isinstance(r, ChunkExtraction) for r in results)

    async def test_extract_batch_empty_list(self):
        """Test extract_batch with empty chunk list."""
        client = ConcreteClient()

        results = await client.extract_batch([], domain_id="causal_inference")

        assert results == []

    async def test_extract_batch_calls_extract_concepts(self):
        """Test extract_batch calls extract_concepts for each chunk."""
        client = ConcreteClient()
        call_count = 0
        original_extract = client.extract_concepts

        async def counting_extract(chunk, domain_id="causal_inference", prompt_type="full"):
            nonlocal call_count
            call_count += 1
            return await original_extract(chunk, domain_id, prompt_type)

        client.extract_concepts = counting_extract

        chunks = ["chunk 1", "chunk 2"]
        await client.extract_batch(chunks, domain_id="causal_inference")

        assert call_count == 2

    async def test_extract_batch_respects_prompt_type(self):
        """Test extract_batch passes prompt_type to extract_concepts."""
        captured_prompt_types = []

        class CapturingClient(ConcreteClient):
            async def extract_concepts(
                self,
                chunk: str,
                domain_id: str = "causal_inference",
                prompt_type: str = "full",
            ):
                captured_prompt_types.append(prompt_type)
                return ChunkExtraction(concepts=[], relationships=[])

        client = CapturingClient()

        await client.extract_batch(
            ["chunk 1", "chunk 2"], domain_id="causal_inference", prompt_type="quick"
        )

        assert all(pt == "quick" for pt in captured_prompt_types)

    async def test_extract_batch_calls_progress_callback(self):
        """Test extract_batch calls progress callback."""
        client = ConcreteClient()
        progress_calls = []

        def on_progress(index, total):
            progress_calls.append((index, total))

        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        await client.extract_batch(chunks, domain_id="causal_inference", on_progress=on_progress)

        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3)
        assert progress_calls[1] == (2, 3)
        assert progress_calls[2] == (3, 3)

    async def test_extract_batch_no_progress_callback(self):
        """Test extract_batch works without progress callback."""
        client = ConcreteClient()

        # Should not raise
        results = await client.extract_batch(["chunk 1", "chunk 2"], domain_id="causal_inference")

        assert len(results) == 2


class TestAsyncContextManager:
    """Tests for async context manager protocol."""

    async def test_aenter_returns_self(self):
        """Test __aenter__ returns the client instance."""
        client = ConcreteClient()

        result = await client.__aenter__()

        assert result is client

    async def test_aexit_calls_close(self):
        """Test __aexit__ calls close method."""
        client = ConcreteClient()
        assert client._closed is False

        await client.__aexit__(None, None, None)

        assert client._closed is True

    async def test_async_with_syntax(self):
        """Test client works with async with syntax."""
        async with ConcreteClient() as client:
            assert isinstance(client, LLMClient)
            result = await client.is_available()
            assert result is True

        assert client._closed is True

    async def test_context_manager_closes_on_exception(self):
        """Test close is called even when exception occurs."""
        client = ConcreteClient()

        with pytest.raises(ValueError):
            async with client:
                raise ValueError("Test error")

        assert client._closed is True


class TestProtocolCompliance:
    """Tests for protocol compliance of existing clients."""

    def test_ollama_client_implements_interface(self):
        """Test OllamaClient implements LLMClient interface."""
        from research_kb_extraction.ollama_client import OllamaClient

        client = OllamaClient()

        assert isinstance(client, LLMClient)
        assert hasattr(client, "extract_concepts")
        assert hasattr(client, "is_available")
        assert hasattr(client, "close")
        assert hasattr(client, "extraction_method")

    def test_anthropic_client_implements_interface(self):
        """Test AnthropicClient implements LLMClient interface."""
        try:
            from research_kb_extraction.anthropic_client import AnthropicClient

            # AnthropicClient requires API key, so just check class structure
            assert issubclass(AnthropicClient, LLMClient)
            assert hasattr(AnthropicClient, "extract_concepts")
            assert hasattr(AnthropicClient, "is_available")
            assert hasattr(AnthropicClient, "close")
            assert hasattr(AnthropicClient, "extraction_method")
        except ImportError:
            pytest.skip("AnthropicClient not available")

    def test_llama_cpp_client_implements_interface(self):
        """Test LlamaCppClient implements LLMClient interface."""
        try:
            from research_kb_extraction.llama_cpp_client import LlamaCppClient

            assert issubclass(LlamaCppClient, LLMClient)
            assert hasattr(LlamaCppClient, "extract_concepts")
            assert hasattr(LlamaCppClient, "is_available")
            assert hasattr(LlamaCppClient, "close")
            assert hasattr(LlamaCppClient, "extraction_method")
        except ImportError:
            pytest.skip("LlamaCppClient not available")


class TestExtractionMethodNaming:
    """Tests for extraction_method naming conventions."""

    def test_extraction_method_format_ollama(self):
        """Test Ollama extraction method format."""
        client = ConcreteClient(model_name="ollama:llama3.1:8b")

        assert ":" in client.extraction_method
        assert "ollama" in client.extraction_method.lower()

    def test_extraction_method_format_anthropic(self):
        """Test Anthropic extraction method format."""
        client = ConcreteClient(model_name="anthropic:haiku")

        assert ":" in client.extraction_method
        assert "anthropic" in client.extraction_method.lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    async def test_extract_batch_with_none_progress(self):
        """Test extract_batch handles None progress callback."""
        client = ConcreteClient()

        # Should not raise
        results = await client.extract_batch(
            ["chunk"], domain_id="causal_inference", on_progress=None
        )

        assert len(results) == 1

    async def test_extract_concepts_with_empty_chunk(self):
        """Test extract_concepts handles empty string."""
        client = ConcreteClient()

        result = await client.extract_concepts("", domain_id="causal_inference")

        assert isinstance(result, ChunkExtraction)

    async def test_multiple_closes_are_safe(self):
        """Test calling close multiple times is safe."""
        client = ConcreteClient()

        await client.close()
        await client.close()  # Should not raise

        assert client._closed is True


class TestPromptTypes:
    """Tests for prompt type parameter."""

    async def test_default_prompt_type(self):
        """Test default prompt type is 'full'."""
        captured_prompt_type = None

        class CapturingClient(ConcreteClient):
            async def extract_concepts(
                self,
                chunk: str,
                prompt_type: str = "full",
                domain_id: str = "causal_inference",
            ):
                nonlocal captured_prompt_type
                captured_prompt_type = prompt_type
                return ChunkExtraction(concepts=[], relationships=[])

        client = CapturingClient()
        await client.extract_concepts("test chunk", domain_id="causal_inference")

        assert captured_prompt_type == "full"

    async def test_prompt_types_supported(self):
        """Test various prompt types are passed through."""
        for prompt_type in ["full", "definition", "relationship", "quick"]:
            captured = None

            class CapturingClient(ConcreteClient):
                async def extract_concepts(
                    self,
                    chunk: str,
                    prompt_type: str = "full",
                    domain_id: str = "causal_inference",
                ):
                    nonlocal captured
                    captured = prompt_type
                    return ChunkExtraction(concepts=[], relationships=[])

            client = CapturingClient()
            await client.extract_concepts(
                "test", domain_id="causal_inference", prompt_type=prompt_type
            )

            assert captured == prompt_type
