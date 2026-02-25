"""Abstract base class for LLM clients.

Provides a common interface for different LLM backends (Ollama, Anthropic, etc.)
used for concept extraction from academic text.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from research_kb_extraction.models import ChunkExtraction


class LLMClient(ABC):
    """Abstract base for LLM clients.

    All LLM backends must implement this interface for use in the
    concept extraction pipeline. This enables swapping between:
    - OllamaClient: Local GPU inference
    - AnthropicClient: Claude API for speed/quality
    - (Future) LlamaCppClient: Direct llama.cpp inference
    """

    @abstractmethod
    async def extract_concepts(
        self,
        chunk: str,
        domain_id: str,
        prompt_type: str = "full",
    ) -> ChunkExtraction:
        """Extract structured concepts from a text chunk.

        Args:
            chunk: Text chunk to analyze
            prompt_type: Prompt variant ("full", "definition", "relationship", "quick")
            domain_id: Knowledge domain for extraction (e.g., "causal_inference", "time_series")

        Returns:
            ChunkExtraction with concepts and relationships
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the backend is available and ready.

        For Ollama: checks server connectivity
        For Anthropic: checks API key is set
        For LlamaCpp: checks model is loaded
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (HTTP clients, model handles, etc.)."""
        pass

    @property
    @abstractmethod
    def extraction_method(self) -> str:
        """Return identifier for database metadata.

        Examples:
            - "ollama:llama3.1:8b"
            - "anthropic:haiku"
            - "anthropic:opus"
            - "llamacpp:Meta-Llama-3.1-8B-Q4_K_M"
        """
        pass

    async def extract_batch(
        self,
        chunks: list[str],
        domain_id: str,
        prompt_type: str = "full",
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[ChunkExtraction]:
        """Extract concepts from multiple chunks.

        Default implementation processes serially. Backends may override
        for parallel/batch processing (e.g., Anthropic Message Batches).

        Args:
            chunks: List of text chunks
            prompt_type: Prompt type for all extractions
            domain_id: Knowledge domain for extraction
            on_progress: Optional callback(index, total) for progress

        Returns:
            List of ChunkExtraction results
        """
        results = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            result = await self.extract_concepts(chunk, domain_id, prompt_type)
            results.append(result)

            if on_progress:
                on_progress(i + 1, total)

        return results

    async def __aenter__(self) -> "LLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
