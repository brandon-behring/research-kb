"""Research KB Extraction - Concept extraction using LLM backends.

This package provides:
- LLMClient: Abstract base for LLM backends
- OllamaClient: GPU-accelerated local LLM wrapper
- AnthropicClient: Claude API for fast, high-quality extraction
- ConceptExtractor: Extract concepts and relationships from text chunks
- Deduplicator: Canonical name normalization and embedding-based deduplication
- GraphSyncService: Sync concepts to Neo4j graph database
- get_llm_client: Factory function for backend selection
"""

from typing import Optional

from research_kb_extraction.models import (
    ChunkExtraction,
    ConceptMatch,
    ExtractedConcept,
    ExtractedRelationship,
    StoredConcept,
)
from research_kb_extraction.base_client import LLMClient
from research_kb_extraction.ollama_client import OllamaClient, OllamaError
from research_kb_extraction.concept_extractor import ConceptExtractor
from research_kb_extraction.deduplicator import Deduplicator, ABBREVIATION_MAP
from research_kb_extraction.graph_sync import GraphSyncService, GraphSyncError


def get_llm_client(
    backend: str = "ollama",
    model: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Factory function to create LLM client.

    Args:
        backend: Backend type ("ollama" or "anthropic")
        model: Model name (default depends on backend)
        **kwargs: Additional arguments passed to client constructor

    Returns:
        LLMClient instance for the specified backend

    Raises:
        ValueError: If backend is unknown
        ImportError: If required package not installed (e.g., anthropic)

    Example:
        >>> # Local Ollama inference
        >>> client = get_llm_client("ollama", model="llama3.1:8b")

        >>> # Anthropic API (fast, high quality)
        >>> client = get_llm_client("anthropic", model="haiku")

        >>> # Anthropic Opus for production quality
        >>> client = get_llm_client("anthropic", model="opus")
    """
    if backend == "anthropic":
        # Import here to avoid requiring anthropic when using Ollama
        from research_kb_extraction.anthropic_client import AnthropicClient

        return AnthropicClient(
            model=model or "haiku",
            **kwargs,
        )
    elif backend == "ollama":
        return OllamaClient(
            model=model or "llama3.1:8b",
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported: 'ollama', 'anthropic'"
        )


__all__ = [
    # Models
    "ExtractedConcept",
    "ExtractedRelationship",
    "ChunkExtraction",
    "ConceptMatch",
    "StoredConcept",
    # Base class
    "LLMClient",
    # Clients
    "OllamaClient",
    "OllamaError",
    # Note: AnthropicClient not exported at module level to avoid
    # requiring anthropic package. Use get_llm_client("anthropic") instead.
    # Extraction
    "ConceptExtractor",
    "Deduplicator",
    "ABBREVIATION_MAP",
    "GraphSyncService",
    "GraphSyncError",
    # Factory
    "get_llm_client",
]
