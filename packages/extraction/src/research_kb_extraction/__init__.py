"""Research KB Extraction - Concept extraction using LLM backends.

This package provides:
- LLMClient: Abstract base for LLM backends
- OllamaClient: GPU-accelerated local LLM wrapper (via Ollama server)
- InstructorOllamaClient: Ollama with instructor for schema validation + retry
- LlamaCppClient: Direct llama.cpp inference (for headless/high-VRAM setups)
- AnthropicClient: Claude API for fast, high-quality extraction
- BatchClient: Anthropic Message Batches API for 50% cost savings
- ConceptExtractor: Extract concepts and relationships from text chunks
- Deduplicator: Canonical name normalization and embedding-based deduplication
- get_llm_client: Factory function for backend selection
- get_batch_client: Factory for batch extraction
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
from research_kb_extraction.metrics import ExtractionMetrics
from research_kb_extraction.domain_prompts import (
    DOMAIN_PROMPTS,
    get_domain_prompt_section,
    get_domain_abbreviations,
    get_domain_config,
    list_domains,
    get_all_abbreviations,
)


def get_llm_client(
    backend: str = "ollama",
    model: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Factory function to create LLM client.

    Args:
        backend: Backend type ("ollama", "instructor", "llamacpp", or "anthropic")
        model: Model name or path (default depends on backend)
        **kwargs: Additional arguments passed to client constructor

    Returns:
        LLMClient instance for the specified backend

    Raises:
        ValueError: If backend is unknown
        ImportError: If required package not installed

    Example:
        >>> # Local Ollama inference (via server)
        >>> client = get_llm_client("ollama", model="llama3.1:8b")

        >>> # Ollama with instructor (schema validation + retry) [RECOMMENDED]
        >>> client = get_llm_client("instructor", model="llama3.1:8b")

        >>> # Direct llama.cpp inference (for headless/high-VRAM setups)
        >>> client = get_llm_client("llamacpp", model="models/llama-3.1-8b.gguf")

        >>> # Anthropic API (fast, high quality)
        >>> client = get_llm_client("anthropic", model="haiku")
    """
    if backend == "anthropic":
        # Import here to avoid requiring anthropic when using Ollama
        from research_kb_extraction.anthropic_client import AnthropicClient

        return AnthropicClient(
            model=model or "haiku-3.5",  # haiku-3.5 follows tool schema better than haiku
            **kwargs,
        )
    elif backend == "instructor":
        # Import here to avoid requiring instructor when using raw Ollama
        from research_kb_extraction.instructor_client import InstructorOllamaClient

        return InstructorOllamaClient(
            model=model or "llama3.1:8b",
            **kwargs,
        )
    elif backend == "llamacpp":
        # Import here to avoid requiring llama-cpp-python when using Ollama
        from research_kb_extraction.llama_cpp_client import LlamaCppClient

        return LlamaCppClient(
            model_path=model,  # None = use default model path
            **kwargs,
        )
    elif backend == "ollama":
        return OllamaClient(
            model=model or "llama3.1:8b",
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported: 'ollama', 'instructor', 'llamacpp', 'anthropic'"
        )


def get_batch_client(
    model: str = "haiku-4.5",
    **kwargs,
):
    """Factory function to create Batch API client.

    Args:
        model: Model name (haiku, haiku-3.5, haiku-4.5, sonnet)
        **kwargs: Additional arguments (max_retries, output_dir)

    Returns:
        BatchClient instance

    Example:
        >>> client = get_batch_client("haiku-4.5")
        >>> batch_id = await client.submit_batch(chunks)
        >>> results = await client.wait_for_results(batch_id)
    """
    from research_kb_extraction.batch_client import BatchClient

    return BatchClient(model=model, **kwargs)


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
    # Note: LlamaCppClient and AnthropicClient not exported at module level
    # to avoid requiring llama-cpp-python/anthropic packages.
    # Use get_llm_client("llamacpp") or get_llm_client("anthropic") instead.
    # Extraction
    "ConceptExtractor",
    "Deduplicator",
    "ABBREVIATION_MAP",
    # Domain Prompts
    "DOMAIN_PROMPTS",
    "get_domain_prompt_section",
    "get_domain_abbreviations",
    "get_domain_config",
    "list_domains",
    "get_all_abbreviations",
    # Metrics
    "ExtractionMetrics",
    # Factory
    "get_llm_client",
    "get_batch_client",
]
