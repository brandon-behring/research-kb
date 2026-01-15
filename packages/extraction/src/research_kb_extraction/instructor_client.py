"""Instructor-based Ollama client for structured extraction.

Uses the instructor library for automatic Pydantic validation and retry
on validation failures. This provides more reliable structured output
than raw JSON mode.

Benefits over raw OllamaClient:
1. Automatic retry on validation failure (up to 3 attempts)
2. Pydantic schema enforcement at extraction time
3. Cleaner error handling
4. Future-proof schema handling
"""

import instructor
from typing import Optional

from research_kb_common import get_logger

from research_kb_extraction.base_client import LLMClient
from research_kb_extraction.models import ChunkExtraction
from research_kb_extraction.prompts import SYSTEM_PROMPT, format_extraction_prompt

logger = get_logger(__name__)


class InstructorOllamaClient(LLMClient):
    """Ollama client using instructor for structured output.

    Uses instructor's from_provider to get automatic Pydantic validation
    and retry logic for schema violations.

    Example:
        >>> client = InstructorOllamaClient(model="llama3.1:8b")
        >>> result = await client.extract_concepts("The backdoor criterion...")
        >>> print(result.concepts)  # Guaranteed to be valid ChunkExtraction
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        """Initialize instructor-based Ollama client.

        Args:
            model: Ollama model name
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            temperature: Sampling temperature
            max_retries: Number of retries on validation failure
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries

        # Create instructor client with Ollama provider
        # Ollama's OpenAI-compatible endpoint is at /v1
        # Use JSON mode explicitly (TOOLS mode has issues with nested arrays)
        openai_base_url = f"{base_url}/v1"
        self._client = instructor.from_provider(
            f"ollama/{model}",
            base_url=openai_base_url,
            timeout=timeout,
            mode=instructor.Mode.JSON,
        )
        self._async_client = instructor.from_provider(
            f"ollama/{model}",
            base_url=openai_base_url,
            timeout=timeout,
            mode=instructor.Mode.JSON,
            async_client=True,
        )

    async def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            import httpx
            async with httpx.AsyncClient(base_url=self.base_url) as client:
                response = await client.get("/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Clean up resources."""
        # Instructor clients don't need explicit cleanup
        pass

    @property
    def extraction_method(self) -> str:
        """Return identifier for database metadata."""
        return f"instructor:{self.model}"

    async def extract_concepts(
        self,
        chunk: str,
        prompt_type: str = "full",
        domain_id: str = "causal_inference",
    ) -> ChunkExtraction:
        """Extract concepts using instructor for validation.

        Args:
            chunk: Text chunk to analyze
            prompt_type: Prompt type ("full", "definition", "relationship", "quick")
            domain_id: Knowledge domain (e.g., "causal_inference", "time_series")

        Returns:
            ChunkExtraction with validated concepts and relationships
        """
        prompt = format_extraction_prompt(chunk, prompt_type, domain_id)

        logger.debug(
            "extracting_concepts_instructor",
            chunk_length=len(chunk),
            prompt_type=prompt_type,
            domain_id=domain_id,
        )

        try:
            # Use instructor's create with response_model for validation
            extraction = await self._async_client.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_model=ChunkExtraction,
                max_retries=self.max_retries,
                temperature=self.temperature,
            )

            logger.info(
                "extraction_complete_instructor",
                concepts=extraction.concept_count,
                relationships=extraction.relationship_count,
            )

            return extraction

        except Exception as e:
            logger.error("instructor_extraction_failed", error=str(e))
            # Return empty extraction on failure
            return ChunkExtraction()

    async def __aenter__(self) -> "InstructorOllamaClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
