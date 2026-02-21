"""Ollama client for GPU-accelerated LLM inference.

Provides structured JSON output using Ollama's native JSON mode.
Designed for concept extraction with llama3.1:8b model.
"""

import json
from typing import Any, Optional

import httpx
from research_kb_common import ExtractionValidationError, get_logger

from research_kb_extraction.base_client import LLMClient
from research_kb_extraction.models import ChunkExtraction
from research_kb_extraction.prompts import SYSTEM_PROMPT, format_extraction_prompt

logger = get_logger(__name__)


class OllamaError(Exception):
    """Error from Ollama API."""

    pass


class OllamaClient(LLMClient):
    """Client for Ollama LLM with structured JSON output.

    Uses Ollama's native JSON mode for reliable structured extraction.
    Designed for GPU-accelerated inference (8GB+ VRAM recommended).

    Example:
        >>> client = OllamaClient(model="llama3.1:8b")
        >>> result = await client.extract_concepts("The backdoor criterion...")
        >>> print(result.concepts)
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        temperature: float = 0.1,
        num_ctx: int = 4096,
    ):
        """Initialize Ollama client.

        Args:
            model: Ollama model name (default: llama3.1:8b)
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            temperature: Sampling temperature (lower = more deterministic)
            num_ctx: Context window size in tokens
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.debug("ollama_unavailable", error=str(e))
            return False

    async def is_model_loaded(self) -> bool:
        """Check if the configured model is available."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            if response.status_code != 200:
                return False

            data = response.json()
            model_names = [m["name"] for m in data.get("models", [])]
            # Check both exact match and without tag
            return (
                self.model in model_names
                or f"{self.model}:latest" in model_names
                or self.model.split(":")[0] in [m.split(":")[0] for m in model_names]
            )
        except Exception as e:
            logger.debug("model_check_failed", model=self.model, error=str(e))
            return False

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = True,
    ) -> str:
        """Generate text completion from Ollama.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            json_mode: If True, request JSON output format

        Returns:
            Generated text response

        Raises:
            OllamaError: If generation fails
        """
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        if system:
            payload["system"] = system

        if json_mode:
            payload["format"] = "json"

        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()

            data = response.json()
            return data.get("response", "")

        except httpx.HTTPStatusError as e:
            logger.error("ollama_http_error", status=e.response.status_code)
            raise OllamaError(f"Ollama HTTP error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error("ollama_request_error", error=str(e))
            raise OllamaError(f"Ollama request failed: {e}") from e

    async def extract_concepts(
        self,
        chunk: str,
        domain_id: str,
        prompt_type: str = "full",
    ) -> ChunkExtraction:
        """Extract concepts and relationships from a text chunk.

        Args:
            chunk: Text chunk to analyze
            prompt_type: Prompt type ("full", "definition", "relationship", "quick")
            domain_id: Knowledge domain (e.g., "causal_inference", "time_series")

        Returns:
            ChunkExtraction with concepts and relationships

        Raises:
            OllamaError: If extraction fails
        """
        prompt = format_extraction_prompt(chunk, domain_id, prompt_type)

        logger.debug(
            "extracting_concepts",
            chunk_length=len(chunk),
            prompt_type=prompt_type,
            domain_id=domain_id,
        )

        response = await self.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            json_mode=True,
        )

        try:
            # Parse JSON response
            data = json.loads(response)
            extraction = ChunkExtraction.model_validate(data)

            logger.info(
                "extraction_complete",
                concepts=extraction.concept_count,
                relationships=extraction.relationship_count,
            )

            return extraction

        except json.JSONDecodeError as e:
            logger.error("json_parse_error", response=response[:200], error=str(e))
            raise ExtractionValidationError(f"Failed to parse JSON response: {e}") from e

        except Exception as e:
            logger.error("extraction_validation_error", error=str(e))
            raise ExtractionValidationError(f"Failed to validate extraction output: {e}") from e

    async def extract_batch(
        self,
        chunks: list[str],
        domain_id: str,
        prompt_type: str = "full",
        on_progress: Optional[callable] = None,
    ) -> list[ChunkExtraction]:
        """Extract concepts from multiple chunks.

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

    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/show",
                json={"name": self.model},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("model_info_error", error=str(e))
            return {}

    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def extraction_method(self) -> str:
        """Return extraction method identifier for database metadata."""
        return f"ollama:{self.model}"
