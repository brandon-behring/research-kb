"""Anthropic Claude client for concept extraction.

Provides fast, high-quality concept extraction using Claude API.
Supports multiple Claude models with different speed/quality tradeoffs:
- haiku: Fast, cheap (~$0.50/18 papers)
- sonnet: Balanced
- opus: Highest quality (~$40/72 papers)
"""

import asyncio
import os
from typing import Optional

from research_kb_common import get_logger

from research_kb_extraction.base_client import LLMClient
from research_kb_extraction.models import ChunkExtraction
from research_kb_extraction.prompts import SYSTEM_PROMPT, format_extraction_prompt

logger = get_logger(__name__)

# Tool definition for structured concept extraction
# This enforces the exact enum values via JSON schema
EXTRACTION_TOOL = {
    "name": "extract_concepts",
    "description": "Extract concepts and relationships from academic text",
    "input_schema": {
        "type": "object",
        "properties": {
            "concepts": {
                "type": "array",
                "description": "Concepts found in the text",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Concept name as it appears in text",
                        },
                        "concept_type": {
                            "type": "string",
                            "enum": [
                                "method",
                                "assumption",
                                "problem",
                                "definition",
                                "theorem",
                                "concept",
                                "principle",
                                "technique",
                                "model",
                            ],
                            "description": "Classification of the concept (method ≤35% of extractions)",
                        },
                        "definition": {
                            "type": "string",
                            "description": "Brief definition if provided in the text",
                        },
                        "aliases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Alternative names or abbreviations",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in extraction (0.0-1.0)",
                        },
                    },
                    "required": ["name", "concept_type"],
                },
            },
            "relationships": {
                "type": "array",
                "description": "Relationships between concepts",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_concept": {
                            "type": "string",
                            "description": "Source concept name",
                        },
                        "target_concept": {
                            "type": "string",
                            "description": "Target concept name",
                        },
                        "relationship_type": {
                            "type": "string",
                            "enum": [
                                "REQUIRES",
                                "USES",
                                "ADDRESSES",
                                "GENERALIZES",
                                "SPECIALIZES",
                                "ALTERNATIVE_TO",
                                "EXTENDS",
                            ],
                            "description": "Relationship type from ontology",
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Text snippet supporting this relationship",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in relationship",
                        },
                    },
                    "required": [
                        "source_concept",
                        "target_concept",
                        "relationship_type",
                    ],
                },
            },
        },
        "required": ["concepts", "relationships"],
    },
}


class AnthropicError(Exception):
    """Error from Anthropic API."""

    pass


class AnthropicClient(LLMClient):
    """Anthropic Claude API client for fast, high-quality extraction.

    Uses Claude's structured output capabilities for reliable JSON extraction.
    Significantly faster than local inference (~1-3 sec/chunk vs ~60-90 sec).

    Example:
        >>> client = AnthropicClient(model="haiku")
        >>> result = await client.extract_concepts("The backdoor criterion...")
        >>> print(result.concepts)
    """

    # Model name -> API model ID mapping
    MODELS = {
        "haiku": "claude-3-haiku-20240307",
        "haiku-3.5": "claude-3-5-haiku-20241022",
        "haiku-4.5": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-20250514",
        "opus": "claude-opus-4-5-20251101",
    }

    def __init__(
        self,
        model: str = "haiku",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        """Initialize Anthropic client.

        Args:
            model: Model name (haiku, haiku-3.5, sonnet, opus) or full model ID
            api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env)
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model
        self.model_id = self.MODELS.get(model, model)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Import here to avoid requiring anthropic when using other backends
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        raw_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._api_key = raw_key.strip() if raw_key else None
        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set and no api_key provided"
            )

        self._client = anthropic.Anthropic(api_key=self._api_key)
        logger.info(
            "anthropic_client_initialized",
            model=self.model_name,
            model_id=self.model_id,
        )

    def _call_api(self, prompt: str) -> dict:
        """Synchronous API call using tool_use for schema enforcement.

        Args:
            prompt: User prompt text

        Returns:
            Parsed tool input as dict (already validated by Claude)

        Raises:
            AnthropicError: If API call fails
        """
        try:
            message = self._client.messages.create(  # type: ignore[call-overload]  # Anthropic SDK overload generics
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                tools=[EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": "extract_concepts"},
            )

            # Extract tool use from response
            for block in message.content:
                if block.type == "tool_use" and block.name == "extract_concepts":
                    return block.input

            # Fallback: no tool use in response
            logger.warning("no_tool_use_in_response", model=self.model_id)
            return {"concepts": [], "relationships": []}

        except Exception as e:
            logger.error("anthropic_api_error", error=str(e), model=self.model_id)
            raise AnthropicError(f"Anthropic API error: {e}") from e

    async def extract_concepts(
        self,
        chunk: str,
        domain_id: str,
        prompt_type: str = "full",
    ) -> ChunkExtraction:
        """Extract concepts using Claude.

        Args:
            chunk: Text chunk to analyze
            prompt_type: Prompt variant ("full", "definition", "relationship", "quick")
            domain_id: Knowledge domain (e.g., "causal_inference", "time_series")

        Returns:
            ChunkExtraction with concepts and relationships

        Raises:
            AnthropicError: If extraction fails
        """
        prompt = format_extraction_prompt(chunk, domain_id, prompt_type)

        logger.debug(
            "extracting_concepts",
            chunk_length=len(chunk),
            prompt_type=prompt_type,
            domain_id=domain_id,
            model=self.model_name,
        )

        # Run sync API call in thread pool for async compatibility
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._call_api, prompt)

        try:
            # Tool use returns dict directly - schema already enforced by Claude
            extraction = ChunkExtraction.model_validate(data)

            logger.info(
                "extraction_complete",
                concepts=extraction.concept_count,
                relationships=extraction.relationship_count,
                model=self.model_name,
            )

            return extraction

        except Exception as e:
            logger.error(
                "extraction_validation_error",
                error=str(e),
                data=str(data)[:500] if data else None,
                model=self.model_name,
            )
            return ChunkExtraction()

    async def is_available(self) -> bool:
        """Check if API key is set and valid.

        Performs a lightweight API call to verify credentials.
        """
        if not self._api_key:
            return False

        try:
            # Use messages.count_tokens for a cheap availability check
            # This is faster than a full completion request
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._client.messages.count_tokens(
                    model=self.model_id,
                    messages=[{"role": "user", "content": "test"}],
                ),
            )
            return True
        except Exception as e:
            logger.warning("anthropic_availability_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """No cleanup needed for API client."""
        pass

    @property
    def extraction_method(self) -> str:
        """Return extraction method identifier for database metadata."""
        return f"anthropic:{self.model_name}"

    async def get_model_info(self) -> dict:
        """Get information about the configured model."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "backend": "anthropic",
        }
