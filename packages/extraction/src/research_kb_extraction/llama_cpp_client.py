"""LlamaCpp client for direct GPU inference without HTTP overhead.

Provides ~40% speedup over Ollama by eliminating server overhead.
Uses grammar-based JSON output for structured extraction.
"""

import json
from pathlib import Path
from typing import Any, Optional

from research_kb_common import get_logger

from research_kb_extraction.base_client import LLMClient
from research_kb_extraction.models import ChunkExtraction
from research_kb_extraction.prompts import SYSTEM_PROMPT, format_extraction_prompt

logger = get_logger(__name__)

# Default model path (research-kb/models/)
# Path: llama_cpp_client.py -> research_kb_extraction -> src -> extraction -> packages -> research-kb
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent.parent.parent / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# JSON schema for structured output (grammar-based sampling)
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "concept_type": {
                        "type": "string",
                        "enum": ["method", "assumption", "problem", "definition", "theorem"]
                    },
                    "definition": {"type": ["string", "null"]},
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["name", "concept_type"]
            }
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_concept": {"type": "string"},
                    "target_concept": {"type": "string"},
                    "relationship_type": {
                        "type": "string",
                        "enum": ["REQUIRES", "USES", "ADDRESSES", "GENERALIZES", "SPECIALIZES", "ALTERNATIVE_TO", "EXTENDS"]
                    },
                    "evidence": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["source_concept", "target_concept", "relationship_type"]
            }
        }
    },
    "required": ["concepts", "relationships"]
}


class LlamaCppError(Exception):
    """Error from llama.cpp inference."""
    pass


class LlamaCppClient(LLMClient):
    """Client for direct llama.cpp inference without HTTP overhead.

    Uses llama-cpp-python for GPU-accelerated inference with grammar-based
    JSON output. Provides ~40% speedup over Ollama by eliminating server
    overhead and HTTP round-trips.

    Example:
        >>> client = LlamaCppClient(model_path="models/llama-3.1-8b-q4.gguf")
        >>> result = await client.extract_concepts("The backdoor criterion...")
        >>> print(result.concepts)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = 20,  # 20 layers fits in 8GB VRAM with desktop overhead
        temperature: float = 0.1,
        max_tokens: int = 4096,  # Increased for long JSON responses
        n_batch: int = 512,
        verbose: bool = False,
    ):
        """Initialize llama.cpp client.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (tokens)
            n_gpu_layers: Layers to offload to GPU (-1 = all)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            n_batch: Batch size for prompt processing
            verbose: Enable llama.cpp verbose output
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_batch = n_batch
        self.verbose = verbose
        self._llm = None
        self._model_name = self.model_path.stem if self.model_path else "unknown"

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._llm is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise LlamaCppError(
                "llama-cpp-python not installed. Run:\n"
                "  sudo ./scripts/install_llama_cpp_cuda.sh\n"
                "  ./scripts/build_llama_cpp_cuda.sh"
            )

        if not self.model_path.exists():
            raise LlamaCppError(
                f"Model not found: {self.model_path}\n"
                "Run: ./scripts/download_gguf_model.sh"
            )

        logger.info(
            "loading_llama_model",
            model=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
        )

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            verbose=self.verbose,
        )

        logger.info("llama_model_loaded", model=self._model_name)

    async def is_available(self) -> bool:
        """Check if model file exists and can be loaded."""
        try:
            if not self.model_path.exists():
                return False
            # Try importing llama_cpp
            import llama_cpp
            return True
        except ImportError:
            return False

    async def close(self) -> None:
        """Release model resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("llama_model_unloaded")

    @property
    def extraction_method(self) -> str:
        """Return identifier for database metadata."""
        return f"llamacpp:{self._model_name}"

    def _format_chat_prompt(self, system: str, user: str) -> str:
        """Format prompt for Llama 3.1 Instruct chat template.

        Uses the official Llama 3.1 chat format with special tokens.
        """
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = True,
    ) -> str:
        """Generate text completion using chat API.

        Args:
            prompt: User prompt
            system: System prompt
            json_mode: If True, use JSON schema for structured output

        Returns:
            Generated text
        """
        self._load_model()

        # Build chat messages
        messages = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            if json_mode:
                # Use create_chat_completion with response_format for JSON
                output = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={
                        "type": "json_object",
                        "schema": EXTRACTION_SCHEMA,
                    },
                )
            else:
                output = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

            return output["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error("llama_generation_error", error=str(e))
            raise LlamaCppError(f"Generation failed: {e}") from e

    async def extract_concepts(
        self,
        chunk: str,
        prompt_type: str = "full",
    ) -> ChunkExtraction:
        """Extract concepts and relationships from a text chunk.

        Args:
            chunk: Text chunk to analyze
            prompt_type: Prompt type ("full", "definition", "relationship", "quick")

        Returns:
            ChunkExtraction with concepts and relationships
        """
        prompt = format_extraction_prompt(chunk, prompt_type)

        logger.debug(
            "extracting_concepts_llamacpp",
            chunk_length=len(chunk),
            prompt_type=prompt_type,
        )

        # Note: llama-cpp-python is synchronous, but we wrap in async interface
        response = self.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            json_mode=True,
        )

        try:
            data = json.loads(response)
            extraction = ChunkExtraction.model_validate(data)

            logger.info(
                "extraction_complete_llamacpp",
                concepts=extraction.concept_count,
                relationships=extraction.relationship_count,
            )

            return extraction

        except json.JSONDecodeError as e:
            logger.error("json_parse_error", response=response[:200], error=str(e))
            return ChunkExtraction()

        except Exception as e:
            logger.error("extraction_validation_error", error=str(e))
            return ChunkExtraction()

    async def __aenter__(self) -> "LlamaCppClient":
        """Async context manager entry - preload model."""
        self._load_model()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
