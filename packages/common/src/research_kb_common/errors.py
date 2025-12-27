"""Custom error types for research-kb system.

All errors follow the "fail fast" principle with explicit messages.
"""


class ResearchKBError(Exception):
    """Base exception for all research-kb errors."""

    pass


class IngestionError(ResearchKBError):
    """Error during document ingestion pipeline."""

    pass


class ChunkExtractionError(IngestionError):
    """Error extracting chunks from source document."""

    pass


class EmbeddingError(IngestionError):
    """Error generating embeddings for chunks."""

    pass


class StorageError(ResearchKBError):
    """Error during database operations."""

    pass


class SearchError(ResearchKBError):
    """Error during search operations (FTS or vector)."""

    pass


class ExtractionError(ResearchKBError):
    """Error during concept extraction from chunks."""

    pass


class ExtractionValidationError(ExtractionError):
    """Error validating extraction output (malformed JSON, invalid schema).

    This error indicates the LLM returned output that could not be parsed.
    The chunk should be retried or sent to a dead-letter queue.
    """

    pass
