"""Tests for custom error types."""

import pytest

from research_kb_common.errors import (
    ChunkExtractionError,
    EmbeddingError,
    IngestionError,
    ResearchKBError,
    SearchError,
    StorageError,
)

pytestmark = pytest.mark.unit


class TestErrorHierarchy:
    """Test error inheritance and hierarchy."""

    def test_all_errors_inherit_from_base(self):
        """Verify all custom errors inherit from ResearchKBError."""
        assert issubclass(IngestionError, ResearchKBError)
        assert issubclass(ChunkExtractionError, ResearchKBError)
        assert issubclass(EmbeddingError, ResearchKBError)
        assert issubclass(StorageError, ResearchKBError)
        assert issubclass(SearchError, ResearchKBError)

    def test_ingestion_errors_inherit_from_ingestion_error(self):
        """Verify ingestion-related errors inherit from IngestionError."""
        assert issubclass(ChunkExtractionError, IngestionError)
        assert issubclass(EmbeddingError, IngestionError)

    def test_errors_can_be_raised_and_caught(self):
        """Test errors can be raised and caught correctly."""
        with pytest.raises(ResearchKBError):
            raise IngestionError("Test error")

        with pytest.raises(IngestionError):
            raise ChunkExtractionError("Extraction failed")

        with pytest.raises(IngestionError):
            raise EmbeddingError("Embedding generation failed")

    def test_error_messages_preserved(self):
        """Test error messages are preserved."""
        message = "GROBID connection timeout after 3 retries"

        try:
            raise IngestionError(message)
        except IngestionError as e:
            assert str(e) == message

    def test_storage_error_independent(self):
        """Test StorageError is independent from IngestionError."""
        assert not issubclass(StorageError, IngestionError)
        assert issubclass(StorageError, ResearchKBError)

    def test_search_error_independent(self):
        """Test SearchError is independent from other errors."""
        assert not issubclass(SearchError, IngestionError)
        assert not issubclass(SearchError, StorageError)
        assert issubclass(SearchError, ResearchKBError)
