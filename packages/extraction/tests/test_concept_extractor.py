"""Tests for concept extractor."""

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4
from datetime import datetime, timezone

from research_kb_contracts import Chunk
from research_kb_extraction.concept_extractor import ConceptExtractor
from research_kb_extraction.models import (
    ChunkExtraction,
    ExtractedConcept,
    ExtractedRelationship,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_ollama():
    """Create mock Ollama client."""
    client = AsyncMock()
    client.extract_concepts = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_chunk():
    """Create sample chunk for testing."""
    return Chunk(
        id=uuid4(),
        source_id=uuid4(),
        content="""
        Instrumental variables (IV) estimation is a widely used approach for addressing
        endogeneity in econometric analysis. The IV method relies on two key assumptions:
        the relevance condition, which requires that the instrument be correlated with
        the endogenous regressor, and the exclusion restriction, which stipulates that
        the instrument affects the outcome only through its effect on the treatment.
        """,
        content_hash="abc123",
        location="Chapter 5, p. 142",
        created_at=datetime.now(timezone.utc),
    )


class TestConceptExtractorInit:
    """Tests for ConceptExtractor initialization."""

    def test_default_config(self, mock_ollama):
        """Test default configuration."""
        extractor = ConceptExtractor(domain_id="causal_inference", ollama_client=mock_ollama)

        assert extractor.confidence_threshold == 0.7
        assert extractor.min_chunk_length == 100

    def test_custom_config(self, mock_ollama):
        """Test custom configuration."""
        extractor = ConceptExtractor(
            domain_id="causal_inference",
            ollama_client=mock_ollama,
            confidence_threshold=0.8,
            min_chunk_length=200,
        )

        assert extractor.confidence_threshold == 0.8
        assert extractor.min_chunk_length == 200


class TestExtractFromChunk:
    """Tests for extract_from_chunk method."""

    async def test_extract_basic(self, mock_ollama, mock_chunk):
        """Test basic extraction from chunk."""
        mock_ollama.extract_concepts.return_value = ChunkExtraction(
            concepts=[
                ExtractedConcept(
                    name="instrumental variables",
                    concept_type="method",
                    confidence=0.9,
                ),
            ],
            relationships=[],
        )

        extractor = ConceptExtractor(domain_id="causal_inference", ollama_client=mock_ollama)
        result = await extractor.extract_from_chunk(mock_chunk)

        assert isinstance(result, ChunkExtraction)
        assert len(result.concepts) == 1
        mock_ollama.extract_concepts.assert_called_once()

    async def test_skip_short_chunk(self, mock_ollama):
        """Test short chunks are skipped."""
        short_chunk = Chunk(
            id=uuid4(),
            source_id=uuid4(),
            content="Too short",
            content_hash="short",
            created_at=datetime.now(timezone.utc),
        )

        extractor = ConceptExtractor(
            domain_id="causal_inference",
            ollama_client=mock_ollama,
            min_chunk_length=100,
        )
        result = await extractor.extract_from_chunk(short_chunk)

        assert len(result.concepts) == 0
        mock_ollama.extract_concepts.assert_not_called()

    async def test_confidence_filtering(self, mock_ollama, mock_chunk):
        """Test low confidence concepts are filtered."""
        mock_ollama.extract_concepts.return_value = ChunkExtraction(
            concepts=[
                ExtractedConcept(name="high conf", concept_type="method", confidence=0.9),
                ExtractedConcept(name="low conf", concept_type="method", confidence=0.5),
            ],
            relationships=[],
        )

        extractor = ConceptExtractor(
            domain_id="causal_inference",
            ollama_client=mock_ollama,
            confidence_threshold=0.7,
        )
        result = await extractor.extract_from_chunk(mock_chunk)

        assert len(result.concepts) == 1
        assert result.concepts[0].name == "high conf"

    async def test_relationship_filtering(self, mock_ollama, mock_chunk):
        """Test relationships with unknown concepts are filtered."""
        mock_ollama.extract_concepts.return_value = ChunkExtraction(
            concepts=[
                ExtractedConcept(name="IV", concept_type="method", confidence=0.9),
            ],
            relationships=[
                # Valid: references known concept
                ExtractedRelationship(
                    source_concept="IV",
                    target_concept="IV",  # Self-reference for test
                    relationship_type="USES",
                    confidence=0.8,
                ),
                # Invalid: references unknown concept
                ExtractedRelationship(
                    source_concept="unknown",
                    target_concept="IV",
                    relationship_type="REQUIRES",
                    confidence=0.8,
                ),
            ],
        )

        extractor = ConceptExtractor(domain_id="causal_inference", ollama_client=mock_ollama)
        result = await extractor.extract_from_chunk(mock_chunk)

        # Only valid relationship should remain
        assert len(result.relationships) == 1
        assert result.relationships[0].source_concept == "IV"


class TestExtractFromText:
    """Tests for extract_from_text method."""

    async def test_extract_from_text(self, mock_ollama, sample_chunk_text):
        """Test extraction from raw text."""
        mock_ollama.extract_concepts.return_value = ChunkExtraction(
            concepts=[
                ExtractedConcept(name="IV", concept_type="method", confidence=0.9),
            ],
            relationships=[],
        )

        extractor = ConceptExtractor(domain_id="causal_inference", ollama_client=mock_ollama)
        result = await extractor.extract_from_text(sample_chunk_text)

        assert isinstance(result, ChunkExtraction)
        assert len(result.concepts) == 1

    async def test_extract_from_short_text(self, mock_ollama):
        """Test short text returns empty extraction."""
        extractor = ConceptExtractor(
            domain_id="causal_inference",
            ollama_client=mock_ollama,
            min_chunk_length=100,
        )
        result = await extractor.extract_from_text("Short")

        assert len(result.concepts) == 0
        mock_ollama.extract_concepts.assert_not_called()


class TestConceptNormalization:
    """Tests for concept normalization."""

    async def test_whitespace_cleanup(self, mock_ollama, mock_chunk):
        """Test whitespace is cleaned in concept names."""
        mock_ollama.extract_concepts.return_value = ChunkExtraction(
            concepts=[
                ExtractedConcept(
                    name="  instrumental   variables  ",
                    concept_type="method",
                    confidence=0.9,
                ),
            ],
            relationships=[],
        )

        extractor = ConceptExtractor(domain_id="causal_inference", ollama_client=mock_ollama)
        result = await extractor.extract_from_chunk(mock_chunk)

        assert result.concepts[0].name == "instrumental variables"

    async def test_quote_removal(self, mock_ollama, mock_chunk):
        """Test quotes are removed from concept names."""
        mock_ollama.extract_concepts.return_value = ChunkExtraction(
            concepts=[
                ExtractedConcept(
                    name='"instrumental variables"',
                    concept_type="method",
                    confidence=0.9,
                ),
            ],
            relationships=[],
        )

        extractor = ConceptExtractor(domain_id="causal_inference", ollama_client=mock_ollama)
        result = await extractor.extract_from_chunk(mock_chunk)

        assert result.concepts[0].name == "instrumental variables"


class TestDeduplication:
    """Tests for concept deduplication."""

    async def test_deduplicate_concepts(self, mock_ollama):
        """Test deduplication across extractions."""
        extractions = [
            ChunkExtraction(
                concepts=[
                    ExtractedConcept(name="IV", concept_type="method", confidence=0.9),
                ],
                relationships=[],
            ),
            ChunkExtraction(
                concepts=[
                    ExtractedConcept(
                        name="instrumental variables",
                        concept_type="method",
                        confidence=0.9,
                    ),
                ],
                relationships=[],
            ),
        ]

        extractor = ConceptExtractor(domain_id="causal_inference", ollama_client=mock_ollama)
        results = await extractor.deduplicate_concepts(extractions)

        # Should recognize IV and instrumental variables as same concept
        canonical_names = {r.matched_canonical_name for r in results}
        assert len(canonical_names) == 1
        assert "instrumental variables" in canonical_names


class TestContextManager:
    """Tests for async context manager."""

    async def test_context_manager_closes(self, mock_ollama):
        """Test context manager closes client."""
        async with ConceptExtractor(domain_id="causal_inference", ollama_client=mock_ollama):
            pass

        mock_ollama.close.assert_called_once()
