"""Extended tests for response formatters.

Tests for formatters not covered in test_formatters.py:
- format_source_detail
- format_citations
- format_citing_sources / format_cited_sources
- format_concept_detail
- format_citation_network
- format_biblio_similar
- format_chunk_concepts
"""

from __future__ import annotations

import pytest
from uuid import uuid4
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from research_kb_mcp.formatters import (
    format_source_detail,
    format_citations,
    format_citing_sources,
    format_cited_sources,
    format_concept_detail,
    format_citation_network,
    format_biblio_similar,
    format_chunk_concepts,
)


# Mock classes for testing
class MockSourceType(Enum):
    PAPER = "paper"
    TEXTBOOK = "textbook"


class MockConceptType(Enum):
    METHOD = "METHOD"
    ASSUMPTION = "ASSUMPTION"
    DEFINITION = "DEFINITION"
    THEOREM = "THEOREM"


class MockSource:
    """Mock Source for testing."""

    def __init__(self, **kwargs):
        self.id = kwargs.get("id", uuid4())
        self.title = kwargs.get("title", "Test Source")
        self.authors = kwargs.get("authors", ["Author A"])
        self.year = kwargs.get("year", 2020)
        self.source_type = kwargs.get("source_type", MockSourceType.PAPER)
        self.metadata = kwargs.get("metadata", {})


class MockChunk:
    """Mock Chunk for testing."""

    def __init__(self, **kwargs):
        self.id = kwargs.get("id", uuid4())
        self.source_id = kwargs.get("source_id", uuid4())
        self.content = kwargs.get("content", "Test chunk content.")
        self.page_start = kwargs.get("page_start", 1)
        self.page_end = kwargs.get("page_end", 1)


class MockConcept:
    """Mock Concept for testing."""

    def __init__(self, **kwargs):
        self.id = kwargs.get("id", uuid4())
        self.name = kwargs.get("name", "Test Concept")
        self.concept_type = kwargs.get("concept_type", MockConceptType.METHOD)
        self.definition = kwargs.get("definition", None)


class MockConceptRelationship:
    """Mock ConceptRelationship for testing."""

    def __init__(self, **kwargs):
        self.source_id = kwargs.get("source_id", uuid4())
        self.target_id = kwargs.get("target_id", uuid4())
        self.relationship_type = kwargs.get("relationship_type", "USES")


class MockChunkConcept:
    """Mock ChunkConcept for testing."""

    def __init__(self, **kwargs):
        self.chunk_id = kwargs.get("chunk_id", uuid4())
        self.concept_id = kwargs.get("concept_id", uuid4())
        self.mention_type = kwargs.get("mention_type", "reference")
        self.relevance_score = kwargs.get("relevance_score", None)


class TestFormatSourceDetail:
    """Tests for format_source_detail function."""

    @pytest.fixture
    def sample_source(self):
        """Create a sample source for testing."""
        return MockSource(
            id=uuid4(),
            title="Causal Inference Methods",
            authors=["Imbens, G.", "Rubin, D.", "Pearl, J."],
            year=2015,
            source_type=MockSourceType.TEXTBOOK,
            metadata={"doi": "10.1234/example", "pages": 450},
        )

    def test_basic_formatting(self, sample_source):
        """Source detail includes title, authors, year, type."""
        result = format_source_detail(sample_source)

        assert "Causal Inference Methods" in result
        assert "Imbens, G." in result
        assert "Rubin, D." in result
        assert "Pearl, J." in result
        assert "2015" in result
        assert "textbook" in result

    def test_includes_metadata(self, sample_source):
        """Source detail includes metadata section."""
        result = format_source_detail(sample_source)

        assert "Metadata" in result
        assert "doi" in result
        assert "10.1234/example" in result

    def test_includes_chunks_when_provided(self, sample_source):
        """Source detail includes chunks when provided."""
        chunks = [
            MockChunk(content="Chapter 1 introduction.", page_start=1),
            MockChunk(content="Chapter 2 methods.", page_start=15),
        ]

        result = format_source_detail(sample_source, chunks=chunks)

        assert "Content Chunks" in result
        assert "2 total" in result
        assert "Chapter 1" in result
        assert "p. 1" in result

    def test_without_chunks(self, sample_source):
        """Source detail works without chunks."""
        result = format_source_detail(sample_source)

        assert "Content Chunks" not in result

    def test_chunks_truncated_to_five(self, sample_source):
        """Only first 5 chunks are shown."""
        chunks = [MockChunk(content=f"Chunk {i}") for i in range(10)]

        result = format_source_detail(sample_source, chunks=chunks)

        assert "... and 5 more chunks" in result

    def test_source_id_included(self, sample_source):
        """Source ID is included for follow-up queries."""
        result = format_source_detail(sample_source)

        assert str(sample_source.id) in result


class TestFormatCitations:
    """Tests for format_citations function."""

    def test_with_citing_and_cited(self):
        """Citations formatted with both directions."""
        source_id = str(uuid4())
        citations_data = {
            "source_id": source_id,
            "citing_sources": [
                {"id": str(uuid4()), "title": "Paper A", "year": 2021},
                {"id": str(uuid4()), "title": "Paper B", "year": 2022},
            ],
            "cited_sources": [
                {"id": str(uuid4()), "title": "Original Paper", "year": 2010},
            ],
        }

        result = format_citations(citations_data)

        assert source_id in result
        assert "Paper A" in result
        assert "2021" in result
        assert "Paper B" in result
        assert "Original Paper" in result
        assert "2010" in result
        assert "Papers Citing This Source (2)" in result
        assert "Papers Cited By This Source (1)" in result

    def test_empty_citing(self):
        """Empty citing list handled gracefully."""
        citations_data = {
            "source_id": str(uuid4()),
            "citing_sources": [],
            "cited_sources": [{"id": str(uuid4()), "title": "Paper X"}],
        }

        result = format_citations(citations_data)

        assert "No citing papers found" in result
        assert "Paper X" in result

    def test_empty_cited(self):
        """Empty cited list handled gracefully."""
        citations_data = {
            "source_id": str(uuid4()),
            "citing_sources": [{"id": str(uuid4()), "title": "Paper Y"}],
            "cited_sources": [],
        }

        result = format_citations(citations_data)

        assert "Paper Y" in result
        assert "No cited papers found" in result

    def test_both_empty(self):
        """Both lists empty handled gracefully."""
        citations_data = {
            "source_id": str(uuid4()),
            "citing_sources": [],
            "cited_sources": [],
        }

        result = format_citations(citations_data)

        assert "No citing papers found" in result
        assert "No cited papers found" in result


class TestFormatCitingSources:
    """Tests for format_citing_sources function."""

    def test_with_sources(self):
        """Citing sources formatted correctly."""
        source_id = str(uuid4())
        sources = [
            MockSource(
                title="Downstream Paper 1",
                authors=["Citer, A.", "Citer, B.", "Citer, C."],
                year=2022,
            ),
            MockSource(title="Downstream Paper 2", authors=["Solo, X."], year=2023),
        ]

        result = format_citing_sources(sources, source_id)

        assert source_id in result
        assert "2 sources cite this paper" in result
        assert "Downstream Paper 1" in result
        assert "Citer, A., Citer, B. et al." in result
        assert "2022" in result
        assert "Downstream Paper 2" in result
        assert "Solo, X." in result

    def test_empty_list(self):
        """Empty citing list handled gracefully."""
        result = format_citing_sources([], "source-123")

        assert "0 sources cite this paper" in result
        assert "No citing sources found" in result


class TestFormatCitedSources:
    """Tests for format_cited_sources function."""

    def test_with_sources(self):
        """Cited sources formatted correctly."""
        source_id = str(uuid4())
        sources = [
            MockSource(title="Foundation Paper", authors=["Ancestor, A."], year=2005),
        ]

        result = format_cited_sources(sources, source_id)

        assert source_id in result
        assert "1 sources are cited" in result
        assert "Foundation Paper" in result
        assert "Ancestor, A." in result

    def test_empty_list(self):
        """Empty cited list handled gracefully."""
        result = format_cited_sources([], "source-456")

        assert "0 sources are cited" in result
        assert "No cited sources found" in result


class TestFormatConceptDetail:
    """Tests for format_concept_detail function."""

    @pytest.fixture
    def sample_concept(self):
        """Create a sample concept for testing."""
        return MockConcept(
            id=uuid4(),
            name="Instrumental Variables",
            concept_type=MockConceptType.METHOD,
            definition="A method for causal inference when there is unmeasured confounding.",
        )

    def test_basic_formatting(self, sample_concept):
        """Concept detail includes name, type, description."""
        result = format_concept_detail(sample_concept)

        assert "Instrumental Variables" in result
        assert "METHOD" in result
        assert "unmeasured confounding" in result

    def test_without_description(self):
        """Concept without description formatted correctly."""
        concept = MockConcept(
            name="Simple Concept",
            concept_type=MockConceptType.ASSUMPTION,
            definition=None,
        )

        result = format_concept_detail(concept)

        assert "Simple Concept" in result
        assert "ASSUMPTION" in result
        assert "Description" not in result

    def test_with_relationships(self, sample_concept):
        """Concept with relationships formatted correctly."""
        relationships = [
            MockConceptRelationship(
                target_id=uuid4(),
                relationship_type="REQUIRES",
            ),
            MockConceptRelationship(
                target_id=uuid4(),
                relationship_type="ADDRESSES",
            ),
        ]

        result = format_concept_detail(sample_concept, relationships=relationships)

        assert "Relationships (2 total)" in result
        assert "REQUIRES" in result
        assert "ADDRESSES" in result

    def test_relationships_truncated_to_twenty(self, sample_concept):
        """Only first 20 relationships shown."""
        relationships = [
            MockConceptRelationship(target_id=uuid4()) for _ in range(30)
        ]

        result = format_concept_detail(sample_concept, relationships=relationships)

        assert "... and 10 more relationships" in result


class TestFormatCitationNetwork:
    """Tests for format_citation_network function."""

    @pytest.fixture
    def center_source(self):
        """Create a center source for testing."""
        return MockSource(
            title="Original Research Paper",
            authors=["Pioneer, P."],
            year=2015,
        )

    def test_with_both_directions(self, center_source):
        """Citation network with both citing and cited."""
        citing = [
            MockSource(title="Follow-up Paper", authors=["Follower, F."], year=2020),
        ]
        cited = [
            MockSource(title="Foundation Paper", authors=["Founder, G."], year=2000),
        ]

        result = format_citation_network(citing, cited, center_source)

        assert "Original Research Paper" in result
        assert "Citing This Source (1)" in result
        assert "Follow-up Paper" in result
        assert "Cited By This Source (1)" in result
        assert "Foundation Paper" in result

    def test_empty_citing(self, center_source):
        """Empty citing list handled gracefully."""
        result = format_citation_network([], [MockSource()], center_source)

        assert "No citing sources found" in result

    def test_empty_cited(self, center_source):
        """Empty cited list handled gracefully."""
        result = format_citation_network([MockSource()], [], center_source)

        assert "No cited sources found" in result

    def test_author_truncation(self, center_source):
        """Authors truncated with 'et al.' when more than 2."""
        citing = [
            MockSource(
                title="Multi-author",
                authors=["Author A", "Author B", "Author C", "Author D"],
            ),
        ]

        result = format_citation_network(citing, [], center_source)

        assert "Author A, Author B et al." in result


class TestFormatBiblioSimilar:
    """Tests for format_biblio_similar function."""

    @pytest.fixture
    def query_source(self):
        """Create a query source for testing."""
        return MockSource(
            title="My Research Paper",
            year=2020,
        )

    def test_with_similar_sources(self, query_source):
        """Similar sources formatted with coupling percentages."""
        similar = [
            {
                "source_id": str(uuid4()),
                "title": "Similar Paper 1",
                "year": 2019,
                "authors": ["Author A"],
                "coupling_strength": 0.75,
                "shared_references": 15,
            },
            {
                "source_id": str(uuid4()),
                "title": "Similar Paper 2",
                "year": 2021,
                "authors": ["Author B", "Author C", "Author D"],
                "coupling_strength": 0.32,
                "shared_references": 6,
            },
        ]

        result = format_biblio_similar(similar, query_source)

        assert "My Research Paper" in result
        assert "2 similar sources" in result
        assert "Similar Paper 1" in result
        assert "75.0%" in result
        assert "15 shared refs" in result
        assert "Similar Paper 2" in result
        assert "32.0%" in result
        assert "Author B, Author C et al." in result

    def test_empty_similar(self, query_source):
        """Empty similar list handled gracefully."""
        result = format_biblio_similar([], query_source)

        assert "No similar sources found" in result
        assert "no shared references" in result


class TestFormatChunkConcepts:
    """Tests for format_chunk_concepts function."""

    @pytest.fixture
    def sample_chunk(self):
        """Create a sample chunk for testing."""
        return MockChunk(
            id=uuid4(),
            content="This chunk discusses instrumental variables.",
            page_start=42,
            page_end=43,
        )

    def test_basic_formatting(self, sample_chunk):
        """Chunk concepts formatted correctly."""
        concepts_with_links = [
            (
                MockConcept(name="IV", concept_type=MockConceptType.METHOD),
                MockChunkConcept(mention_type="reference", relevance_score=0.85),
            ),
        ]

        result = format_chunk_concepts(sample_chunk, concepts_with_links)

        assert str(sample_chunk.id) in result
        assert "pp. 42-43" in result
        assert "1 concepts linked" in result
        assert "IV" in result
        assert "METHOD" in result
        assert "0.85" in result

    def test_grouped_by_mention_type(self, sample_chunk):
        """Concepts grouped by mention type."""
        concepts_with_links = [
            (
                MockConcept(name="Main Concept", concept_type=MockConceptType.DEFINITION),
                MockChunkConcept(mention_type="defines"),
            ),
            (
                MockConcept(name="Referenced Concept", concept_type=MockConceptType.METHOD),
                MockChunkConcept(mention_type="reference"),
            ),
            (
                MockConcept(name="Example Concept", concept_type=MockConceptType.THEOREM),
                MockChunkConcept(mention_type="example"),
            ),
        ]

        result = format_chunk_concepts(sample_chunk, concepts_with_links)

        assert "### Defines" in result
        assert "Main Concept" in result
        assert "### References" in result
        assert "Referenced Concept" in result
        assert "### Examples" in result
        assert "Example Concept" in result

    def test_single_page_reference(self):
        """Single page displayed as 'p. X'."""
        chunk = MockChunk(page_start=10, page_end=10)
        concepts_with_links = [
            (MockConcept(), MockChunkConcept()),
        ]

        result = format_chunk_concepts(chunk, concepts_with_links)

        assert "p. 10" in result
        assert "pp." not in result

    def test_empty_concepts(self, sample_chunk):
        """Empty concepts list handled gracefully."""
        result = format_chunk_concepts(sample_chunk, [])

        assert "0 concepts linked" in result

    def test_no_relevance_score(self, sample_chunk):
        """Missing relevance score handled gracefully."""
        concepts_with_links = [
            (MockConcept(name="No Score"), MockChunkConcept(relevance_score=None)),
        ]

        result = format_chunk_concepts(sample_chunk, concepts_with_links)

        assert "No Score" in result
        assert "relevance:" not in result
