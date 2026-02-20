"""Tests for response formatters."""

from __future__ import annotations

import pytest
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Optional

from research_kb_mcp.formatters import (
    truncate,
    format_search_results,
    format_source_list,
    format_concept_list,
    format_graph_neighborhood,
    format_graph_path,
    format_stats,
    format_health,
)

pytestmark = pytest.mark.unit


# Mock dataclasses for testing
@dataclass
class MockScoreBreakdown:
    fts: float = 0.0
    vector: float = 0.0
    graph: float = 0.0
    citation: float = 0.0
    combined: float = 0.0


@dataclass
class MockSourceSummary:
    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    source_type: Optional[str] = None


@dataclass
class MockChunkSummary:
    id: str
    content: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None


@dataclass
class MockSearchResultDetail:
    source: MockSourceSummary
    chunk: MockChunkSummary
    concepts: list[str] = field(default_factory=list)
    scores: MockScoreBreakdown = field(default_factory=MockScoreBreakdown)
    combined_score: float = 0.0


@dataclass
class MockSearchResponse:
    query: str
    expanded_query: Optional[str] = None
    results: list[MockSearchResultDetail] = field(default_factory=list)
    execution_time_ms: float = 0.0


class MockSource:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", uuid4())
        self.title = kwargs.get("title", "Test Source")
        self.authors = kwargs.get("authors", ["Author A"])
        self.year = kwargs.get("year", 2020)
        self.source_type = kwargs.get("source_type", "paper")
        self.metadata = kwargs.get("metadata", {})


class MockConcept:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", uuid4())
        self.name = kwargs.get("name", "Test Concept")
        self.concept_type = kwargs.get("concept_type", "METHOD")
        self.description = kwargs.get("description", "A test concept")


class MockRelationship:
    def __init__(self, **kwargs):
        self.source_id = kwargs.get("source_id", uuid4())
        self.target_id = kwargs.get("target_id", uuid4())
        self.relationship_type = kwargs.get("relationship_type", "USES")


class TestTruncate:
    """Tests for truncate function."""

    def test_short_text_unchanged(self):
        """Short text passes through unchanged."""
        text = "short text"
        assert truncate(text) == text

    def test_long_text_truncated(self):
        """Long text gets truncated with ellipsis."""
        text = "x" * 2000
        result = truncate(text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_exact_length(self):
        """Text at exact limit unchanged."""
        text = "x" * 100
        assert truncate(text, max_length=100) == text


class TestFormatSearchResults:
    """Tests for search result formatting."""

    def test_empty_results(self):
        """Empty results handled gracefully."""
        response = MockSearchResponse(query="test query")
        result = format_search_results(response)
        assert "test query" in result
        assert "0 results" in result
        assert "No results found" in result

    def test_with_results(self):
        """Results formatted correctly."""
        source = MockSourceSummary(
            id=str(uuid4()),
            title="Test Paper",
            authors=["Alice", "Bob"],
            year=2020,
            source_type="paper",
        )
        chunk = MockChunkSummary(
            id=str(uuid4()),
            content="This is test content about causal inference.",
            page_start=10,
            section="Introduction",
        )
        result_detail = MockSearchResultDetail(
            source=source,
            chunk=chunk,
            scores=MockScoreBreakdown(fts=0.5, vector=0.8, combined=0.65),
            combined_score=0.65,
        )
        response = MockSearchResponse(
            query="causal inference",
            results=[result_detail],
            execution_time_ms=150.0,
        )

        result = format_search_results(response)
        assert "Test Paper" in result
        assert "Alice" in result
        assert "2020" in result
        assert "causal inference" in result
        assert "p. 10" in result

    def test_expanded_query_shown(self):
        """Expanded query displayed when present."""
        response = MockSearchResponse(
            query="DML",
            expanded_query="double machine learning debiased",
        )
        result = format_search_results(response)
        assert "double machine learning" in result


class TestFormatSourceList:
    """Tests for source list formatting."""

    def test_empty_list(self):
        """Empty list handled gracefully."""
        result = format_source_list([])
        assert "No sources found" in result

    def test_with_sources(self):
        """Sources formatted correctly."""
        sources = [
            MockSource(title="Paper A", authors=["A", "B", "C"], year=2020),
            MockSource(title="Book B", authors=["X"], year=2019),
        ]
        result = format_source_list(sources)
        assert "Paper A" in result
        assert "Book B" in result
        assert "2 total" in result


class TestFormatConceptList:
    """Tests for concept list formatting."""

    def test_empty_list(self):
        """Empty list handled gracefully."""
        result = format_concept_list([])
        assert "No concepts found" in result

    def test_with_concepts(self):
        """Concepts formatted correctly."""
        concepts = [
            MockConcept(name="Causal Forest", concept_type="METHOD"),
            MockConcept(name="Unconfoundedness", concept_type="ASSUMPTION"),
        ]
        result = format_concept_list(concepts)
        assert "Causal Forest" in result
        assert "Unconfoundedness" in result


class TestFormatGraphNeighborhood:
    """Tests for graph neighborhood formatting."""

    def test_error_case(self):
        """Error case handled."""
        result = format_graph_neighborhood({"error": "Not found"})
        assert "Error" in result
        assert "Not found" in result

    def test_with_data(self):
        """Neighborhood data formatted correctly."""
        data = {
            "center": {"id": "123", "name": "DML", "type": "METHOD"},
            "nodes": [
                {"id": "456", "name": "Cross-fitting", "type": "METHOD"},
            ],
            "edges": [
                {"source": "123", "target": "456", "type": "USES"},
            ],
        }
        result = format_graph_neighborhood(data)
        assert "DML" in result
        assert "Cross-fitting" in result
        assert "1 connected" in result


class TestFormatGraphPath:
    """Tests for graph path formatting."""

    def test_error_case(self):
        """Error case handled."""
        result = format_graph_path({"error": "No path found"})
        assert "Error" in result

    def test_no_path(self):
        """No path case handled."""
        result = format_graph_path({"from": "A", "to": "B", "path": []})
        assert "No path found" in result

    def test_with_path(self):
        """Path formatted correctly."""
        data = {
            "from": "DML",
            "to": "IV",
            "path": [
                {"name": "DML"},
                {"name": "LATE"},
                {"name": "IV"},
            ],
        }
        result = format_graph_path(data)
        assert "DML" in result
        assert "IV" in result
        assert "2 hops" in result


class TestFormatStats:
    """Tests for stats formatting."""

    def test_formats_table(self):
        """Stats formatted as table."""
        stats = {
            "sources": 100,
            "chunks": 5000,
            "concepts": 800,
        }
        result = format_stats(stats)
        assert "Sources" in result
        assert "100" in result
        assert "5,000" in result


class TestFormatHealth:
    """Tests for health formatting."""

    def test_healthy(self):
        """Healthy status formatted correctly."""
        result = format_health(healthy=True, details={"db": "ok"})
        assert "Healthy" in result
        assert "db" in result

    def test_unhealthy(self):
        """Unhealthy status formatted correctly."""
        result = format_health(healthy=False, details={"error": "Connection failed"})
        assert "Unhealthy" in result
        assert "Connection failed" in result
