"""Tests for citation MCP tools."""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock
from uuid import uuid4
from datetime import datetime

from research_kb_contracts import (
    Source,
    SourceType,
)
from research_kb_mcp.tools.citations import register_citation_tools
from research_kb_mcp.tools.concepts import register_concept_tools

pytestmark = pytest.mark.unit


class MockFastMCP:
    """Mock FastMCP server for testing tool registration."""

    def __init__(self):
        self.tools = {}

    def tool(self, **kwargs):
        """Decorator that captures tool functions."""

        def decorator(func):
            self.tools[func.__name__] = {
                "func": func,
                "kwargs": kwargs,
            }
            return func

        return decorator


class TestCitationToolRegistration:
    """Tests for citation tool registration."""

    def test_citation_network_tool_registered(self):
        """Citation network tool is registered correctly."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        assert "research_kb_citation_network" in mcp.tools
        # Check docstring is present
        doc = mcp.tools["research_kb_citation_network"]["func"].__doc__
        assert doc is not None
        assert "bidirectional" in doc.lower()
        assert "citing" in doc.lower()

    def test_biblio_coupling_tool_registered(self):
        """Bibliographic coupling tool is registered correctly."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        assert "research_kb_biblio_coupling" in mcp.tools
        # Check docstring is present
        doc = mcp.tools["research_kb_biblio_coupling"]["func"].__doc__
        assert doc is not None
        assert "coupling" in doc.lower()
        assert "jaccard" in doc.lower()

    def test_chunk_concepts_tool_registered(self):
        """Chunk concepts tool is still registered but retired (RS4, ADR-0001)."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        assert "research_kb_chunk_concepts" in mcp.tools
        # Chunk-concept links were retired in RS4; the tool is an inert stub
        # whose docstring reflects retirement.
        doc = mcp.tools["research_kb_chunk_concepts"]["func"].__doc__
        assert doc is not None
        assert "retired" in doc.lower()

    def test_all_citation_tools_have_docstrings(self):
        """All citation tools have docstrings for MCP schema."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        for name, tool in mcp.tools.items():
            assert tool["func"].__doc__, f"Tool {name} missing docstring"


class TestCitationNetworkTool:
    """Tests for citation network tool functionality."""

    @pytest.fixture
    def sample_source(self):
        """Create a sample source for testing."""
        return Source(
            id=uuid4(),
            title="Double Machine Learning for Treatment Effects",
            source_type=SourceType.PAPER,
            authors=["Chernozhukov, V.", "Chetverikov, D."],
            year=2018,
            domain_id="causal_inference",
            file_hash="abc123",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.fixture
    def citing_sources(self):
        """Create citing sources for testing."""
        return [
            Source(
                id=uuid4(),
                title="Causal Forest Applications",
                source_type=SourceType.PAPER,
                authors=["Wager, S."],
                year=2019,
                domain_id="causal_inference",
                file_hash="def456",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

    @pytest.fixture
    def cited_sources(self):
        """Create cited sources for testing."""
        return [
            Source(
                id=uuid4(),
                title="Rubin Causal Model",
                source_type=SourceType.PAPER,
                authors=["Rubin, D."],
                year=1974,
                domain_id="causal_inference",
                file_hash="ghi789",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

    async def test_citation_network_success(self, sample_source, citing_sources, cited_sources):
        """Citation network returns formatted results."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with (
            patch("research_kb_mcp.tools.citations.get_source_by_id") as get_source_mock,
            patch("research_kb_mcp.tools.citations.get_citing_sources") as citing_mock,
            patch("research_kb_mcp.tools.citations.get_cited_sources") as cited_mock,
        ):

            get_source_mock.return_value = sample_source
            citing_mock.return_value = citing_sources
            cited_mock.return_value = cited_sources

            result = await mcp.tools["research_kb_citation_network"]["func"](
                source_id=str(sample_source.id),
                limit=20,
            )

            assert "Citation Network" in result
            assert sample_source.title in result
            assert "Citing This Source" in result
            assert "Cited By This Source" in result

    async def test_citation_network_not_found(self):
        """Citation network returns error for missing source."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with patch("research_kb_mcp.tools.citations.get_source_by_id") as get_source_mock:
            get_source_mock.return_value = None

            result = await mcp.tools["research_kb_citation_network"]["func"](
                source_id="nonexistent-id",
            )

            assert "Error" in result
            assert "not found" in result


class TestBiblioCouplingTool:
    """Tests for bibliographic coupling tool functionality."""

    @pytest.fixture
    def sample_source(self):
        """Create a sample source for testing."""
        return Source(
            id=uuid4(),
            title="Original Paper",
            source_type=SourceType.PAPER,
            authors=["Author, A."],
            year=2020,
            domain_id="causal_inference",
            file_hash="test123",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.fixture
    def similar_sources(self):
        """Create similar sources for testing."""
        return [
            {
                "source_id": uuid4(),
                "title": "Similar Paper 1",
                "authors": ["Other, B."],
                "year": 2021,
                "source_type": "paper",
                "shared_references": 5,
                "coupling_strength": 0.45,
            },
            {
                "source_id": uuid4(),
                "title": "Similar Paper 2",
                "authors": ["Another, C."],
                "year": 2020,
                "source_type": "paper",
                "shared_references": 3,
                "coupling_strength": 0.25,
            },
        ]

    async def test_biblio_coupling_success(self, sample_source, similar_sources):
        """Bibliographic coupling returns formatted results."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with (
            patch("research_kb_mcp.tools.citations.get_source_by_id") as get_source_mock,
            patch("research_kb_mcp.tools.citations.BiblioStore") as biblio_mock,
        ):

            get_source_mock.return_value = sample_source
            biblio_mock.get_similar_sources = AsyncMock(return_value=similar_sources)

            result = await mcp.tools["research_kb_biblio_coupling"]["func"](
                source_id=str(sample_source.id),
                limit=10,
                min_coupling=0.1,
            )

            assert "Bibliographically Similar" in result
            assert sample_source.title in result
            assert "Similar Paper 1" in result
            assert "45.0%" in result  # coupling percentage
            assert "5 shared refs" in result

    async def test_biblio_coupling_not_found(self):
        """Bibliographic coupling returns error for missing source."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with patch("research_kb_mcp.tools.citations.get_source_by_id") as get_source_mock:
            get_source_mock.return_value = None

            result = await mcp.tools["research_kb_biblio_coupling"]["func"](
                source_id="nonexistent-id",
            )

            assert "Error" in result
            assert "not found" in result

    async def test_biblio_coupling_empty_results(self, sample_source):
        """Bibliographic coupling handles no similar sources."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with (
            patch("research_kb_mcp.tools.citations.get_source_by_id") as get_source_mock,
            patch("research_kb_mcp.tools.citations.BiblioStore") as biblio_mock,
        ):

            get_source_mock.return_value = sample_source
            biblio_mock.get_similar_sources = AsyncMock(return_value=[])

            result = await mcp.tools["research_kb_biblio_coupling"]["func"](
                source_id=str(sample_source.id),
            )

            assert "No similar sources found" in result
