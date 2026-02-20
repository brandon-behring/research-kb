"""Tests for citation MCP tools."""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock
from uuid import uuid4
from datetime import datetime

from research_kb_contracts import (
    Source,
    SourceType,
    Concept,
    ConceptType,
    Chunk,
    ChunkConcept,
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
        """Chunk concepts tool is registered correctly."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        assert "research_kb_chunk_concepts" in mcp.tools
        # Check docstring is present
        doc = mcp.tools["research_kb_chunk_concepts"]["func"].__doc__
        assert doc is not None
        assert "mention" in doc.lower()
        assert "defines" in doc.lower()

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
                file_hash="ghi789",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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


class TestChunkConceptsTool:
    """Tests for chunk concepts tool functionality."""

    @pytest.fixture
    def sample_chunk(self):
        """Create a sample chunk for testing."""
        return Chunk(
            id=uuid4(),
            source_id=uuid4(),
            content="Double machine learning uses Neyman orthogonality.",
            content_hash="chunk123",
            page_start=15,
            page_end=15,
            metadata={"section": "Methods"},
            created_at=datetime.now(),
        )

    @pytest.fixture
    def sample_concepts(self, sample_chunk):
        """Create sample concepts for testing."""
        concept_id1 = uuid4()
        concept_id2 = uuid4()

        concepts = [
            Concept(
                id=concept_id1,
                name="Double Machine Learning",
                canonical_name="double_machine_learning",
                concept_type=ConceptType.METHOD,
                created_at=datetime.now(),
            ),
            Concept(
                id=concept_id2,
                name="Neyman Orthogonality",
                canonical_name="neyman_orthogonality",
                concept_type=ConceptType.THEOREM,
                created_at=datetime.now(),
            ),
        ]

        chunk_concepts = [
            ChunkConcept(
                chunk_id=sample_chunk.id,
                concept_id=concept_id1,
                mention_type="defines",
                relevance_score=0.95,
                created_at=datetime.now(),
            ),
            ChunkConcept(
                chunk_id=sample_chunk.id,
                concept_id=concept_id2,
                mention_type="reference",
                relevance_score=0.75,
                created_at=datetime.now(),
            ),
        ]

        return concepts, chunk_concepts

    @pytest.mark.asyncio
    async def test_chunk_concepts_success(self, sample_chunk, sample_concepts):
        """Chunk concepts returns formatted results."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        concepts, chunk_concepts = sample_concepts

        with (
            patch("research_kb_mcp.tools.concepts.ChunkStore") as chunk_mock,
            patch("research_kb_mcp.tools.concepts.ChunkConceptStore") as cc_mock,
            patch("research_kb_mcp.tools.concepts.ConceptStore") as concept_mock,
        ):

            chunk_mock.get_by_id = AsyncMock(return_value=sample_chunk)
            cc_mock.list_concepts_for_chunk = AsyncMock(return_value=chunk_concepts)

            # Return concepts in order
            concept_mock.get = AsyncMock(side_effect=concepts)

            result = await mcp.tools["research_kb_chunk_concepts"]["func"](
                chunk_id=str(sample_chunk.id),
            )

            assert "Concepts in Chunk" in result
            assert "Double Machine Learning" in result
            assert "Defines" in result
            assert "References" in result

    @pytest.mark.asyncio
    async def test_chunk_concepts_not_found(self):
        """Chunk concepts returns error for missing chunk."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        nonexistent_id = str(uuid4())

        with patch("research_kb_mcp.tools.concepts.ChunkStore") as chunk_mock:
            chunk_mock.get_by_id = AsyncMock(return_value=None)

            result = await mcp.tools["research_kb_chunk_concepts"]["func"](
                chunk_id=nonexistent_id,
            )

            assert "Error" in result
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_chunk_concepts_empty(self, sample_chunk):
        """Chunk concepts handles no linked concepts."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with (
            patch("research_kb_mcp.tools.concepts.ChunkStore") as chunk_mock,
            patch("research_kb_mcp.tools.concepts.ChunkConceptStore") as cc_mock,
        ):

            chunk_mock.get_by_id = AsyncMock(return_value=sample_chunk)
            cc_mock.list_concepts_for_chunk = AsyncMock(return_value=[])

            result = await mcp.tools["research_kb_chunk_concepts"]["func"](
                chunk_id=str(sample_chunk.id),
            )

            assert "No concepts linked" in result
