"""Tests for source MCP tools."""

from __future__ import annotations

import pytest
from unittest.mock import patch
from uuid import uuid4
from datetime import datetime

from research_kb_contracts import Source, SourceType, Chunk
from research_kb_mcp.tools.sources import register_source_tools

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


class TestSourceToolRegistration:
    """Tests for source tool registration."""

    def test_list_sources_registered(self):
        """List sources tool is registered correctly."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        assert "research_kb_list_sources" in mcp.tools
        doc = mcp.tools["research_kb_list_sources"]["func"].__doc__
        assert doc is not None
        assert "sources" in doc.lower()

    def test_get_source_registered(self):
        """Get source tool is registered correctly."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        assert "research_kb_get_source" in mcp.tools
        doc = mcp.tools["research_kb_get_source"]["func"].__doc__
        assert doc is not None
        assert "source_id" in doc

    def test_get_source_citations_registered(self):
        """Get source citations tool is registered correctly."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        assert "research_kb_get_source_citations" in mcp.tools

    def test_get_citing_sources_registered(self):
        """Get citing sources tool is registered correctly."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        assert "research_kb_get_citing_sources" in mcp.tools

    def test_get_cited_sources_registered(self):
        """Get cited sources tool is registered correctly."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        assert "research_kb_get_cited_sources" in mcp.tools


class TestListSources:
    """Tests for list sources tool functionality."""

    @pytest.fixture
    def sample_sources(self):
        """Create sample sources for testing."""
        return [
            Source(
                id=uuid4(),
                title="Double Machine Learning",
                source_type=SourceType.PAPER,
                authors=["Chernozhukov, V."],
                year=2018,
                file_hash="abc123",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            Source(
                id=uuid4(),
                title="Causal Inference Textbook",
                source_type=SourceType.TEXTBOOK,
                authors=["Imbens, G.", "Rubin, D."],
                year=2015,
                file_hash="def456",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

    @pytest.mark.asyncio
    async def test_list_sources_default(self, sample_sources):
        """List sources returns formatted results with defaults."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        with patch("research_kb_mcp.tools.sources.get_sources") as get_mock:
            get_mock.return_value = sample_sources

            result = await mcp.tools["research_kb_list_sources"]["func"]()

            get_mock.assert_called_once_with(limit=50, offset=0, source_type=None)
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_list_sources_pagination(self, sample_sources):
        """List sources respects pagination parameters."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        with patch("research_kb_mcp.tools.sources.get_sources") as get_mock:
            get_mock.return_value = sample_sources

            await mcp.tools["research_kb_list_sources"]["func"](
                limit=20,
                offset=10,
            )

            get_mock.assert_called_once_with(limit=20, offset=10, source_type=None)

    @pytest.mark.asyncio
    async def test_list_sources_by_type(self, sample_sources):
        """List sources filters by type."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        with patch("research_kb_mcp.tools.sources.get_sources") as get_mock:
            get_mock.return_value = [sample_sources[0]]  # Just papers

            await mcp.tools["research_kb_list_sources"]["func"](
                source_type="paper",
            )

            get_mock.assert_called_once_with(limit=50, offset=0, source_type="paper")

    @pytest.mark.asyncio
    async def test_list_sources_limit_clamping(self, sample_sources):
        """List sources clamps limit to valid range."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        with patch("research_kb_mcp.tools.sources.get_sources") as get_mock:
            get_mock.return_value = sample_sources

            # Test upper bound
            await mcp.tools["research_kb_list_sources"]["func"](limit=200)
            assert get_mock.call_args[1]["limit"] == 100

            # Test lower bound
            await mcp.tools["research_kb_list_sources"]["func"](limit=0)
            assert get_mock.call_args[1]["limit"] == 1


class TestGetSource:
    """Tests for get source tool functionality."""

    @pytest.fixture
    def sample_source(self):
        """Create a sample source for testing."""
        return Source(
            id=uuid4(),
            title="Treatment Effects Paper",
            source_type=SourceType.PAPER,
            authors=["Author, A."],
            year=2020,
            file_hash="test123",
            metadata={"doi": "10.1234/example"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.fixture
    def sample_chunks(self, sample_source):
        """Create sample chunks for testing."""
        return [
            Chunk(
                id=uuid4(),
                source_id=sample_source.id,
                content="This is the introduction.",
                content_hash="chunk1",
                page_start=1,
                page_end=1,
                metadata={},
                created_at=datetime.now(),
            ),
            Chunk(
                id=uuid4(),
                source_id=sample_source.id,
                content="This is the methods section.",
                content_hash="chunk2",
                page_start=5,
                page_end=5,
                metadata={},
                created_at=datetime.now(),
            ),
        ]

    @pytest.mark.asyncio
    async def test_get_source_success(self, sample_source):
        """Get source returns formatted details."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        with patch("research_kb_mcp.tools.sources.get_source_by_id") as get_mock:
            get_mock.return_value = sample_source

            result = await mcp.tools["research_kb_get_source"]["func"](
                source_id=str(sample_source.id),
            )

            get_mock.assert_called_once_with(str(sample_source.id))
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_source_with_chunks(self, sample_source, sample_chunks):
        """Get source includes chunks when requested."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        with (
            patch("research_kb_mcp.tools.sources.get_source_by_id") as get_mock,
            patch("research_kb_mcp.tools.sources.get_source_chunks") as chunks_mock,
        ):
            get_mock.return_value = sample_source
            chunks_mock.return_value = sample_chunks

            result = await mcp.tools["research_kb_get_source"]["func"](
                source_id=str(sample_source.id),
                include_chunks=True,
                chunk_limit=5,
            )

            chunks_mock.assert_called_once_with(str(sample_source.id), limit=5)
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_source_not_found(self):
        """Get source returns error for missing source."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        with patch("research_kb_mcp.tools.sources.get_source_by_id") as get_mock:
            get_mock.return_value = None

            result = await mcp.tools["research_kb_get_source"]["func"](
                source_id="nonexistent-id",
            )

            assert "Error" in result
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_get_source_chunk_limit_clamping(self, sample_source, sample_chunks):
        """Get source clamps chunk limit to valid range."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        with (
            patch("research_kb_mcp.tools.sources.get_source_by_id") as get_mock,
            patch("research_kb_mcp.tools.sources.get_source_chunks") as chunks_mock,
        ):
            get_mock.return_value = sample_source
            chunks_mock.return_value = sample_chunks

            # Test upper bound
            await mcp.tools["research_kb_get_source"]["func"](
                source_id=str(sample_source.id),
                include_chunks=True,
                chunk_limit=100,
            )
            assert chunks_mock.call_args[1]["limit"] == 50


class TestSourceCitations:
    """Tests for source citation tools functionality."""

    @pytest.fixture
    def sample_source(self):
        """Create a sample source for testing."""
        return Source(
            id=uuid4(),
            title="Original Paper",
            source_type=SourceType.PAPER,
            authors=["Author, A."],
            year=2018,
            file_hash="test123",
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
                title="Citing Paper 1",
                source_type=SourceType.PAPER,
                authors=["Citer, C."],
                year=2020,
                file_hash="cite1",
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
                title="Cited Paper 1",
                source_type=SourceType.PAPER,
                authors=["Cited, D."],
                year=2010,
                file_hash="cited1",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

    @pytest.mark.asyncio
    async def test_get_source_citations_success(self, sample_source):
        """Get source citations returns formatted results."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        # format_citations expects these keys
        mock_citations = {
            "source_id": str(sample_source.id),
            "citing_sources": [],
            "cited_sources": [],
        }

        with (
            patch("research_kb_mcp.tools.sources.get_source_by_id") as get_mock,
            patch("research_kb_mcp.tools.sources.get_citations_for_source") as cite_mock,
        ):
            get_mock.return_value = sample_source
            cite_mock.return_value = mock_citations

            result = await mcp.tools["research_kb_get_source_citations"]["func"](
                source_id=str(sample_source.id),
            )

            cite_mock.assert_called_once()
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_source_citations_not_found(self):
        """Get source citations returns error for missing source."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        with patch("research_kb_mcp.tools.sources.get_source_by_id") as get_mock:
            get_mock.return_value = None

            result = await mcp.tools["research_kb_get_source_citations"]["func"](
                source_id="nonexistent-id",
            )

            assert "Error" in result
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_get_citing_sources_success(self, sample_source, citing_sources):
        """Get citing sources returns formatted results."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        with (
            patch("research_kb_mcp.tools.sources.get_source_by_id") as get_mock,
            patch("research_kb_mcp.tools.sources.get_citing_sources") as cite_mock,
        ):
            get_mock.return_value = sample_source
            cite_mock.return_value = citing_sources

            result = await mcp.tools["research_kb_get_citing_sources"]["func"](
                source_id=str(sample_source.id),
            )

            cite_mock.assert_called_once()
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_cited_sources_success(self, sample_source, cited_sources):
        """Get cited sources returns formatted results."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        with (
            patch("research_kb_mcp.tools.sources.get_source_by_id") as get_mock,
            patch("research_kb_mcp.tools.sources.get_cited_sources") as cite_mock,
        ):
            get_mock.return_value = sample_source
            cite_mock.return_value = cited_sources

            result = await mcp.tools["research_kb_get_cited_sources"]["func"](
                source_id=str(sample_source.id),
            )

            cite_mock.assert_called_once()
            assert isinstance(result, str)
