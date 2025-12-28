"""Tests for search MCP tool."""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime
from dataclasses import dataclass

from research_kb_contracts import Source, SourceType, Chunk
from research_kb_mcp.tools.search import register_search_tools


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


@dataclass
class MockSearchResponse:
    """Mock search response for testing."""
    query: str
    results: list
    total_count: int = 0
    search_time_ms: float = 50.0
    expanded_query: str = None
    execution_time_ms: float = 50.0


@dataclass
class MockSearchResult:
    """Mock search result for testing."""
    source: Source
    chunk: Chunk
    fts_score: float = 0.5
    vector_score: float = 0.7
    graph_score: float = 0.1
    citation_score: float = 0.05
    combined_score: float = 0.8


class TestSearchToolRegistration:
    """Tests for search tool registration."""

    def test_search_tool_registered(self):
        """Search tool is registered correctly."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        assert "research_kb_search" in mcp.tools

    def test_search_tool_has_docstring(self):
        """Search tool has comprehensive docstring."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        doc = mcp.tools["research_kb_search"]["func"].__doc__
        assert doc is not None
        assert "hybrid search" in doc.lower()
        assert "full-text" in doc.lower()
        assert "vector" in doc.lower()
        assert "knowledge graph" in doc.lower()
        assert "citation" in doc.lower()

    def test_search_tool_documents_parameters(self):
        """Search tool documents all parameters."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        doc = mcp.tools["research_kb_search"]["func"].__doc__
        assert "query" in doc
        assert "limit" in doc
        assert "context_type" in doc
        assert "use_graph" in doc
        assert "use_rerank" in doc
        assert "use_citations" in doc


class TestSearchToolExecution:
    """Tests for search tool execution."""

    @pytest.fixture
    def sample_source(self):
        """Create a sample source for testing."""
        return Source(
            id=uuid4(),
            title="Instrumental Variables Methods",
            source_type=SourceType.PAPER,
            authors=["Angrist, J.", "Imbens, G."],
            year=1995,
            file_hash="test123",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.fixture
    def sample_chunk(self, sample_source):
        """Create a sample chunk for testing."""
        return Chunk(
            id=uuid4(),
            source_id=sample_source.id,
            content="Instrumental variables provide a way to estimate causal effects when there is unmeasured confounding.",
            content_hash="chunk123",
            page_start=10,
            page_end=10,
            metadata={"section_header": "Introduction"},
            created_at=datetime.now(),
        )

    @pytest.fixture
    def mock_search_response(self, sample_source, sample_chunk):
        """Create a mock search response."""
        result = MockSearchResult(
            source=sample_source,
            chunk=sample_chunk,
        )
        return MockSearchResponse(
            query="instrumental variables",
            results=[result],
            total_count=1,
        )

    @pytest.mark.asyncio
    async def test_search_with_default_params(self, mock_search_response):
        """Search returns formatted results with defaults."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with patch("research_kb_mcp.tools.search.search") as search_mock, \
             patch("research_kb_mcp.tools.search.format_search_results") as format_mock:
            search_mock.return_value = mock_search_response
            format_mock.return_value = "## Search Results\n\nFormatted results here"

            result = await mcp.tools["research_kb_search"]["func"](
                query="instrumental variables",
            )

            # Verify search was called
            search_mock.assert_called_once()
            call_args = search_mock.call_args[0][0]
            assert call_args.query == "instrumental variables"
            assert call_args.limit == 10  # default
            assert call_args.use_graph is True  # default
            assert call_args.use_citations is True  # default

            # Verify result format
            assert "Search Results" in result or "results" in result.lower()

    @pytest.mark.asyncio
    async def test_search_with_context_building(self, mock_search_response):
        """Search respects building context type."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with patch("research_kb_mcp.tools.search.search") as search_mock, \
             patch("research_kb_mcp.tools.search.format_search_results") as format_mock:
            search_mock.return_value = mock_search_response
            format_mock.return_value = "## Search Results"

            await mcp.tools["research_kb_search"]["func"](
                query="treatment effects",
                context_type="building",
            )

            call_args = search_mock.call_args[0][0]
            assert call_args.context_type.value == "building"

    @pytest.mark.asyncio
    async def test_search_with_context_auditing(self, mock_search_response):
        """Search respects auditing context type."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with patch("research_kb_mcp.tools.search.search") as search_mock, \
             patch("research_kb_mcp.tools.search.format_search_results") as format_mock:
            search_mock.return_value = mock_search_response
            format_mock.return_value = "## Search Results"

            await mcp.tools["research_kb_search"]["func"](
                query="treatment effects",
                context_type="auditing",
            )

            call_args = search_mock.call_args[0][0]
            assert call_args.context_type.value == "auditing"

    @pytest.mark.asyncio
    async def test_search_disabling_features(self, mock_search_response):
        """Search respects disabled features."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with patch("research_kb_mcp.tools.search.search") as search_mock, \
             patch("research_kb_mcp.tools.search.format_search_results") as format_mock:
            search_mock.return_value = mock_search_response
            format_mock.return_value = "## Search Results"

            await mcp.tools["research_kb_search"]["func"](
                query="DML",
                use_graph=False,
                use_rerank=False,
                use_expand=False,
                use_citations=False,
            )

            call_args = search_mock.call_args[0][0]
            assert call_args.use_graph is False
            assert call_args.use_rerank is False
            assert call_args.use_expand is False
            assert call_args.use_citations is False

    @pytest.mark.asyncio
    async def test_search_limit_clamping_upper(self, mock_search_response):
        """Search clamps limit to maximum 50."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with patch("research_kb_mcp.tools.search.search") as search_mock, \
             patch("research_kb_mcp.tools.search.format_search_results") as format_mock:
            search_mock.return_value = mock_search_response
            format_mock.return_value = "## Search Results"

            await mcp.tools["research_kb_search"]["func"](
                query="test",
                limit=100,  # exceeds max
            )

            call_args = search_mock.call_args[0][0]
            assert call_args.limit == 50  # clamped

    @pytest.mark.asyncio
    async def test_search_limit_clamping_lower(self, mock_search_response):
        """Search clamps limit to minimum 1."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with patch("research_kb_mcp.tools.search.search") as search_mock, \
             patch("research_kb_mcp.tools.search.format_search_results") as format_mock:
            search_mock.return_value = mock_search_response
            format_mock.return_value = "## Search Results"

            await mcp.tools["research_kb_search"]["func"](
                query="test",
                limit=0,  # below min
            )

            call_args = search_mock.call_args[0][0]
            assert call_args.limit == 1  # clamped

    @pytest.mark.asyncio
    async def test_search_citation_weight_clamping(self, mock_search_response):
        """Search clamps citation weight to 0-1 range."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with patch("research_kb_mcp.tools.search.search") as search_mock, \
             patch("research_kb_mcp.tools.search.format_search_results") as format_mock:
            search_mock.return_value = mock_search_response
            format_mock.return_value = "## Search Results"

            await mcp.tools["research_kb_search"]["func"](
                query="test",
                citation_weight=1.5,  # exceeds max
            )

            call_args = search_mock.call_args[0][0]
            assert call_args.citation_weight == 1.0  # clamped

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """Search handles empty results gracefully."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        empty_response = MockSearchResponse(
            query="nonexistent topic xyz",
            results=[],
            total_count=0,
        )

        with patch("research_kb_mcp.tools.search.search") as search_mock, \
             patch("research_kb_mcp.tools.search.format_search_results") as format_mock:
            search_mock.return_value = empty_response
            format_mock.return_value = "## Search Results\n\nNo results found."

            result = await mcp.tools["research_kb_search"]["func"](
                query="nonexistent topic xyz",
            )

            # Should return something indicating no results
            assert result is not None
            assert isinstance(result, str)
