"""Tests for graph MCP tools."""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock
from uuid import uuid4

from research_kb_mcp.tools.graph import register_graph_tools


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


class TestGraphToolRegistration:
    """Tests for graph tool registration."""

    def test_neighborhood_registered(self):
        """Graph neighborhood tool is registered correctly."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        assert "research_kb_graph_neighborhood" in mcp.tools
        doc = mcp.tools["research_kb_graph_neighborhood"]["func"].__doc__
        assert doc is not None
        assert "neighborhood" in doc.lower()
        assert "hops" in doc.lower()

    def test_path_registered(self):
        """Graph path tool is registered correctly."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        assert "research_kb_graph_path" in mcp.tools
        doc = mcp.tools["research_kb_graph_path"]["func"].__doc__
        assert doc is not None
        assert "path" in doc.lower()


class TestGraphNeighborhood:
    """Tests for graph neighborhood tool functionality."""

    @pytest.fixture
    def sample_neighborhood(self):
        """Create sample neighborhood data for testing."""
        return {
            "center": {
                "id": str(uuid4()),
                "name": "Double Machine Learning",
                "type": "METHOD",
            },
            "concepts": [
                {
                    "id": str(uuid4()),
                    "name": "Neyman Orthogonality",
                    "type": "THEOREM",
                    "distance": 1,
                },
                {
                    "id": str(uuid4()),
                    "name": "Cross-Fitting",
                    "type": "TECHNIQUE",
                    "distance": 1,
                },
            ],
            "relationships": [
                {"source": "Double Machine Learning", "target": "Neyman Orthogonality", "type": "USES"},
                {"source": "Double Machine Learning", "target": "Cross-Fitting", "type": "USES"},
            ],
        }

    @pytest.mark.asyncio
    async def test_neighborhood_default_hops(self, sample_neighborhood):
        """Neighborhood uses default hop count."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as get_mock:
            get_mock.return_value = sample_neighborhood

            result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="Double Machine Learning",
            )

            get_mock.assert_called_once_with(
                concept_name="Double Machine Learning",
                hops=2,  # default
                limit=50,  # default
            )
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_neighborhood_custom_hops(self, sample_neighborhood):
        """Neighborhood respects custom hop count."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as get_mock:
            get_mock.return_value = sample_neighborhood

            await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="DML",
                hops=1,
                limit=20,
            )

            get_mock.assert_called_once_with(
                concept_name="DML",
                hops=1,
                limit=20,
            )

    @pytest.mark.asyncio
    async def test_neighborhood_hops_clamping(self, sample_neighborhood):
        """Neighborhood clamps hops to valid range."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as get_mock:
            get_mock.return_value = sample_neighborhood

            # Test upper bound
            await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="test",
                hops=10,
            )
            assert get_mock.call_args[1]["hops"] == 3  # clamped to max

            # Test lower bound
            await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="test",
                hops=0,
            )
            assert get_mock.call_args[1]["hops"] == 1  # clamped to min

    @pytest.mark.asyncio
    async def test_neighborhood_limit_clamping(self, sample_neighborhood):
        """Neighborhood clamps limit to valid range."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as get_mock:
            get_mock.return_value = sample_neighborhood

            await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="test",
                limit=200,
            )
            assert get_mock.call_args[1]["limit"] == 100

    @pytest.mark.asyncio
    async def test_neighborhood_not_found(self):
        """Neighborhood handles concept not found."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as get_mock:
            # Use correct keys: nodes and edges (not concepts and relationships)
            get_mock.return_value = {"center": None, "nodes": [], "edges": []}

            result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="nonexistent concept xyz",
            )

            assert isinstance(result, str)


class TestGraphPath:
    """Tests for graph path tool functionality."""

    @pytest.fixture
    def sample_path(self):
        """Create sample path data for testing."""
        return {
            "found": True,
            "path": [
                {"id": str(uuid4()), "name": "Regression Discontinuity", "type": "METHOD"},
                {"id": str(uuid4()), "name": "Local Average Treatment Effect", "type": "DEFINITION"},
                {"id": str(uuid4()), "name": "Instrumental Variables", "type": "METHOD"},
            ],
            "length": 2,
        }

    @pytest.mark.asyncio
    async def test_path_found(self, sample_path):
        """Path returns formatted result when path exists."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_path") as get_mock:
            get_mock.return_value = sample_path

            result = await mcp.tools["research_kb_graph_path"]["func"](
                concept_a="Regression Discontinuity",
                concept_b="Instrumental Variables",
            )

            get_mock.assert_called_once_with(
                concept_a="Regression Discontinuity",
                concept_b="Instrumental Variables",
            )
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_path_no_connection(self):
        """Path handles no connection between concepts."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_path") as get_mock:
            get_mock.return_value = {"found": False, "path": [], "length": 0}

            result = await mcp.tools["research_kb_graph_path"]["func"](
                concept_a="Concept A",
                concept_b="Concept B",
            )

            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_path_same_concept(self):
        """Path handles same concept for both endpoints."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_path") as get_mock:
            get_mock.return_value = {
                "found": True,
                "path": [{"id": str(uuid4()), "name": "DML", "type": "METHOD"}],
                "length": 0,
            }

            result = await mcp.tools["research_kb_graph_path"]["func"](
                concept_a="DML",
                concept_b="DML",
            )

            assert isinstance(result, str)
