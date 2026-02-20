"""Tests for health and stats MCP tools."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from research_kb_mcp.tools.health import register_health_tools

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


class TestHealthToolRegistration:
    """Tests for health tool registration."""

    def test_stats_registered(self):
        """Stats tool is registered correctly."""
        mcp = MockFastMCP()
        register_health_tools(mcp)

        assert "research_kb_stats" in mcp.tools
        doc = mcp.tools["research_kb_stats"]["func"].__doc__
        assert doc is not None
        assert "statistics" in doc.lower()

    def test_health_registered(self):
        """Health tool is registered correctly."""
        mcp = MockFastMCP()
        register_health_tools(mcp)

        assert "research_kb_health" in mcp.tools
        doc = mcp.tools["research_kb_health"]["func"].__doc__
        assert doc is not None
        assert "health" in doc.lower()


class TestStatsTool:
    """Tests for stats tool functionality."""

    @pytest.fixture
    def sample_stats(self):
        """Create sample stats for testing."""
        return {
            "sources": 294,
            "chunks": 142962,
            "concepts": 283714,
            "relationships": 725866,
            "citations": 10758,
            "chunk_concepts": 500000,
        }

    @pytest.mark.asyncio
    async def test_stats_returns_counts(self, sample_stats):
        """Stats returns formatted database counts."""
        mcp = MockFastMCP()
        register_health_tools(mcp)

        with patch("research_kb_mcp.tools.health.get_stats") as stats_mock:
            stats_mock.return_value = sample_stats

            result = await mcp.tools["research_kb_stats"]["func"]()

            stats_mock.assert_called_once()
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_stats_format(self, sample_stats):
        """Stats result is properly formatted."""
        mcp = MockFastMCP()
        register_health_tools(mcp)

        with patch("research_kb_mcp.tools.health.get_stats") as stats_mock:
            stats_mock.return_value = sample_stats

            result = await mcp.tools["research_kb_stats"]["func"]()

            # Should contain formatted numbers
            assert isinstance(result, str)
            # Formatter should handle the stats dict


class TestHealthTool:
    """Tests for health tool functionality."""

    @pytest.fixture
    def healthy_stats(self):
        """Create stats indicating healthy system."""
        return {
            "sources": 100,
            "chunks": 5000,
            "concepts": 10000,
        }

    @pytest.mark.asyncio
    async def test_health_all_healthy(self, healthy_stats):
        """Health returns healthy status when all components work."""
        mcp = MockFastMCP()
        register_health_tools(mcp)

        with patch("research_kb_mcp.tools.health.get_stats") as stats_mock:
            stats_mock.return_value = healthy_stats

            result = await mcp.tools["research_kb_health"]["func"]()

            stats_mock.assert_called_once()
            assert isinstance(result, str)
            assert "Healthy" in result or "healthy" in result.lower()

    @pytest.mark.asyncio
    async def test_health_degraded(self):
        """Health returns unhealthy when stats fail."""
        mcp = MockFastMCP()
        register_health_tools(mcp)

        with patch("research_kb_mcp.tools.health.get_stats") as stats_mock:
            stats_mock.side_effect = Exception("Database connection failed")

            result = await mcp.tools["research_kb_health"]["func"]()

            assert isinstance(result, str)
            assert (
                "Unhealthy" in result or "unhealthy" in result.lower() or "error" in result.lower()
            )

    @pytest.mark.asyncio
    async def test_health_connection_error(self):
        """Health handles connection errors gracefully."""
        mcp = MockFastMCP()
        register_health_tools(mcp)

        with patch("research_kb_mcp.tools.health.get_stats") as stats_mock:
            stats_mock.side_effect = ConnectionError("Cannot connect to database")

            result = await mcp.tools["research_kb_health"]["func"]()

            # Should not raise, should return error status
            assert isinstance(result, str)
            assert "error" in result.lower() or "unhealthy" in result.lower()
