"""Tests for MCP tools."""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock

from research_kb_mcp.tools.search import register_search_tools
from research_kb_mcp.tools.sources import register_source_tools
from research_kb_mcp.tools.concepts import register_concept_tools
from research_kb_mcp.tools.graph import register_graph_tools
from research_kb_mcp.tools.health import register_health_tools


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


class TestToolRegistration:
    """Tests for tool registration."""

    def test_search_tools_registered(self):
        """Search tools are registered correctly."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        assert "research_kb_search" in mcp.tools
        # Check docstring is present
        assert mcp.tools["research_kb_search"]["func"].__doc__ is not None

    def test_source_tools_registered(self):
        """Source tools are registered correctly."""
        mcp = MockFastMCP()
        register_source_tools(mcp)

        assert "research_kb_list_sources" in mcp.tools
        assert "research_kb_get_source" in mcp.tools
        assert "research_kb_get_source_citations" in mcp.tools

    def test_concept_tools_registered(self):
        """Concept tools are registered correctly."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        assert "research_kb_list_concepts" in mcp.tools
        assert "research_kb_get_concept" in mcp.tools

    def test_graph_tools_registered(self):
        """Graph tools are registered correctly."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        assert "research_kb_graph_neighborhood" in mcp.tools
        assert "research_kb_graph_path" in mcp.tools

    def test_health_tools_registered(self):
        """Health tools are registered correctly."""
        mcp = MockFastMCP()
        register_health_tools(mcp)

        assert "research_kb_stats" in mcp.tools
        assert "research_kb_health" in mcp.tools

    def test_all_tools_have_docstrings(self):
        """All registered tools have docstrings for MCP schema."""
        mcp = MockFastMCP()

        register_search_tools(mcp)
        register_source_tools(mcp)
        register_concept_tools(mcp)
        register_graph_tools(mcp)
        register_health_tools(mcp)

        for name, tool in mcp.tools.items():
            assert tool["func"].__doc__, f"Tool {name} missing docstring"


class TestToolDocstrings:
    """Tests for tool documentation."""

    def test_search_tool_has_examples(self):
        """Search tool docstring includes examples."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        doc = mcp.tools["research_kb_search"]["func"].__doc__
        assert "Example" in doc
        assert "instrumental variables" in doc

    def test_concept_tool_documents_types(self):
        """Concept tool docstring documents types."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        doc = mcp.tools["research_kb_list_concepts"]["func"].__doc__
        assert "METHOD" in doc
        assert "ASSUMPTION" in doc

    def test_graph_neighborhood_explains_hops(self):
        """Graph neighborhood docstring explains hops parameter."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        doc = mcp.tools["research_kb_graph_neighborhood"]["func"].__doc__
        assert "hops" in doc
        assert "1-3" in doc


# NOTE: Integration tests with mocked service layer are complex due to
# async context managers and nested mocks. The service layer itself is
# tested in packages/api/tests/. These tests focus on:
# 1. Tool registration (TestToolRegistration) - verified
# 2. Docstring presence (test_all_tools_have_docstrings) - verified
# 3. Formatter logic (test_formatters.py) - verified
#
# End-to-end testing should be done via:
#   python -m research_kb_mcp.server  # start server
#   # Then connect with MCP client
