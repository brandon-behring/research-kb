"""Tests for graph MCP tools — RETIRED stubs (RS4).

The chunk-level concept graph was retired in slice RS4 (decision R3, ADR-0001).
These tools remain registered as fail-loud tombstones; the tests assert the
retirement contract: still registered, docstring marked RETIRED, body returns a
redirect to synthesis-kb (never a silently-empty graph result).
"""

from __future__ import annotations

import pytest

from research_kb_mcp.tools.graph import register_graph_tools

pytestmark = pytest.mark.unit


class MockFastMCP:
    """Mock FastMCP server capturing registered tool functions."""

    def __init__(self):
        self.tools = {}

    def tool(self, **kwargs):
        def decorator(func):
            self.tools[func.__name__] = {"func": func, "kwargs": kwargs}
            return func

        return decorator


GRAPH_TOOLS = [
    "research_kb_graph_neighborhood",
    "research_kb_graph_path",
    "research_kb_cross_domain_concepts",
    "research_kb_explain_connection",
]


def test_all_graph_tools_registered():
    """All four graph tools stay registered as retirement tombstones."""
    mcp = MockFastMCP()
    register_graph_tools(mcp)
    for name in GRAPH_TOOLS:
        assert name in mcp.tools, f"{name} should stay registered as a retirement stub"


@pytest.mark.parametrize("name", GRAPH_TOOLS)
def test_graph_tool_docstring_marks_retired(name):
    """Each tool's docstring fails loud: marked RETIRED."""
    mcp = MockFastMCP()
    register_graph_tools(mcp)
    doc = mcp.tools[name]["func"].__doc__ or ""
    assert "RETIRED" in doc


async def test_graph_neighborhood_returns_retirement_notice():
    mcp = MockFastMCP()
    register_graph_tools(mcp)
    result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
        concept_name="double machine learning",
    )
    assert "RETIRED" in result
    assert "synthesis-kb" in result


async def test_graph_path_returns_retirement_notice():
    mcp = MockFastMCP()
    register_graph_tools(mcp)
    result = await mcp.tools["research_kb_graph_path"]["func"](
        concept_a="A",
        concept_b="B",
    )
    assert "RETIRED" in result
    assert "synthesis-kb" in result


async def test_cross_domain_returns_retirement_notice():
    mcp = MockFastMCP()
    register_graph_tools(mcp)
    result = await mcp.tools["research_kb_cross_domain_concepts"]["func"](
        source_domain="causal_inference",
        target_domain="time_series",
    )
    assert "RETIRED" in result
    assert "synthesis-kb" in result


async def test_explain_connection_returns_retirement_notice():
    mcp = MockFastMCP()
    register_graph_tools(mcp)
    result = await mcp.tools["research_kb_explain_connection"]["func"](
        concept_a="A",
        concept_b="B",
    )
    assert "RETIRED" in result
    assert "synthesis-kb" in result
