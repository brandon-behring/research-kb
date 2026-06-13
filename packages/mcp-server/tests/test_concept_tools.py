"""Tests for concept MCP tools — RETIRED stubs (RS4).

The chunk-level concept layer was retired in slice RS4 (decision R3, ADR-0001);
the concept data is preserved (dormant) but no longer exposed. These tools remain
registered as fail-loud tombstones redirecting to synthesis-kb.
"""

from __future__ import annotations

import pytest

from research_kb_mcp.tools.concepts import register_concept_tools

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


CONCEPT_TOOLS = [
    "research_kb_list_concepts",
    "research_kb_get_concept",
    "research_kb_chunk_concepts",
    "research_kb_find_similar_concepts",
]


def test_all_concept_tools_registered():
    """All four concept tools stay registered as retirement tombstones."""
    mcp = MockFastMCP()
    register_concept_tools(mcp)
    for name in CONCEPT_TOOLS:
        assert name in mcp.tools, f"{name} should stay registered as a retirement stub"


@pytest.mark.parametrize("name", CONCEPT_TOOLS)
def test_concept_tool_docstring_marks_retired(name):
    """Each tool's docstring fails loud: marked RETIRED."""
    mcp = MockFastMCP()
    register_concept_tools(mcp)
    doc = mcp.tools[name]["func"].__doc__ or ""
    assert "RETIRED" in doc


async def test_list_concepts_returns_retirement_notice():
    mcp = MockFastMCP()
    register_concept_tools(mcp)
    result = await mcp.tools["research_kb_list_concepts"]["func"]()
    assert "RETIRED" in result
    assert "synthesis-kb" in result


async def test_get_concept_returns_retirement_notice():
    mcp = MockFastMCP()
    register_concept_tools(mcp)
    result = await mcp.tools["research_kb_get_concept"]["func"](concept_id="x")
    assert "RETIRED" in result
    assert "synthesis-kb" in result


async def test_chunk_concepts_returns_retirement_notice():
    mcp = MockFastMCP()
    register_concept_tools(mcp)
    result = await mcp.tools["research_kb_chunk_concepts"]["func"](chunk_id="x")
    assert "RETIRED" in result
    assert "synthesis-kb" in result


async def test_find_similar_concepts_returns_retirement_notice():
    mcp = MockFastMCP()
    register_concept_tools(mcp)
    result = await mcp.tools["research_kb_find_similar_concepts"]["func"](concept_id="x")
    assert "RETIRED" in result
    assert "synthesis-kb" in result
