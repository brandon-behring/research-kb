"""Tests for the literature-review MCP tool — RETIRED stub (RS4).

literature_review seeded from the retired chunk-level concept graph (decision R3,
ADR-0001) and produced silently-weak reviews for domains without extracted
concepts. Retired as a fail-loud tombstone; these tests assert the contract.
"""

from __future__ import annotations

import pytest

from research_kb_mcp.tools.review import register_review_tools

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


def test_literature_review_registered():
    """The tool stays registered as a retirement tombstone with a RETIRED docstring."""
    mcp = MockFastMCP()
    register_review_tools(mcp)
    assert "research_kb_literature_review" in mcp.tools
    doc = mcp.tools["research_kb_literature_review"]["func"].__doc__ or ""
    assert "RETIRED" in doc


async def test_literature_review_returns_retirement_notice():
    mcp = MockFastMCP()
    register_review_tools(mcp)
    result = await mcp.tools["research_kb_literature_review"]["func"](
        topic="instrumental variables"
    )
    assert "RETIRED" in result
    assert "synthesis-kb" in result
