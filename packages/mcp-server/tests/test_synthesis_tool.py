"""Tests for MCP explain_connection tool and formatters.

Phase AC: Validates tool registration, JSON/markdown output, error handling,
and parameter passthrough for research_kb_explain_connection (Tool #21).
"""

from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

import pytest

from research_kb_mcp.tools.graph import register_graph_tools
from research_kb_storage.synthesis import (
    ConnectionExplanation,
    EvidenceChunk,
    PathStep,
)
from research_kb_mcp.formatters import (
    format_connection_explanation,
    format_connection_explanation_json,
)

pytestmark = pytest.mark.unit


# ─── Shared Test Infrastructure ──────────────────────────────────────────────


class MockFastMCP:
    """Mock FastMCP server for testing tool registration."""

    def __init__(self):
        self.tools = {}

    def tool(self, **kwargs):
        def decorator(func):
            self.tools[func.__name__] = {"func": func, "kwargs": kwargs}
            return func

        return decorator


def _make_connection_result(
    source: str = "graph+llm",
    with_evidence: bool = True,
    with_synthesis: bool = True,
) -> ConnectionExplanation:
    """Create a ConnectionExplanation for testing."""
    ev = (
        [
            EvidenceChunk(
                chunk_id=uuid4(),
                source_id=uuid4(),
                source_title="Chernozhukov et al. (2018)",
                content="DML requires sample splitting via cross-fitting...",
                page_start=42,
                page_end=43,
                concept_name="double machine learning",
                mention_type="defines",
            )
        ]
        if with_evidence
        else []
    )
    return ConnectionExplanation(
        concept_a="double machine learning",
        concept_b="cross-fitting",
        path=[
            PathStep(
                concept_id=uuid4(),
                concept_name="double machine learning",
                concept_type="method",
                definition="A method for causal inference using machine learning",
                relationship_to_next="requires",
                evidence=ev,
            ),
            PathStep(
                concept_id=uuid4(),
                concept_name="cross-fitting",
                concept_type="technique",
                definition="Sample splitting to avoid overfitting bias",
                evidence=[],
            ),
        ],
        path_length=1,
        path_explanation="double machine learning -> (requires) -> cross-fitting",
        synthesis="DML depends on cross-fitting because..." if with_synthesis else None,
        synthesis_style="educational",
        source=source,
        confidence=0.85 if with_synthesis else 1.0,
        evidence_count=len(ev),
    )


# ─── Tool Registration ──────────────────────────────────────────────────────


class TestToolRegistration:
    def test_tool_registered(self):
        """research_kb_explain_connection is registered as a tool."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)
        assert "research_kb_explain_connection" in mcp.tools

    def test_existing_tools_preserved(self):
        """Other graph tools still registered alongside explain_connection."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)
        assert "research_kb_graph_neighborhood" in mcp.tools
        assert "research_kb_graph_path" in mcp.tools
        assert "research_kb_cross_domain_concepts" in mcp.tools


# ─── JSON Output ─────────────────────────────────────────────────────────────


class TestJsonOutput:
    def test_json_output_valid(self):
        """output_format='json' returns parseable JSON."""
        result = _make_connection_result()
        json_str = format_connection_explanation_json(result)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_json_structure(self):
        """JSON output has correct top-level keys."""
        result = _make_connection_result()
        parsed = json.loads(format_connection_explanation_json(result))

        assert parsed["concept_a"] == "double machine learning"
        assert parsed["concept_b"] == "cross-fitting"
        assert len(parsed["path"]) == 2
        assert parsed["synthesis"] is not None
        assert parsed["source"] == "graph+llm"
        assert parsed["path_length"] == 1
        assert parsed["confidence"] == 0.85
        assert parsed["evidence_count"] == 1

    def test_json_path_step_structure(self):
        """Each path step has expected fields."""
        result = _make_connection_result()
        parsed = json.loads(format_connection_explanation_json(result))
        step = parsed["path"][0]

        assert "concept_id" in step
        assert step["concept_name"] == "double machine learning"
        assert step["concept_type"] == "method"
        assert "definition" in step
        assert step["relationship_to_next"] == "requires"
        assert len(step["evidence"]) == 1

    def test_json_evidence_structure(self):
        """Evidence chunks in JSON have correct fields."""
        result = _make_connection_result()
        parsed = json.loads(format_connection_explanation_json(result))
        ev = parsed["path"][0]["evidence"][0]

        assert "chunk_id" in ev
        assert "source_id" in ev
        assert ev["source_title"] == "Chernozhukov et al. (2018)"
        assert ev["page_start"] == 42
        assert ev["concept_name"] == "double machine learning"


# ─── Markdown Output ─────────────────────────────────────────────────────────


class TestMarkdownOutput:
    def test_markdown_has_sections(self):
        """Markdown output contains key sections."""
        result = _make_connection_result()
        md = format_connection_explanation(result)

        assert "## Connection:" in md
        assert "### Step 1:" in md
        assert "### Step 2:" in md
        assert "### Synthesis" in md

    def test_markdown_evidence_cited(self):
        """Markdown includes evidence citations."""
        result = _make_connection_result()
        md = format_connection_explanation(result)
        assert "Chernozhukov et al. (2018)" in md
        assert "p. 42" in md

    def test_no_synthesis_omits_section(self):
        """Without synthesis, no Synthesis section in markdown."""
        result = _make_connection_result(with_synthesis=False, source="graph_only")
        md = format_connection_explanation(result)
        assert "### Synthesis" not in md


# ─── Error Handling ──────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_no_path_markdown(self):
        """source='no_path' produces appropriate message."""
        result = ConnectionExplanation(
            concept_a="A",
            concept_b="B",
            path=[],
            path_length=0,
            path_explanation="No path found between A and B",
            source="no_path",
            confidence=0.0,
        )
        md = format_connection_explanation(result)
        assert "No path found" in md
        assert "research_kb_graph_neighborhood" in md

    def test_error_markdown(self):
        """source='error' produces error message."""
        result = ConnectionExplanation(
            concept_a="nonexistent",
            concept_b="also",
            path=[],
            path_length=0,
            path_explanation="Concept not found: nonexistent",
            source="error",
            confidence=0.0,
        )
        md = format_connection_explanation(result)
        assert "Error" in md
        assert "not found" in md

    def test_no_path_json(self):
        """source='no_path' in JSON has correct structure."""
        result = ConnectionExplanation(
            concept_a="A",
            concept_b="B",
            path=[],
            path_length=0,
            path_explanation="No path found",
            source="no_path",
            confidence=0.0,
        )
        parsed = json.loads(format_connection_explanation_json(result))
        assert parsed["source"] == "no_path"
        assert parsed["path"] == []


# ─── Style Passthrough ───────────────────────────────────────────────────────


class TestStylePassthrough:
    @pytest.mark.parametrize("style", ["educational", "research", "implementation"])
    def test_style_in_markdown(self, style):
        """All 3 styles appear correctly in markdown output."""
        result = _make_connection_result()
        result.synthesis_style = style
        md = format_connection_explanation(result)
        assert f"### Synthesis ({style})" in md

    @pytest.mark.parametrize("style", ["educational", "research", "implementation"])
    def test_style_in_json(self, style):
        """All 3 styles serialize correctly in JSON."""
        result = _make_connection_result()
        result.synthesis_style = style
        parsed = json.loads(format_connection_explanation_json(result))
        assert parsed["synthesis_style"] == style


# ─── Tool Integration ────────────────────────────────────────────────────────


class TestToolIntegration:
    @patch("research_kb_mcp.tools.graph.explain_connection")
    async def test_tool_calls_explain_connection(self, mock_ec):
        """Tool passes parameters through to explain_connection."""
        mock_ec.return_value = _make_connection_result()

        mcp = MockFastMCP()
        register_graph_tools(mcp)
        tool_func = mcp.tools["research_kb_explain_connection"]["func"]

        result = await tool_func(
            concept_a="DML",
            concept_b="cross-fitting",
            style="research",
            max_evidence_per_step=3,
            use_llm=False,
            output_format="json",
        )

        mock_ec.assert_called_once_with(
            concept_a="DML",
            concept_b="cross-fitting",
            style="research",
            max_evidence_per_step=3,
            use_llm=False,
        )
        # Output should be valid JSON
        parsed = json.loads(result)
        assert parsed["concept_a"] == "double machine learning"

    @patch("research_kb_mcp.tools.graph.explain_connection")
    async def test_tool_max_evidence_clamped(self, mock_ec):
        """max_evidence_per_step outside 1-3 is clamped."""
        mock_ec.return_value = _make_connection_result()

        mcp = MockFastMCP()
        register_graph_tools(mcp)
        tool_func = mcp.tools["research_kb_explain_connection"]["func"]

        await tool_func(
            concept_a="A",
            concept_b="B",
            max_evidence_per_step=10,
            output_format="markdown",
        )

        # Should be clamped to 3
        call_kwargs = mock_ec.call_args
        assert call_kwargs.kwargs["max_evidence_per_step"] == 3

    @patch("research_kb_mcp.tools.graph.explain_connection")
    async def test_tool_error_returns_message(self, mock_ec):
        """Tool exceptions are caught and return error message."""
        mock_ec.side_effect = Exception("Database unavailable")

        mcp = MockFastMCP()
        register_graph_tools(mcp)
        tool_func = mcp.tools["research_kb_explain_connection"]["func"]

        result = await tool_func(concept_a="A", concept_b="B")
        assert "Error" in result
        assert "Database unavailable" in result
