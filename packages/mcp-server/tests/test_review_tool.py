"""Tests for literature review MCP tool."""

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from research_kb_storage.literature_review import (
    LiteratureReview,
    ReviewEvidence,
    ReviewSection,
)

pytestmark = pytest.mark.unit


def _make_review(topic: str = "test topic") -> LiteratureReview:
    """Create a test LiteratureReview."""
    return LiteratureReview(
        topic=topic,
        introduction="Test introduction.",
        sections=[
            ReviewSection(
                section_id="overview",
                title="Overview",
                description="Overview section",
                concepts=["concept_a", "concept_b"],
                evidence=[
                    ReviewEvidence(
                        chunk_id=uuid4(),
                        source_id=uuid4(),
                        source_title="Test Source",
                        content="Evidence text",
                        page_start=1,
                        page_end=5,
                        relevance_score=0.9,
                    )
                ],
                synthesis="Synthesized overview text.",
            ),
            ReviewSection(
                section_id="methods",
                title="Methods",
                description="Methods section",
                concepts=["method_a"],
                synthesis="Synthesized methods text.",
            ),
        ],
        conclusion="Test conclusion.",
        quality_metrics={
            "sources_cited": 3,
            "concepts_covered": 5,
            "evidence_chunks": 8,
            "sections_total": 4,
            "sections_synthesized": 2,
        },
        style="educational",
    )


class TestReviewToolMarkdown:
    """Test research_kb_literature_review tool markdown output."""

    @pytest.mark.asyncio
    @patch("research_kb_mcp.tools.review.generate_literature_review")
    async def test_markdown_output(self, mock_gen):
        """Tool returns markdown by default."""
        mock_gen.return_value = _make_review("instrumental variables")

        from research_kb_mcp.tools.review import register_review_tools
        from unittest.mock import MagicMock

        mcp = MagicMock()
        captured_fn = None

        def capture_tool():
            def decorator(fn):
                nonlocal captured_fn
                captured_fn = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register_review_tools(mcp)

        # Call the captured function
        result = await captured_fn(
            topic="instrumental variables",
            output_format="markdown",
        )

        assert "# Literature Review: instrumental variables" in result
        assert "## Overview" in result
        assert "Synthesized overview text." in result

    @pytest.mark.asyncio
    @patch("research_kb_mcp.tools.review.generate_literature_review")
    async def test_json_output(self, mock_gen):
        """Tool returns valid JSON when format=json."""
        mock_gen.return_value = _make_review("DML")

        from research_kb_mcp.tools.review import register_review_tools
        from unittest.mock import MagicMock

        mcp = MagicMock()
        captured_fn = None

        def capture_tool():
            def decorator(fn):
                nonlocal captured_fn
                captured_fn = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register_review_tools(mcp)

        result = await captured_fn(
            topic="DML",
            output_format="json",
        )

        parsed = json.loads(result)
        assert parsed["topic"] == "DML"
        assert len(parsed["sections"]) == 2
        assert parsed["quality_metrics"]["sources_cited"] == 3


class TestReviewToolParams:
    """Test parameter forwarding to generate_literature_review."""

    @pytest.mark.asyncio
    @patch("research_kb_mcp.tools.review.generate_literature_review")
    async def test_params_forwarded(self, mock_gen):
        """Style, use_llm, max_concepts are forwarded."""
        mock_gen.return_value = _make_review()

        from research_kb_mcp.tools.review import register_review_tools
        from unittest.mock import MagicMock

        mcp = MagicMock()
        captured_fn = None

        def capture_tool():
            def decorator(fn):
                nonlocal captured_fn
                captured_fn = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register_review_tools(mcp)

        await captured_fn(
            topic="test",
            style="research",
            use_llm=False,
            max_concepts=15,
            max_evidence_per_section=4,
            output_format="markdown",
        )

        mock_gen.assert_called_once_with(
            topic="test",
            style="research",
            use_llm=False,
            max_concepts=15,
            max_evidence_per_section=4,
        )
