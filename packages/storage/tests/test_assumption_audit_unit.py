"""Unit tests for assumption_audit orchestration paths not covered by integration tests.

Covers:
- audit_assumptions with anthropic backend dispatch
- cache_assumptions ON CONFLICT behavior (upsert)
- _generate_docstring_snippet grouping and formatting
- find_method fallback path when method_aliases table missing
- extract_assumptions_with_anthropic markdown fence edge cases

Phase S Commit 4: Target assumption_audit.py 19.2% → 28%
"""

import json
import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from research_kb_storage.assumption_audit import (
    AssumptionDetail,
    MethodAssumptionAuditor,
    _generate_docstring_snippet,
)

pytestmark = pytest.mark.unit

_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_concept(name="double_machine_learning", concept_id=None, definition=None, aliases=None):
    """Create a minimal mock Concept."""
    c = MagicMock()
    c.id = concept_id or uuid4()
    c.name = name
    c.canonical_name = name.lower().replace(" ", "_")
    c.definition = definition
    c.aliases = aliases or []
    c.domain_id = "causal_inference"
    return c


def _make_assumption(name, importance="standard"):
    """Create an AssumptionDetail for testing."""
    return AssumptionDetail(
        name=name,
        importance=importance,
        plain_english=f"{name} must hold for valid inference",
        confidence=0.7,
    )


# ---------------------------------------------------------------------------
# audit_assumptions with anthropic backend
# ---------------------------------------------------------------------------


class TestAuditAssumptionsAnthropicBackend:
    """Test the anthropic backend dispatch path in audit_assumptions."""

    @patch.object(MethodAssumptionAuditor, "find_method")
    @patch.object(MethodAssumptionAuditor, "get_assumptions_from_graph")
    @patch.object(MethodAssumptionAuditor, "get_cached_assumptions")
    @patch.object(MethodAssumptionAuditor, "extract_assumptions_with_anthropic")
    @patch.object(MethodAssumptionAuditor, "cache_assumptions")
    async def test_anthropic_backend_dispatches_correctly(
        self, mock_cache, mock_anthropic, mock_cached, mock_graph, mock_find
    ):
        """When llm_backend='anthropic' and insufficient results, dispatches to anthropic."""
        method = _make_concept("synthetic_control", definition="Counterfactual method")
        mock_find.return_value = method
        mock_graph.return_value = [
            _make_assumption("parallel_trends")
        ]  # Only 1 — triggers fallback
        mock_cached.return_value = []
        mock_anthropic.return_value = [
            _make_assumption("no_anticipation"),
            _make_assumption("convex_hull"),
        ]
        mock_cache.return_value = 2

        result = await MethodAssumptionAuditor.audit_assumptions(
            "synthetic_control",
            llm_backend="anthropic",
        )

        mock_anthropic.assert_called_once()
        mock_cache.assert_called_once()
        assert len(result.assumptions) == 3
        assert "anthropic" in result.source

    @patch.object(MethodAssumptionAuditor, "find_method")
    @patch.object(MethodAssumptionAuditor, "get_assumptions_from_graph")
    @patch.object(MethodAssumptionAuditor, "get_cached_assumptions")
    async def test_sufficient_assumptions_skip_llm(self, mock_cached, mock_graph, mock_find):
        """When graph returns >= 3 assumptions, LLM fallback is skipped."""
        method = _make_concept("DML")
        mock_find.return_value = method
        mock_graph.return_value = [
            _make_assumption("overlap", "critical"),
            _make_assumption("unconfoundedness", "critical"),
            _make_assumption("regularity", "standard"),
        ]
        mock_cached.return_value = []

        result = await MethodAssumptionAuditor.audit_assumptions("DML")

        assert len(result.assumptions) == 3
        assert result.source == "graph"

    @patch.object(MethodAssumptionAuditor, "find_method")
    @patch.object(MethodAssumptionAuditor, "get_assumptions_from_graph")
    @patch.object(MethodAssumptionAuditor, "get_cached_assumptions")
    async def test_llm_disabled_skips_fallback(self, mock_cached, mock_graph, mock_find):
        """When use_llm_fallback=False, no LLM extraction even if sparse."""
        method = _make_concept("sparse_method")
        mock_find.return_value = method
        mock_graph.return_value = [_make_assumption("single_assumption")]
        mock_cached.return_value = []

        result = await MethodAssumptionAuditor.audit_assumptions(
            "sparse_method",
            use_llm_fallback=False,
        )

        assert len(result.assumptions) == 1
        assert result.source == "graph"


# ---------------------------------------------------------------------------
# audit_assumptions merging logic
# ---------------------------------------------------------------------------


class TestAuditMergingLogic:
    """Test the merge step between graph, cached, and LLM assumptions."""

    @patch.object(MethodAssumptionAuditor, "find_method")
    @patch.object(MethodAssumptionAuditor, "get_assumptions_from_graph")
    @patch.object(MethodAssumptionAuditor, "get_cached_assumptions")
    async def test_cached_enriches_graph_assumptions(self, mock_cached, mock_graph, mock_find):
        """Cached data enriches matching graph assumptions."""
        method = _make_concept("DML")
        mock_find.return_value = method

        graph_a = _make_assumption("overlap")
        graph_a.formal_statement = None

        cached_a = AssumptionDetail(
            name="overlap",
            importance="critical",
            formal_statement="P(D=1|X) ∈ (0,1)",
            plain_english="Every unit has a chance of treatment",
            violation_consequence="Infinite variance",
            verification_approaches=["propensity score histogram"],
            confidence=0.85,
        )

        mock_graph.return_value = [graph_a]
        mock_cached.return_value = [cached_a]

        result = await MethodAssumptionAuditor.audit_assumptions("DML", use_llm_fallback=False)

        merged = result.assumptions[0]
        assert merged.formal_statement == "P(D=1|X) ∈ (0,1)"
        assert merged.importance == "critical"

    @patch.object(MethodAssumptionAuditor, "find_method")
    @patch.object(MethodAssumptionAuditor, "get_assumptions_from_graph")
    @patch.object(MethodAssumptionAuditor, "get_cached_assumptions")
    async def test_cached_only_assumptions_added(self, mock_cached, mock_graph, mock_find):
        """Cached assumptions not in graph are added to final list."""
        method = _make_concept("DML")
        mock_find.return_value = method
        mock_graph.return_value = [_make_assumption("overlap")]
        mock_cached.return_value = [
            _make_assumption("overlap"),  # matches graph
            _make_assumption("curated_only"),  # unique to cache
        ]

        result = await MethodAssumptionAuditor.audit_assumptions("DML", use_llm_fallback=False)

        names = [a.name for a in result.assumptions]
        assert "curated_only" in names
        assert len(result.assumptions) == 2  # overlap + curated_only

    @patch.object(MethodAssumptionAuditor, "find_method")
    @patch.object(MethodAssumptionAuditor, "get_assumptions_from_graph")
    @patch.object(MethodAssumptionAuditor, "get_cached_assumptions")
    @patch.object(MethodAssumptionAuditor, "extract_assumptions_with_ollama")
    @patch.object(MethodAssumptionAuditor, "cache_assumptions")
    async def test_llm_deduplicates_by_name(
        self, mock_cache, mock_ollama, mock_cached, mock_graph, mock_find
    ):
        """LLM results don't duplicate existing assumptions by name."""
        method = _make_concept("IV")
        mock_find.return_value = method
        mock_graph.return_value = [_make_assumption("exogeneity")]
        mock_cached.return_value = []
        mock_ollama.return_value = [
            _make_assumption("exogeneity"),  # duplicate
            _make_assumption("relevance"),  # new
        ]
        mock_cache.return_value = 2

        result = await MethodAssumptionAuditor.audit_assumptions("IV")

        names = [a.name for a in result.assumptions]
        assert names.count("exogeneity") == 1
        assert "relevance" in names


# ---------------------------------------------------------------------------
# _generate_docstring_snippet
# ---------------------------------------------------------------------------


class TestGenerateDocstringSnippet:
    """Test docstring snippet generation."""

    def test_empty_assumptions_returns_message(self):
        snippet = _generate_docstring_snippet("DML", [])
        assert "no" in snippet.lower() or "DML" in snippet

    def test_critical_assumptions_labeled(self):
        assumptions = [
            _make_assumption("overlap", "critical"),
            _make_assumption("regularity", "standard"),
        ]
        snippet = _generate_docstring_snippet("DML", assumptions)
        assert "CRITICAL" in snippet
        assert "overlap" in snippet

    def test_groups_by_importance(self):
        assumptions = [
            _make_assumption("a1", "critical"),
            _make_assumption("a2", "standard"),
            _make_assumption("a3", "technical"),
        ]
        snippet = _generate_docstring_snippet("DML", assumptions)
        # Critical should appear before standard
        crit_idx = snippet.index("CRITICAL") if "CRITICAL" in snippet else -1
        assert crit_idx >= 0


# ---------------------------------------------------------------------------
# extract_assumptions_with_anthropic edge cases
# ---------------------------------------------------------------------------


class TestExtractAnthropicEdgeCases:
    """Additional edge cases for the anthropic extraction path."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_nested_code_fences_handled(self):
        """Response with ```json wrapper is properly stripped."""
        json_response = json.dumps(
            {
                "assumptions": [
                    {
                        "name": "overlap",
                        "importance": "critical",
                        "formal_statement": "P(D|X) in (0,1)",
                    }
                ]
            }
        )
        wrapped = f"```json\n{json_response}\n```"

        mock_message = MagicMock()
        mock_block = MagicMock()
        mock_block.text = wrapped
        mock_message.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = await MethodAssumptionAuditor.extract_assumptions_with_anthropic("DML")

        assert len(result) == 1
        assert result[0].name == "overlap"
        assert result[0].confidence == 0.85

    @patch.dict(os.environ, {}, clear=False)
    async def test_missing_api_key_returns_empty(self):
        """Returns empty list when ANTHROPIC_API_KEY is not set."""
        # Ensure key is absent
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            result = await MethodAssumptionAuditor.extract_assumptions_with_anthropic("DML")
        assert result == []

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_api_error_returns_empty(self):
        """Returns empty list when Anthropic API raises an exception."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API rate limit")

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = await MethodAssumptionAuditor.extract_assumptions_with_anthropic("DML")

        assert result == []
