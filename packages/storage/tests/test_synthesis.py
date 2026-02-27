"""Unit tests for synthesis.py — connection explanation via graph + evidence + LLM.

Covers:
- hydrate_path_with_evidence waterfall logic (evidence_chunk_ids → defines → any)
- synthesize_connection Anthropic API call pattern
- explain_connection orchestrator with all degradation modes
- ConnectionExplanation.to_dict() serialization

Phase AC: explain_connection (Tier 2 Synthesis)
"""

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from research_kb_storage.synthesis import (
    ConnectionExplanation,
    EvidenceChunk,
    PathStep,
    _build_evidence_sections,
    _truncate_content,
    hydrate_path_with_evidence,
    synthesize_connection,
    explain_connection,
)

pytestmark = pytest.mark.unit

_NOW = datetime.now(timezone.utc)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_concept(name="double_machine_learning", concept_id=None, definition=None):
    """Create a mock Concept object."""
    c = MagicMock()
    c.id = concept_id or uuid4()
    c.name = name
    c.canonical_name = name.lower().replace(" ", "_")
    c.definition = definition or f"Definition of {name}"
    # Use a mock for concept_type so .value is settable
    ct = MagicMock()
    ct.value = "method"
    c.concept_type = ct
    return c


def _make_relationship(source_id=None, target_id=None, evidence_ids=None):
    """Create a mock ConceptRelationship."""
    r = MagicMock()
    r.id = uuid4()
    r.source_concept_id = source_id or uuid4()
    r.target_concept_id = target_id or uuid4()
    rt = MagicMock()
    rt.value = "requires"
    r.relationship_type = rt
    r.evidence_chunk_ids = evidence_ids or []
    r.created_at = _NOW
    return r


def _make_chunk(chunk_id=None, source_id=None, content="Sample evidence text"):
    """Create a mock Chunk object."""
    c = MagicMock()
    c.id = chunk_id or uuid4()
    c.source_id = source_id or uuid4()
    c.content = content
    c.page_start = 42
    c.page_end = 43
    return c


def _make_source(source_id=None, title="Chernozhukov et al. (2018)"):
    """Create a mock Source object."""
    s = MagicMock()
    s.id = source_id or uuid4()
    s.title = title
    return s


def _make_chunk_concept(chunk_id=None, concept_id=None, mention_type="reference"):
    """Create a mock ChunkConcept."""
    cc = MagicMock()
    cc.chunk_id = chunk_id or uuid4()
    cc.concept_id = concept_id or uuid4()
    cc.mention_type = mention_type
    cc.relevance_score = 0.9
    return cc


# ─── _truncate_content ───────────────────────────────────────────────────────


class TestTruncateContent:
    def test_short_text_unchanged(self):
        assert _truncate_content("hello") == "hello"

    def test_long_text_truncated(self):
        text = "x" * 400
        result = _truncate_content(text, max_chars=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_exact_boundary_unchanged(self):
        text = "x" * 300
        assert _truncate_content(text) == text


# ─── _build_evidence_sections ────────────────────────────────────────────────


class TestBuildEvidenceSections:
    def test_builds_sections_with_evidence(self):
        ev = EvidenceChunk(
            chunk_id=uuid4(),
            source_id=uuid4(),
            source_title="Test Paper",
            content="Evidence content here",
            page_start=10,
            concept_name="IV",
        )
        step = PathStep(
            concept_id=uuid4(),
            concept_name="instrumental variables",
            concept_type="method",
            definition="A method for causal inference",
            relationship_to_next="requires",
            evidence=[ev],
        )
        result = _build_evidence_sections([step])
        assert "STEP 1: instrumental variables" in result
        assert "Test Paper, p. 10" in result
        assert "Evidence content here" in result

    def test_empty_evidence_section(self):
        step = PathStep(
            concept_id=uuid4(),
            concept_name="IV",
            concept_type="method",
        )
        result = _build_evidence_sections([step])
        assert "(none available)" in result


# ─── hydrate_path_with_evidence ──────────────────────────────────────────────


class TestHydratePathWithEvidence:
    @patch("research_kb_storage.source_store.SourceStore.get_by_id")
    @patch("research_kb_storage.chunk_store.ChunkStore.get_by_id")
    @patch("research_kb_storage.chunk_concept_store.ChunkConceptStore.list_chunks_for_concept")
    async def test_happy_path(self, mock_list_chunks, mock_chunk_get, mock_source_get):
        """Full waterfall: evidence_chunk_ids → defines → any."""
        chunk_id = uuid4()
        source_id = uuid4()
        concept_a = _make_concept("DML")
        concept_b = _make_concept("cross-fitting")
        rel = _make_relationship(evidence_ids=[chunk_id])

        chunk = _make_chunk(chunk_id=chunk_id, source_id=source_id)
        source = _make_source(source_id=source_id)

        mock_chunk_get.return_value = chunk
        mock_source_get.return_value = source
        mock_list_chunks.return_value = []

        path = [(concept_a, None), (concept_b, rel)]
        steps = await hydrate_path_with_evidence(path, max_evidence_per_step=2)

        assert len(steps) == 2
        assert steps[0].concept_name == "dml"
        assert steps[1].concept_name == "cross-fitting"
        # concept_b has evidence from relationship
        assert len(steps[1].evidence) == 1
        assert steps[1].evidence[0].chunk_id == chunk_id

    async def test_empty_path(self):
        """Empty path returns empty list."""
        steps = await hydrate_path_with_evidence([], max_evidence_per_step=2)
        assert steps == []

    @patch("research_kb_storage.source_store.SourceStore.get_by_id")
    @patch("research_kb_storage.chunk_store.ChunkStore.get_by_id")
    @patch("research_kb_storage.chunk_concept_store.ChunkConceptStore.list_chunks_for_concept")
    async def test_evidence_chunk_ids_used_first(
        self, mock_list_chunks, mock_chunk_get, mock_source_get
    ):
        """evidence_chunk_ids from relationship takes priority over defines."""
        ev_chunk_id = uuid4()
        source_id = uuid4()

        concept_a = _make_concept("A")
        concept_b = _make_concept("B")
        rel = _make_relationship(evidence_ids=[ev_chunk_id])

        chunk = _make_chunk(chunk_id=ev_chunk_id, source_id=source_id)
        source = _make_source(source_id=source_id)

        mock_chunk_get.return_value = chunk
        mock_source_get.return_value = source
        # defines lookup should not be reached (max_evidence_per_step=1)
        mock_list_chunks.return_value = []

        steps = await hydrate_path_with_evidence(
            [(concept_a, None), (concept_b, rel)],
            max_evidence_per_step=1,
        )
        assert len(steps[1].evidence) == 1
        assert steps[1].evidence[0].mention_type == "relationship_evidence"

    @patch("research_kb_storage.source_store.SourceStore.get_by_id")
    @patch("research_kb_storage.chunk_store.ChunkStore.get_by_id")
    @patch("research_kb_storage.chunk_concept_store.ChunkConceptStore.list_chunks_for_concept")
    async def test_defines_fallback(self, mock_list_chunks, mock_chunk_get, mock_source_get):
        """When no evidence_chunk_ids, falls back to 'defines' chunks."""
        defines_chunk_id = uuid4()
        source_id = uuid4()

        concept_a = _make_concept("A")
        concept_b = _make_concept("B")
        rel = _make_relationship(evidence_ids=[])  # No evidence_chunk_ids

        defines_cc = _make_chunk_concept(chunk_id=defines_chunk_id, mention_type="defines")
        chunk = _make_chunk(chunk_id=defines_chunk_id, source_id=source_id)
        source = _make_source(source_id=source_id)

        mock_chunk_get.return_value = chunk
        mock_source_get.return_value = source
        mock_list_chunks.side_effect = lambda cid, mention_type=None, limit=100: (
            [defines_cc] if mention_type == "defines" else []
        )

        steps = await hydrate_path_with_evidence(
            [(concept_a, None), (concept_b, rel)],
            max_evidence_per_step=2,
        )
        # concept_b should have defines evidence
        assert len(steps[1].evidence) >= 1
        assert steps[1].evidence[0].mention_type == "defines"

    @patch("research_kb_storage.source_store.SourceStore.get_by_id")
    @patch("research_kb_storage.chunk_store.ChunkStore.get_by_id")
    @patch("research_kb_storage.chunk_concept_store.ChunkConceptStore.list_chunks_for_concept")
    async def test_max_evidence_clamped(self, mock_list_chunks, mock_chunk_get, mock_source_get):
        """max_evidence_per_step clamped to 1-3."""
        mock_list_chunks.return_value = []
        mock_chunk_get.return_value = None

        concept = _make_concept("X")
        steps = await hydrate_path_with_evidence([(concept, None)], max_evidence_per_step=10)
        # Should not crash; clamped to 3 internally
        assert len(steps) == 1


# ─── synthesize_connection ───────────────────────────────────────────────────


class TestSynthesizeConnection:
    async def test_no_api_key_returns_none(self):
        """Without ANTHROPIC_API_KEY, returns None."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            result = await synthesize_connection(
                path_steps=[],
                concept_a="A",
                concept_b="B",
                path_explanation="A -> B",
            )
            assert result is None

    async def test_calls_anthropic_with_correct_prompt(self):
        """Verifies Anthropic client is called with evidence in the prompt."""
        mock_msg = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "DML requires cross-fitting for valid estimation."
        mock_msg.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        step = PathStep(
            concept_id=uuid4(),
            concept_name="DML",
            concept_type="method",
            definition="Double ML estimator",
            evidence=[
                EvidenceChunk(
                    chunk_id=uuid4(),
                    source_id=uuid4(),
                    source_title="Chernozhukov (2018)",
                    content="DML uses cross-fitting...",
                    page_start=42,
                    concept_name="DML",
                )
            ],
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                result = await synthesize_connection(
                    path_steps=[step],
                    concept_a="DML",
                    concept_b="cross-fitting",
                    path_explanation="DML -> cross-fitting",
                    style="educational",
                )

        assert result == "DML requires cross-fitting for valid estimation."
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        prompt_text = call_kwargs.kwargs["messages"][0]["content"]
        assert "Chernozhukov (2018)" in prompt_text
        assert "DML uses cross-fitting" in prompt_text

    async def test_api_error_returns_none(self):
        """API error during synthesis returns None gracefully."""
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.side_effect = Exception("API down")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                result = await synthesize_connection(
                    path_steps=[],
                    concept_a="A",
                    concept_b="B",
                    path_explanation="A -> B",
                )

        assert result is None

    async def test_import_error_returns_none(self):
        """Missing anthropic package returns None."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"anthropic": None}):
                result = await synthesize_connection(
                    path_steps=[],
                    concept_a="A",
                    concept_b="B",
                    path_explanation="A -> B",
                )
        # Should return None (ImportError path)
        assert result is None


# ─── explain_connection ──────────────────────────────────────────────────────


class TestExplainConnection:
    @patch("research_kb_storage.synthesis.synthesize_connection")
    @patch("research_kb_storage.synthesis.hydrate_path_with_evidence")
    @patch("research_kb_storage.graph_queries.find_shortest_path")
    @patch("research_kb_storage.concept_store.ConceptStore.search")
    async def test_full_flow(self, mock_search, mock_fsp, mock_hydrate, mock_synth):
        """Full happy path: find → hydrate → synthesize."""
        concept_a = _make_concept("DML", definition="Double machine learning")
        concept_b = _make_concept("cross-fitting", definition="Sample splitting technique")
        mock_search.side_effect = [[concept_a], [concept_b]]

        rel = _make_relationship()
        mock_fsp.return_value = [(concept_a, None), (concept_b, rel)]

        ev = EvidenceChunk(
            chunk_id=uuid4(),
            source_id=uuid4(),
            source_title="Paper",
            content="Evidence",
            concept_name="DML",
        )
        step_a = PathStep(
            concept_id=concept_a.id,
            concept_name="dml",
            concept_type="method",
            definition="Double machine learning",
        )
        step_b = PathStep(
            concept_id=concept_b.id,
            concept_name="cross-fitting",
            concept_type="method",
            definition="Sample splitting technique",
            evidence=[ev],
        )
        mock_hydrate.return_value = [step_a, step_b]
        mock_synth.return_value = "DML requires cross-fitting because..."

        result = await explain_connection("DML", "cross-fitting")

        assert isinstance(result, ConnectionExplanation)
        assert result.concept_a == "dml"
        assert result.concept_b == "cross-fitting"
        assert result.path_length == 1
        assert result.synthesis == "DML requires cross-fitting because..."
        assert result.source == "graph+llm"
        assert result.confidence == 0.85
        assert result.evidence_count == 1

    @patch("research_kb_storage.concept_store.ConceptStore.search")
    async def test_concept_not_found(self, mock_search):
        """Concept not found → source='error'."""
        mock_search.return_value = []

        result = await explain_connection("nonexistent", "also_nonexistent")

        assert result.source == "error"
        assert "not found" in result.path_explanation.lower()

    @patch("research_kb_storage.synthesis.hydrate_path_with_evidence")
    @patch("research_kb_storage.graph_queries.find_shortest_path")
    @patch("research_kb_storage.concept_store.ConceptStore.search")
    async def test_no_path(self, mock_search, mock_fsp, mock_hydrate):
        """No path found → source='no_path'."""
        concept_a = _make_concept("A")
        concept_b = _make_concept("B")
        mock_search.side_effect = [[concept_a], [concept_b]]
        mock_fsp.return_value = None

        result = await explain_connection("A", "B")

        assert result.source == "no_path"
        assert result.path_length == 0

    @patch("research_kb_storage.synthesis.synthesize_connection")
    @patch("research_kb_storage.synthesis.hydrate_path_with_evidence")
    @patch("research_kb_storage.graph_queries.find_shortest_path")
    @patch("research_kb_storage.concept_store.ConceptStore.search")
    async def test_use_llm_false(self, mock_search, mock_fsp, mock_hydrate, mock_synth):
        """use_llm=False → no synthesis, source='graph_only'."""
        concept_a = _make_concept("A")
        concept_b = _make_concept("B")
        mock_search.side_effect = [[concept_a], [concept_b]]

        rel = _make_relationship()
        mock_fsp.return_value = [(concept_a, None), (concept_b, rel)]
        mock_hydrate.return_value = [
            PathStep(concept_id=concept_a.id, concept_name="a", concept_type="method"),
            PathStep(concept_id=concept_b.id, concept_name="b", concept_type="method"),
        ]

        result = await explain_connection("A", "B", use_llm=False)

        assert result.source == "graph_only"
        assert result.synthesis is None
        assert result.confidence == 1.0
        mock_synth.assert_not_called()

    @patch("research_kb_storage.synthesis.synthesize_connection")
    @patch("research_kb_storage.synthesis.hydrate_path_with_evidence")
    @patch("research_kb_storage.graph_queries.find_shortest_path")
    @patch("research_kb_storage.concept_store.ConceptStore.search")
    async def test_llm_no_evidence(self, mock_search, mock_fsp, mock_hydrate, mock_synth):
        """LLM succeeds but no evidence → source='graph+llm_no_evidence'."""
        concept_a = _make_concept("A")
        concept_b = _make_concept("B")
        mock_search.side_effect = [[concept_a], [concept_b]]

        rel = _make_relationship()
        mock_fsp.return_value = [(concept_a, None), (concept_b, rel)]
        mock_hydrate.return_value = [
            PathStep(concept_id=concept_a.id, concept_name="a", concept_type="method"),
            PathStep(concept_id=concept_b.id, concept_name="b", concept_type="method"),
        ]  # No evidence in any step
        mock_synth.return_value = "Synthesized from definitions only."

        result = await explain_connection("A", "B")

        assert result.source == "graph+llm_no_evidence"
        assert result.confidence == 0.7


# ─── ConnectionExplanation.to_dict ───────────────────────────────────────────


class TestConnectionExplanationToDict:
    def test_full_serialization(self):
        """to_dict produces complete JSON-serializable structure."""
        ev = EvidenceChunk(
            chunk_id=uuid4(),
            source_id=uuid4(),
            source_title="Test Source",
            content="Evidence text",
            page_start=10,
            page_end=11,
            concept_name="IV",
            mention_type="defines",
        )
        step = PathStep(
            concept_id=uuid4(),
            concept_name="instrumental variables",
            concept_type="method",
            definition="A causal inference technique",
            relationship_to_next="requires",
            evidence=[ev],
        )
        result = ConnectionExplanation(
            concept_a="IV",
            concept_b="endogeneity",
            path=[step],
            path_length=1,
            path_explanation="IV -> endogeneity",
            synthesis="IV addresses endogeneity...",
            synthesis_style="educational",
            source="graph+llm",
            confidence=0.85,
            evidence_count=1,
        )

        d = result.to_dict()

        # Validate structure
        assert d["concept_a"] == "IV"
        assert d["concept_b"] == "endogeneity"
        assert len(d["path"]) == 1
        assert d["path"][0]["concept_name"] == "instrumental variables"
        assert len(d["path"][0]["evidence"]) == 1
        assert d["path"][0]["evidence"][0]["source_title"] == "Test Source"
        assert d["synthesis"] == "IV addresses endogeneity..."
        assert d["source"] == "graph+llm"
        assert d["confidence"] == 0.85

        # Must be JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["path_length"] == 1

    def test_no_evidence_serialization(self):
        """Empty evidence lists serialize correctly."""
        result = ConnectionExplanation(
            concept_a="A",
            concept_b="B",
            path=[
                PathStep(
                    concept_id=uuid4(),
                    concept_name="A",
                    concept_type="method",
                )
            ],
            path_length=0,
            path_explanation="A",
            source="graph_only",
        )
        d = result.to_dict()
        assert d["path"][0]["evidence"] == []
        assert d["synthesis"] is None

        # JSON roundtrip
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["evidence_count"] == 0

    def test_error_result_serialization(self):
        """Error results serialize with empty path."""
        result = ConnectionExplanation(
            concept_a="nonexistent",
            concept_b="also",
            path=[],
            path_length=0,
            path_explanation="Concept not found: nonexistent",
            source="error",
            confidence=0.0,
        )
        d = result.to_dict()
        assert d["path"] == []
        assert d["source"] == "error"

        json_str = json.dumps(d)
        assert '"error"' in json_str


# ─── EvidenceChunk.to_dict ───────────────────────────────────────────────────


class TestEvidenceChunkToDict:
    def test_serialization(self):
        chunk_id = uuid4()
        source_id = uuid4()
        ev = EvidenceChunk(
            chunk_id=chunk_id,
            source_id=source_id,
            source_title="Paper Title",
            content="Some evidence",
            page_start=5,
            page_end=None,
            concept_name="DML",
            mention_type="defines",
        )
        d = ev.to_dict()
        assert d["chunk_id"] == str(chunk_id)
        assert d["source_id"] == str(source_id)
        assert d["page_end"] is None
        assert d["mention_type"] == "defines"
