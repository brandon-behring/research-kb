"""Tests for literature review generation.

Tests the literature review module: topic exploration, section organization,
evidence assembly, LLM synthesis, and quality metrics.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from research_kb_storage.literature_review import (
    EVIDENCE_CONTENT_MAX_CHARS,
    MAX_CONCEPTS_EXPLORED,
    MAX_EVIDENCE_PER_SECTION,
    REVIEW_SECTIONS,
    LiteratureReview,
    ReviewEvidence,
    ReviewSection,
    _explore_topic_concepts,
    _search_evidence_for_section,
    _truncate,
    generate_literature_review,
)

pytestmark = pytest.mark.unit


# ── Helpers ──────────────────────────────────────────────────────────────────

_NOW = datetime.now(timezone.utc)


def _make_concept(
    name: str = "test_concept",
    concept_type: str = "method",
    definition: str = "A test concept",
) -> MagicMock:
    """Create a mock Concept."""
    concept = MagicMock()
    concept.id = uuid4()
    concept.name = name
    concept.canonical_name = name
    concept.definition = definition
    concept.concept_type = MagicMock(value=concept_type)
    return concept


def _make_search_result(
    title: str = "Test Source",
    content: str = "Test evidence content for the review.",
    score: float = 0.8,
) -> MagicMock:
    """Create a mock SearchResult."""
    result = MagicMock()
    result.chunk = MagicMock()
    result.chunk.id = uuid4()
    result.chunk.content = content
    result.chunk.page_start = 1
    result.chunk.page_end = 5
    result.chunk.domain_id = "causal_inference"
    result.source = MagicMock()
    result.source.id = uuid4()
    result.source.title = title
    result.combined_score = score
    result.rank = 1
    return result


# ── Data class tests ─────────────────────────────────────────────────────────


class TestReviewEvidence:
    """Test ReviewEvidence data class."""

    def test_to_dict(self):
        """ReviewEvidence.to_dict() includes all fields."""
        ev = ReviewEvidence(
            chunk_id=uuid4(),
            source_id=uuid4(),
            source_title="A Great Paper",
            content="Evidence text here",
            page_start=10,
            page_end=12,
            relevance_score=0.85,
            domain="causal_inference",
        )
        d = ev.to_dict()
        assert d["source_title"] == "A Great Paper"
        assert d["page_start"] == 10
        assert d["relevance_score"] == 0.85
        assert d["domain"] == "causal_inference"

    def test_content_truncated_in_dict(self):
        """Long content is truncated in to_dict()."""
        long_content = "x" * 1000
        ev = ReviewEvidence(
            chunk_id=uuid4(),
            source_id=uuid4(),
            source_title="Source",
            content=long_content,
        )
        d = ev.to_dict()
        assert len(d["content"]) <= EVIDENCE_CONTENT_MAX_CHARS


class TestReviewSection:
    """Test ReviewSection data class."""

    def test_to_dict_includes_all_fields(self):
        """ReviewSection.to_dict() serializes concepts, evidence, and synthesis."""
        section = ReviewSection(
            section_id="methods",
            title="Methods",
            description="Test methods section",
            concepts=[{"name": "DML", "type": "method"}, {"name": "cross-fitting", "type": "technique"}],
            synthesis="This section covers DML methods.",
        )
        d = section.to_dict()
        assert d["section_id"] == "methods"
        assert d["concepts"] == [{"name": "DML", "type": "method"}, {"name": "cross-fitting", "type": "technique"}]
        assert d["synthesis"] is not None


class TestLiteratureReview:
    """Test LiteratureReview data class."""

    def test_to_dict_full(self):
        """LiteratureReview.to_dict() includes all sections and metrics."""
        review = LiteratureReview(
            topic="DML",
            introduction="Intro text",
            sections=[
                ReviewSection(
                    section_id="overview",
                    title="Overview",
                    description="Overview section",
                    concepts=[{"name": "DML", "type": "method"}],
                    synthesis="Overview synthesis.",
                ),
            ],
            conclusion="Conclusion text",
            quality_metrics={"sources_cited": 5, "concepts_covered": 3},
            style="educational",
        )
        d = review.to_dict()
        assert d["topic"] == "DML"
        assert d["introduction"] == "Intro text"
        assert len(d["sections"]) == 1
        assert d["quality_metrics"]["sources_cited"] == 5

    def test_to_markdown_has_heading(self):
        """to_markdown() produces a document with the topic heading."""
        review = LiteratureReview(
            topic="Instrumental Variables",
            sections=[
                ReviewSection(
                    section_id="overview",
                    title="Overview",
                    description="Overview desc",
                    synthesis="IV overview text.",
                ),
            ],
            quality_metrics={"sources_cited": 2, "concepts_covered": 4, "evidence_chunks": 6},
        )
        md = review.to_markdown()
        assert "# Literature Review: Instrumental Variables" in md
        assert "## Overview" in md
        assert "IV overview text." in md
        assert "2 sources" in md

    def test_to_markdown_no_synthesis(self):
        """to_markdown() shows description when synthesis is None."""
        review = LiteratureReview(
            topic="Test",
            sections=[
                ReviewSection(
                    section_id="methods",
                    title="Methods",
                    description="Methods desc",
                    concepts=[{"name": "A", "type": "concept"}, {"name": "B", "type": "method"}],
                ),
            ],
        )
        md = review.to_markdown()
        assert "*Methods desc*" in md
        assert "A, B" in md

    def test_to_dict_roundtrip(self):
        """to_dict() output is JSON-serializable."""
        review = LiteratureReview(
            topic="Test",
            sections=[
                ReviewSection(
                    section_id="overview",
                    title="Overview",
                    description="desc",
                    evidence=[
                        ReviewEvidence(
                            chunk_id=uuid4(),
                            source_id=uuid4(),
                            source_title="Source",
                            content="Evidence",
                        )
                    ],
                )
            ],
            quality_metrics={"sources_cited": 1},
        )
        # Should not raise
        serialized = json.dumps(review.to_dict())
        parsed = json.loads(serialized)
        assert parsed["topic"] == "Test"
        assert len(parsed["sections"]) == 1


# ── Truncation ───────────────────────────────────────────────────────────────


class TestTruncate:
    """Test _truncate helper."""

    def test_short_text_unchanged(self):
        """Short text is returned unchanged."""
        assert _truncate("hello", 100) == "hello"

    def test_long_text_truncated(self):
        """Long text is truncated with ellipsis."""
        result = _truncate("a" * 500, 100)
        assert len(result) <= 100
        assert result.endswith("...")

    def test_exact_length_unchanged(self):
        """Text exactly at limit is unchanged."""
        text = "a" * 100
        assert _truncate(text, 100) == text

    def test_sentence_boundary_truncation(self):
        """Truncates at sentence boundary when possible."""
        text = "First sentence here. Second sentence here. Third sentence that goes on and on."
        result = _truncate(text, 50)
        # Should end at a sentence boundary
        assert result.endswith(".")
        assert len(result) <= 50

    def test_word_boundary_fallback(self):
        """Falls back to word boundary when no sentence end in range."""
        text = "word " * 100  # No sentence boundaries
        result = _truncate(text, 50)
        assert len(result) <= 50
        # Should not cut mid-word
        assert result.endswith("...")

    def test_no_early_sentence_cut(self):
        """Does not cut at very early sentence boundary (< 60% of limit)."""
        # Sentence ends at char 10, limit is 100 — should not use that boundary
        text = "Short one. " + "x" * 200
        result = _truncate(text, 100)
        # Should not cut at "Short one." (only 10 chars, well below 60% of 100)
        assert len(result) > 10


# ── Section definitions ──────────────────────────────────────────────────────


class TestReviewSections:
    """Test REVIEW_SECTIONS configuration."""

    def test_four_sections_defined(self):
        """Exactly 4 review sections are defined."""
        assert len(REVIEW_SECTIONS) == 4

    def test_section_ids_unique(self):
        """All section IDs are unique."""
        ids = [s["id"] for s in REVIEW_SECTIONS]
        assert len(ids) == len(set(ids))

    def test_each_section_has_required_fields(self):
        """Each section has id, title, description, concept_types, search_suffix."""
        for section in REVIEW_SECTIONS:
            assert "id" in section
            assert "title" in section
            assert "description" in section
            assert "concept_types" in section
            assert "search_suffix" in section

    def test_expected_section_ids(self):
        """Section IDs match expected values."""
        ids = {s["id"] for s in REVIEW_SECTIONS}
        assert ids == {"overview", "methods", "assumptions", "applications"}


# ── Topic exploration ────────────────────────────────────────────────────────


class TestExploreTopicConcepts:
    """Test _explore_topic_concepts()."""

    @pytest.mark.asyncio
    @patch("research_kb_storage.graph_queries.get_neighborhood")
    @patch("research_kb_storage.concept_store.ConceptStore.search")
    async def test_returns_concepts_from_graph(
        self, mock_search, mock_neighborhood
    ):
        """Finds seed concepts and expands via neighborhood."""
        seed = _make_concept("DML", "method")
        neighbor = _make_concept("cross-fitting", "technique")
        mock_search.return_value = [seed]
        mock_neighborhood.return_value = {
            "concepts": [neighbor],
            "relationships": [],
            "center": seed,
        }

        concepts = await _explore_topic_concepts("double machine learning")

        assert len(concepts) >= 1
        names = [c["name"] for c in concepts]
        assert "DML" in names
        # Verify get_neighborhood called with UUID, not string
        mock_neighborhood.assert_called_once_with(seed.id, hops=1)

    @pytest.mark.asyncio
    @patch("research_kb_storage.graph_queries.get_neighborhood")
    @patch("research_kb_storage.concept_store.ConceptStore.search")
    async def test_empty_topic_returns_empty(self, mock_search, mock_neighborhood):
        """Returns empty list when no concepts found."""
        mock_search.return_value = []

        concepts = await _explore_topic_concepts("nonexistent_topic_xyz")
        assert concepts == []
        mock_neighborhood.assert_not_called()

    @pytest.mark.asyncio
    @patch("research_kb_storage.graph_queries.get_neighborhood")
    @patch("research_kb_storage.concept_store.ConceptStore.search")
    async def test_respects_max_concepts(self, mock_search, mock_neighborhood):
        """Limits results to max_concepts."""
        seeds = [_make_concept(f"concept_{i}", "concept") for i in range(10)]
        mock_search.return_value = seeds
        mock_neighborhood.return_value = {
            "concepts": [],
            "relationships": [],
            "center": seeds[0],
        }

        concepts = await _explore_topic_concepts("test", max_concepts=5)
        assert len(concepts) <= 5


# ── Evidence search ──────────────────────────────────────────────────────────


class TestSearchEvidenceForSection:
    """Test _search_evidence_for_section()."""

    @pytest.mark.asyncio
    @patch("research_kb_storage.search.search_hybrid")
    @patch("research_kb_pdf.EmbeddingClient")
    async def test_returns_evidence_from_search(self, mock_embed_cls, mock_search):
        """Converts search results to ReviewEvidence."""
        mock_client = MagicMock()
        mock_client.embed_query.return_value = [0.1] * 1024
        mock_embed_cls.return_value = mock_client

        r1 = _make_search_result("Source A", "Evidence from source A")
        r2 = _make_search_result("Source B", "Evidence from source B")
        mock_search.return_value = [r1, r2]

        section_def = REVIEW_SECTIONS[0]  # overview
        concepts = [{"name": "DML", "type": "method"}]

        evidence = await _search_evidence_for_section("DML", section_def, concepts)

        assert len(evidence) == 2
        assert evidence[0].source_title == "Source A"
        assert evidence[1].source_title == "Source B"

    @pytest.mark.asyncio
    @patch("research_kb_storage.search.search_hybrid")
    @patch("research_kb_pdf.EmbeddingClient")
    async def test_limits_source_diversity(self, mock_embed_cls, mock_search):
        """Limits evidence to 2 chunks per source for diversity."""
        mock_client = MagicMock()
        mock_client.embed_query.return_value = [0.1] * 1024
        mock_embed_cls.return_value = mock_client

        shared_source_id = uuid4()
        results = []
        for i in range(5):
            r = _make_search_result("Same Source", f"Evidence {i}")
            r.source.id = shared_source_id
            results.append(r)
        mock_search.return_value = results

        section_def = REVIEW_SECTIONS[1]  # methods
        evidence = await _search_evidence_for_section("test", section_def, [])

        # Should cap at 2 from same source
        assert len(evidence) <= 2

    @pytest.mark.asyncio
    @patch("research_kb_storage.search.search_hybrid")
    @patch("research_kb_pdf.EmbeddingClient")
    async def test_excludes_chunk_ids(self, mock_embed_cls, mock_search):
        """Respects exclude_chunk_ids for cross-section dedup."""
        mock_client = MagicMock()
        mock_client.embed_query.return_value = [0.1] * 1024
        mock_embed_cls.return_value = mock_client

        r1 = _make_search_result("Source A", "Evidence A")
        r2 = _make_search_result("Source B", "Evidence B")
        r3 = _make_search_result("Source C", "Evidence C")
        mock_search.return_value = [r1, r2, r3]

        section_def = REVIEW_SECTIONS[0]

        # Exclude r1's chunk
        exclude = {r1.chunk.id}
        evidence = await _search_evidence_for_section(
            "test", section_def, [], exclude_chunk_ids=exclude
        )

        chunk_ids = {e.chunk_id for e in evidence}
        assert r1.chunk.id not in chunk_ids
        assert len(evidence) == 2


# ── Full orchestration ───────────────────────────────────────────────────────


class TestGenerateLiteratureReview:
    """Test generate_literature_review() orchestration."""

    @pytest.mark.asyncio
    @patch("research_kb_storage.literature_review._synthesize_intro_conclusion")
    @patch("research_kb_storage.literature_review._synthesize_section")
    @patch("research_kb_storage.literature_review._search_evidence_for_section")
    @patch("research_kb_storage.literature_review._explore_topic_concepts")
    async def test_full_review_without_llm(
        self, mock_explore, mock_search_ev, mock_synth_sec, mock_synth_ic
    ):
        """Generates review structure without LLM synthesis."""
        mock_explore.return_value = [
            {"id": str(uuid4()), "name": "DML", "type": "method",
             "definition": "Double ML", "relevance": 1.0},
            {"id": str(uuid4()), "name": "overlap", "type": "assumption",
             "definition": "Support overlap", "relevance": 0.8},
        ]
        mock_search_ev.return_value = [
            ReviewEvidence(
                chunk_id=uuid4(),
                source_id=uuid4(),
                source_title="Test Source",
                content="Evidence content",
            )
        ]

        review = await generate_literature_review(
            topic="double machine learning",
            style="educational",
            use_llm=False,
        )

        assert review.topic == "double machine learning"
        assert len(review.sections) == 4
        assert review.quality_metrics["concepts_explored"] == 2
        assert review.source == "graph+search"
        mock_synth_sec.assert_not_called()
        mock_synth_ic.assert_not_called()

    @pytest.mark.asyncio
    @patch("research_kb_storage.literature_review._synthesize_intro_conclusion")
    @patch("research_kb_storage.literature_review._synthesize_section")
    @patch("research_kb_storage.literature_review._search_evidence_for_section")
    @patch("research_kb_storage.literature_review._explore_topic_concepts")
    async def test_full_review_with_llm(
        self, mock_explore, mock_search_ev, mock_synth_sec, mock_synth_ic
    ):
        """Generates review with LLM synthesis."""
        mock_explore.return_value = [
            {"id": str(uuid4()), "name": "IV", "type": "method",
             "definition": "Instrumental variables", "relevance": 1.0},
        ]
        mock_search_ev.return_value = [
            ReviewEvidence(
                chunk_id=uuid4(),
                source_id=uuid4(),
                source_title="IV Paper",
                content="IV evidence",
            )
        ]
        mock_synth_sec.return_value = "Synthesized section text."
        mock_synth_ic.return_value = "Synthesized intro/conclusion."

        review = await generate_literature_review(
            topic="instrumental variables",
            style="research",
            use_llm=True,
        )

        assert review.topic == "instrumental variables"
        assert review.style == "research"
        assert review.source == "graph+search+llm"
        assert review.introduction == "Synthesized intro/conclusion."
        assert review.conclusion == "Synthesized intro/conclusion."
        # Synthesis called for each section
        assert mock_synth_sec.call_count == 4

    @pytest.mark.asyncio
    @patch("research_kb_storage.literature_review._search_evidence_for_section")
    @patch("research_kb_storage.literature_review._explore_topic_concepts")
    async def test_search_only_fallback(self, mock_explore, mock_search_ev):
        """Falls back to search-only when no graph concepts found."""
        mock_explore.return_value = []  # No concepts
        mock_search_ev.return_value = [
            ReviewEvidence(
                chunk_id=uuid4(),
                source_id=uuid4(),
                source_title="Source",
                content="Evidence",
            )
        ]

        review = await generate_literature_review(
            topic="unknown topic",
            use_llm=False,
        )

        assert review.quality_metrics.get("mode") == "search_only"
        assert review.source == "search_only"
        assert len(review.sections) == 4

    @pytest.mark.asyncio
    @patch("research_kb_storage.literature_review._search_evidence_for_section")
    @patch("research_kb_storage.literature_review._explore_topic_concepts")
    async def test_quality_metrics_populated(self, mock_explore, mock_search_ev):
        """Quality metrics are computed correctly."""
        src_id = uuid4()
        mock_explore.return_value = [
            {"id": str(uuid4()), "name": "A", "type": "concept",
             "definition": "", "relevance": 1.0},
            {"id": str(uuid4()), "name": "B", "type": "method",
             "definition": "", "relevance": 0.5},
        ]
        mock_search_ev.return_value = [
            ReviewEvidence(
                chunk_id=uuid4(),
                source_id=src_id,
                source_title="Source",
                content="Evidence",
                domain="causal_inference",
            )
        ]

        review = await generate_literature_review("test", use_llm=False)
        qm = review.quality_metrics

        assert qm["concepts_explored"] == 2
        assert qm["sources_cited"] >= 1
        assert qm["evidence_chunks"] >= 4  # 1 per section * 4 sections
        assert qm["sections_total"] == 4
        assert qm["sections_synthesized"] == 0  # No LLM
        assert "evidence_overlap_ratio" in qm
        assert "unique_evidence_chunks" in qm
        assert isinstance(qm["evidence_overlap_ratio"], float)

    @pytest.mark.asyncio
    @patch("research_kb_storage.literature_review._search_evidence_for_section")
    @patch("research_kb_storage.literature_review._explore_topic_concepts")
    async def test_to_markdown_and_json_consistency(self, mock_explore, mock_search_ev):
        """to_markdown() and to_dict() both work on the same review."""
        mock_explore.return_value = [
            {"id": str(uuid4()), "name": "X", "type": "concept",
             "definition": "", "relevance": 1.0},
        ]
        mock_search_ev.return_value = []

        review = await generate_literature_review("consistency test", use_llm=False)

        md = review.to_markdown()
        assert "consistency test" in md

        d = review.to_dict()
        serialized = json.dumps(d)
        assert "consistency test" in serialized
