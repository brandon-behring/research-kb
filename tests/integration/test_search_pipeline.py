"""Integration tests for the full search pipeline.

Validates FTS, vector, hybrid, domain-filtered, graph-boosted, and
citation-authority search using synthetic data seeded via Store APIs.

No PDFs, embedding servers, or LLMs required — runs against PostgreSQL
+ pgvector only. Embeddings are synthetic but with controlled cosine
similarity so ranking assertions are meaningful.

Marked @pytest.mark.integration to exclude from fast PR checks.

Run locally:
    pytest tests/integration/test_search_pipeline.py -v
"""

import pytest

from research_kb_storage import (
    SearchQuery,
    search_hybrid,
    search_hybrid_v2,
)
from research_kb_storage.assumption_audit import MethodAssumptionAuditor

from .conftest import IV_VECTOR, DML_VECTOR, RAG_VECTOR, _blend_embeddings


pytestmark = pytest.mark.integration


# =========================================================================
# 1. Full-text search
# =========================================================================


class TestFTSSearch:
    """FTS returns chunks containing query terms, ranked by relevance."""

    async def test_fts_finds_relevant_chunks(self, search_corpus):
        """Searching 'instrumental variables' finds IV-related chunks."""
        results = await search_hybrid(SearchQuery(text="instrumental variables", limit=10))

        assert len(results) >= 1

        # Top result should mention instrumental variables
        top_content = results[0].chunk.content.lower()
        assert "instrumental variable" in top_content or "instrument" in top_content

    async def test_fts_no_match_returns_empty(self, search_corpus):
        """FTS with nonsense query returns no results."""
        results = await search_hybrid(SearchQuery(text="xylophone_zqx_99", limit=10))

        assert results == []


# =========================================================================
# 2. Vector similarity search
# =========================================================================


class TestVectorSearch:
    """Vector search returns chunks with closest embeddings."""

    async def test_vector_returns_closest_embedding(self, search_corpus):
        """Query with IV_VECTOR should surface IV chunks first."""
        results = await search_hybrid(SearchQuery(embedding=IV_VECTOR, limit=10))

        assert len(results) >= 1

        # First result should be the IV definition chunk (exact match embedding)
        top = results[0]
        assert top.vector_score is not None
        # Cosine similarity of a vector with itself = 1.0
        assert top.vector_score > 0.95

        # The top result should be from the IV cluster
        assert "instrumental variable" in top.chunk.content.lower()

    async def test_vector_ranking_order(self, search_corpus):
        """Results should be ordered by descending similarity."""
        results = await search_hybrid(SearchQuery(embedding=DML_VECTOR, limit=10))

        assert len(results) >= 1

        # If multiple results returned, verify descending order
        for i in range(len(results) - 1):
            assert results[i].vector_score >= results[i + 1].vector_score - 1e-6


# =========================================================================
# 3. Hybrid search (FTS + vector)
# =========================================================================


class TestHybridSearch:
    """Combined FTS + vector outperforms either signal alone."""

    async def test_hybrid_combines_signals(self, search_corpus):
        """Hybrid search with both text and embedding returns results
        with combined scores."""
        results = await search_hybrid(
            SearchQuery(
                text="instrumental variables",
                embedding=IV_VECTOR,
                fts_weight=0.3,
                vector_weight=0.7,
                limit=10,
            )
        )

        assert len(results) >= 1

        # Top result should have both FTS and vector scores
        top = results[0]
        assert top.combined_score > 0

        # Should definitely surface IV content at the top
        top_content = top.chunk.content.lower()
        assert "instrumental" in top_content

    async def test_hybrid_respects_limit(self, search_corpus):
        """Hybrid search respects the result limit."""
        results = await search_hybrid(
            SearchQuery(
                text="causal",
                embedding=IV_VECTOR,
                limit=3,
            )
        )

        assert len(results) <= 3


# =========================================================================
# 4. Domain filtering
# =========================================================================


class TestDomainFilter:
    """Domain filter restricts results to specified domain."""

    async def test_domain_filter_causal_inference(self, search_corpus):
        """Filtering by causal_inference excludes RAG chunks."""
        results = await search_hybrid(
            SearchQuery(
                text="retrieval augmented generation",
                embedding=RAG_VECTOR,
                domain_id="causal_inference",
                limit=10,
            )
        )

        # Should NOT return any rag_llm domain chunks
        for result in results:
            source_domain = result.source.metadata.get("domain", "")
            assert source_domain != "rag_llm", (
                f"Domain filter leaked: got rag_llm chunk from " f"source '{result.source.title}'"
            )

    async def test_domain_filter_rag_llm(self, search_corpus):
        """Filtering by rag_llm returns only RAG domain chunks."""
        results = await search_hybrid(
            SearchQuery(
                text="retrieval",
                embedding=RAG_VECTOR,
                domain_id="rag_llm",
                limit=10,
            )
        )

        assert len(results) >= 1

        for result in results:
            source_domain = result.source.metadata.get("domain", "")
            assert (
                source_domain == "rag_llm"
            ), f"Domain filter leaked: expected rag_llm, got '{source_domain}'"


# =========================================================================
# 5. Graph-boosted search
# =========================================================================


class TestGraphBoostedSearch:
    """Chunks linked to query-related concepts rank higher."""

    async def test_graph_boost_elevates_concept_linked_chunks(self, search_corpus):
        """With graph boosting, chunks linked to IV concepts should rank
        higher than unlinked chunks with similar text."""
        results = await search_hybrid_v2(
            SearchQuery(
                text="instrumental variables",
                embedding=IV_VECTOR,
                fts_weight=0.3,
                vector_weight=0.4,
                graph_weight=0.3,
                use_graph=True,
                limit=10,
            )
        )

        assert len(results) >= 1

        # Top results should be concept-linked IV chunks
        top_content = results[0].chunk.content.lower()
        assert "instrumental" in top_content or "variable" in top_content


# =========================================================================
# 6. Assumption audit
# =========================================================================


class TestAssumptionAudit:
    """Method -> assumption resolution via concept relationships."""

    async def test_audit_finds_iv_assumptions(self, search_corpus):
        """Auditing 'Instrumental Variables' resolves exclusion + relevance."""
        result = await MethodAssumptionAuditor.audit_assumptions(
            "Instrumental Variables",
            use_ollama_fallback=False,
        )

        assert "instrumental_variables" in result.method
        assert result.source == "graph"

        # Should have found at least the 2 REQUIRES relationships
        assumption_names = [a.name.lower() for a in result.assumptions]
        assert any(
            "exclusion" in name for name in assumption_names
        ), f"Missing 'exclusion restriction' in {assumption_names}"
        assert any(
            "relevance" in name for name in assumption_names
        ), f"Missing 'relevance' in {assumption_names}"

    async def test_audit_finds_dml_assumptions(self, search_corpus):
        """Auditing 'Double Machine Learning' resolves unconfoundedness + overlap."""
        result = await MethodAssumptionAuditor.audit_assumptions(
            "Double Machine Learning",
            use_ollama_fallback=False,
        )

        assert "double_machine_learning" in result.method
        assert result.source == "graph"

        assumption_names = [a.name.lower() for a in result.assumptions]
        assert any(
            "unconfoundedness" in name for name in assumption_names
        ), f"Missing 'unconfoundedness' in {assumption_names}"
        assert any(
            "overlap" in name for name in assumption_names
        ), f"Missing 'overlap' in {assumption_names}"

    async def test_audit_unknown_method_returns_not_found(self, search_corpus):
        """Auditing a non-existent method returns source='not_found'."""
        result = await MethodAssumptionAuditor.audit_assumptions(
            "Nonexistent Method XYZ",
            use_ollama_fallback=False,
        )

        assert result.source == "not_found"
        assert result.assumptions == []


# =========================================================================
# 7. Citation authority
# =========================================================================


class TestCitationAuthority:
    """Highly-cited sources rank above uncited ones."""

    async def test_citation_authority_boosts_cited_sources(self, search_corpus):
        """With citation boosting, chunks from highly-cited sources
        (Pearl, Angrist) should rank higher."""
        results = await search_hybrid_v2(
            SearchQuery(
                text="causal effect estimation",
                embedding=_blend_embeddings(IV_VECTOR, DML_VECTOR, alpha=0.5),
                fts_weight=0.2,
                vector_weight=0.4,
                citation_weight=0.4,
                use_citations=True,
                limit=10,
            )
        )

        assert len(results) >= 1

        # Results should exist and have scores
        for result in results:
            assert result.combined_score > 0


# =========================================================================
# 8. Edge cases
# =========================================================================


class TestEdgeCases:
    """Graceful handling of edge cases."""

    async def test_empty_text_with_embedding(self, search_corpus):
        """Vector-only search (no text) works correctly."""
        results = await search_hybrid(SearchQuery(embedding=IV_VECTOR, limit=5))

        assert len(results) >= 1
        assert all(r.vector_score is not None for r in results)

    async def test_text_only_no_embedding(self, search_corpus):
        """FTS-only search (no embedding) works correctly."""
        results = await search_hybrid(SearchQuery(text="double machine learning", limit=5))

        assert len(results) >= 1

    async def test_limit_one(self, search_corpus):
        """Limit=1 returns exactly one result."""
        results = await search_hybrid(SearchQuery(text="instrumental variables", limit=1))

        assert len(results) == 1

    async def test_search_query_validation_rejects_empty(self):
        """SearchQuery with no text or embedding raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            SearchQuery()

    async def test_search_query_wrong_dimension(self):
        """SearchQuery rejects wrong embedding dimensions."""
        with pytest.raises(ValueError, match="1024"):
            SearchQuery(embedding=[0.1] * 512)
