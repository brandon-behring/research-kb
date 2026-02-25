"""Tests for citation graph functionality.

Phase 3: Tests for citation graph building, PageRank computation, and citation queries.
"""

import pytest
from uuid import uuid4

from research_kb_contracts import SourceType
from research_kb_storage import SourceStore, CitationStore
from research_kb_storage.citation_graph import (
    match_citation_to_source,
    compute_pagerank_authority,
    get_citing_sources,
    get_cited_sources,
    get_citation_stats,
    get_corpus_citation_summary,
    get_most_cited_sources,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def citation_test_sources(db_pool):
    """Create test sources for citation graph tests."""
    # Create 3 papers and 2 textbooks
    paper1 = await SourceStore.create(
        domain_id="causal_inference",
        source_type=SourceType.PAPER,
        title="Double Machine Learning",
        authors=["Chernozhukov", "Chetverikov"],
        year=2018,
        file_hash=f"sha256:test_paper1_{uuid4().hex[:8]}",
        metadata={"doi": "10.1111/ectj.12097"},
    )

    paper2 = await SourceStore.create(
        domain_id="causal_inference",
        source_type=SourceType.PAPER,
        title="Instrumental Variables Regression",
        authors=["Angrist", "Imbens"],
        year=1995,
        file_hash=f"sha256:test_paper2_{uuid4().hex[:8]}",
        metadata={"arxiv_id": "econ.em/9501001"},
    )

    paper3 = await SourceStore.create(
        domain_id="causal_inference",
        source_type=SourceType.PAPER,
        title="Treatment Effects Under Endogeneity",
        authors=["Test Author"],
        year=2020,
        file_hash=f"sha256:test_paper3_{uuid4().hex[:8]}",
    )

    textbook1 = await SourceStore.create(
        domain_id="causal_inference",
        source_type=SourceType.TEXTBOOK,
        title="Causality: Models, Reasoning and Inference",
        authors=["Judea Pearl"],
        year=2009,
        file_hash=f"sha256:test_textbook1_{uuid4().hex[:8]}",
        metadata={"isbn": "978-0521895606"},
    )

    textbook2 = await SourceStore.create(
        domain_id="causal_inference",
        source_type=SourceType.TEXTBOOK,
        title="Causal Inference: What If",
        authors=["Hernan", "Robins"],
        year=2020,
        file_hash=f"sha256:test_textbook2_{uuid4().hex[:8]}",
    )

    return {
        "paper1": paper1,  # DML
        "paper2": paper2,  # IV
        "paper3": paper3,  # Treatment effects
        "textbook1": textbook1,  # Pearl
        "textbook2": textbook2,  # Hernan/Robins
    }


@pytest.fixture
async def citation_graph_data(citation_test_sources, db_pool):
    """Create citations and citation graph edges for testing.

    Graph structure:
    - paper1 (DML) cites: paper2 (IV), textbook1 (Pearl)
    - paper3 (Treatment) cites: paper1 (DML), textbook1 (Pearl)
    - textbook2 (Hernan) cites: textbook1 (Pearl)

    Expected authority: textbook1 > paper1 > paper2 (textbook1 most cited)
    """
    sources = citation_test_sources
    pool = db_pool

    # Create citations for paper1 (DML cites IV and Pearl)
    citation_dml_iv = await CitationStore.create(
        source_id=sources["paper1"].id,
        raw_string="Angrist & Imbens (1995). Instrumental Variables.",
        title="Instrumental Variables Regression",
        authors=["Angrist", "Imbens"],
        year=1995,
    )

    citation_dml_pearl = await CitationStore.create(
        source_id=sources["paper1"].id,
        raw_string="Pearl, J. (2009). Causality.",
        title="Causality: Models, Reasoning and Inference",
        authors=["Judea Pearl"],
        year=2009,
    )

    # Create citations for paper3 (Treatment cites DML and Pearl)
    citation_treatment_dml = await CitationStore.create(
        source_id=sources["paper3"].id,
        raw_string="Chernozhukov et al. (2018). Double Machine Learning.",
        title="Double Machine Learning",
        authors=["Chernozhukov"],
        year=2018,
    )

    citation_treatment_pearl = await CitationStore.create(
        source_id=sources["paper3"].id,
        raw_string="Pearl, J. (2009). Causality.",
        title="Causality: Models, Reasoning and Inference",
        authors=["Pearl"],
        year=2009,
    )

    # Create citation for textbook2 (Hernan cites Pearl)
    citation_hernan_pearl = await CitationStore.create(
        source_id=sources["textbook2"].id,
        raw_string="Pearl, J. (2009). Causality.",
        title="Causality: Models, Reasoning and Inference",
        authors=["Judea Pearl"],
        year=2009,
    )

    # Create source_citations edges
    async with pool.acquire() as conn:
        # paper1 -> paper2 (IV)
        await conn.execute(
            """
            INSERT INTO source_citations (citing_source_id, cited_source_id, citation_id)
            VALUES ($1, $2, $3)
            """,
            sources["paper1"].id,
            sources["paper2"].id,
            citation_dml_iv.id,
        )

        # paper1 -> textbook1 (Pearl)
        await conn.execute(
            """
            INSERT INTO source_citations (citing_source_id, cited_source_id, citation_id)
            VALUES ($1, $2, $3)
            """,
            sources["paper1"].id,
            sources["textbook1"].id,
            citation_dml_pearl.id,
        )

        # paper3 -> paper1 (DML)
        await conn.execute(
            """
            INSERT INTO source_citations (citing_source_id, cited_source_id, citation_id)
            VALUES ($1, $2, $3)
            """,
            sources["paper3"].id,
            sources["paper1"].id,
            citation_treatment_dml.id,
        )

        # paper3 -> textbook1 (Pearl)
        await conn.execute(
            """
            INSERT INTO source_citations (citing_source_id, cited_source_id, citation_id)
            VALUES ($1, $2, $3)
            """,
            sources["paper3"].id,
            sources["textbook1"].id,
            citation_treatment_pearl.id,
        )

        # textbook2 -> textbook1 (Pearl)
        await conn.execute(
            """
            INSERT INTO source_citations (citing_source_id, cited_source_id, citation_id)
            VALUES ($1, $2, $3)
            """,
            sources["textbook2"].id,
            sources["textbook1"].id,
            citation_hernan_pearl.id,
        )

    return sources


class TestCitationMatcher:
    """Test citation matching to corpus sources."""

    async def test_match_by_exact_title_and_year(self, citation_test_sources):
        """Test matching citation by exact title and year."""
        # Create a citation that should match paper2
        citation = await CitationStore.create(
            source_id=citation_test_sources["paper1"].id,
            raw_string="Angrist (1995). Instrumental Variables.",
            title="Instrumental Variables Regression",
            authors=["Angrist"],
            year=1995,
        )

        matched = await match_citation_to_source(citation)

        assert matched is not None
        assert matched == citation_test_sources["paper2"].id

    async def test_match_no_match_for_external(self, citation_test_sources):
        """Test that external citations return None."""
        citation = await CitationStore.create(
            source_id=citation_test_sources["paper1"].id,
            raw_string="External Paper (2000). Not in corpus.",
            title="Not in Our Corpus at All",
            authors=["External Author"],
            year=2000,
        )

        matched = await match_citation_to_source(citation)

        # Should be None because this title doesn't exist in corpus
        assert matched is None

    async def test_match_returns_stdlib_uuid(self, citation_test_sources):
        """Regression: match must return stdlib uuid.UUID, not asyncpg UUID (Python 3.13)."""
        from uuid import UUID as StdlibUUID

        citation = await CitationStore.create(
            source_id=citation_test_sources["paper1"].id,
            raw_string="Angrist (1995). IV.",
            title="Instrumental Variables Regression",
            authors=["Angrist"],
            year=1995,
        )

        matched = await match_citation_to_source(citation)

        assert matched is not None
        assert type(matched) is StdlibUUID


class TestCitingSourcesQueries:
    """Test get_citing_sources and get_cited_sources queries."""

    async def test_get_citing_sources_all(self, citation_graph_data):
        """Test getting all sources that cite a source."""
        sources = citation_graph_data

        # Pearl (textbook1) is cited by: paper1, paper3, textbook2
        citing = await get_citing_sources(sources["textbook1"].id)

        # Results are dicts with 'id' key
        citing_ids = {s["id"] for s in citing}
        assert sources["paper1"].id in citing_ids
        assert sources["paper3"].id in citing_ids
        assert sources["textbook2"].id in citing_ids
        assert len(citing) == 3

    async def test_get_citing_sources_filtered_by_type(self, citation_graph_data):
        """Test getting citing sources filtered by type."""
        sources = citation_graph_data

        # Get only papers that cite Pearl
        citing_papers = await get_citing_sources(
            sources["textbook1"].id, source_type=SourceType.PAPER
        )

        # Results are dicts with 'id' key
        citing_ids = {s["id"] for s in citing_papers}
        assert sources["paper1"].id in citing_ids
        assert sources["paper3"].id in citing_ids
        assert sources["textbook2"].id not in citing_ids
        assert len(citing_papers) == 2

    async def test_get_cited_sources_all(self, citation_graph_data):
        """Test getting sources cited by a source."""
        sources = citation_graph_data

        # paper1 (DML) cites: paper2 (IV), textbook1 (Pearl)
        cited = await get_cited_sources(sources["paper1"].id)

        # Results are dicts with 'id' key
        cited_ids = {s["id"] for s in cited}
        assert sources["paper2"].id in cited_ids
        assert sources["textbook1"].id in cited_ids
        assert len(cited) == 2

    async def test_get_cited_sources_filtered_by_type(self, citation_graph_data):
        """Test getting cited sources filtered by type."""
        sources = citation_graph_data

        # paper1 cites textbook1 (Pearl), filter to textbooks only
        cited_textbooks = await get_cited_sources(
            sources["paper1"].id, source_type=SourceType.TEXTBOOK
        )

        # Results are dicts with 'id' key
        cited_ids = {s["id"] for s in cited_textbooks}
        assert sources["textbook1"].id in cited_ids
        assert sources["paper2"].id not in cited_ids
        assert len(cited_textbooks) == 1


class TestCitationStats:
    """Test citation statistics queries."""

    async def test_get_citation_stats(self, citation_graph_data):
        """Test getting citation stats for a source."""
        sources = citation_graph_data

        # Pearl (textbook1) stats: cited by 2 papers, 1 textbook
        stats = await get_citation_stats(sources["textbook1"].id)

        assert stats["cited_by_papers"] == 2
        assert stats["cited_by_textbooks"] == 1
        assert stats["cites_papers"] == 0  # Pearl doesn't cite anything in our graph
        assert stats["cites_textbooks"] == 0

    async def test_get_citation_stats_for_paper_citer(self, citation_graph_data):
        """Test citation stats for a source that cites others."""
        sources = citation_graph_data

        # paper1 (DML) cites: paper2 (IV), textbook1 (Pearl)
        # paper1 is cited by: paper3
        stats = await get_citation_stats(sources["paper1"].id)

        assert stats["cited_by_papers"] == 1  # paper3
        assert stats["cited_by_textbooks"] == 0
        assert stats["cites_papers"] == 1  # paper2
        assert stats["cites_textbooks"] == 1  # textbook1


class TestCorpusSummary:
    """Test corpus-wide citation statistics."""

    async def test_get_corpus_citation_summary(self, citation_graph_data):
        """Test getting corpus-wide citation summary."""
        summary = await get_corpus_citation_summary()

        # We have 5 edges total
        assert summary["total_edges"] == 5
        assert summary["internal_edges"] == 5  # All edges are internal (corpus->corpus)
        assert summary["external_edges"] == 0

        # Type breakdown:
        # paper1->paper2: paper_to_paper
        # paper1->textbook1: paper_to_textbook
        # paper3->paper1: paper_to_paper
        # paper3->textbook1: paper_to_textbook
        # textbook2->textbook1: textbook_to_textbook
        assert summary["paper_to_paper"] == 2
        assert summary["paper_to_textbook"] == 2
        assert summary["textbook_to_textbook"] == 1
        assert summary["textbook_to_paper"] == 0


class TestMostCitedSources:
    """Test most cited sources query."""

    async def test_get_most_cited_sources(self, citation_graph_data):
        """Test getting most cited sources."""
        sources = citation_graph_data

        most_cited = await get_most_cited_sources(limit=5)

        # Pearl (textbook1) should be #1 with 3 citations
        assert len(most_cited) > 0
        # Dict uses 'id' key, not 'source_id'
        assert most_cited[0]["id"] == sources["textbook1"].id
        assert most_cited[0]["cited_by_count"] == 3
        assert most_cited[0]["source_type"] == "textbook"


class TestPageRankComputation:
    """Test PageRank authority score computation."""

    async def test_compute_pagerank_updates_sources(self, citation_graph_data, db_pool):
        """Test that PageRank computation updates source authority scores."""
        sources = citation_graph_data

        # Compute PageRank
        stats = await compute_pagerank_authority(iterations=10, damping=0.85)

        assert stats["sources"] > 0

        # Verify scores are updated
        async with db_pool.acquire() as conn:
            pearl_authority = await conn.fetchval(
                "SELECT citation_authority FROM sources WHERE id = $1",
                sources["textbook1"].id,
            )

        # Pearl should have highest authority (most cited)
        assert pearl_authority > 0

    async def test_pagerank_ranking_order(self, citation_graph_data, db_pool):
        """Test that PageRank produces correct relative ranking."""
        sources = citation_graph_data

        # Compute PageRank
        await compute_pagerank_authority(iterations=20, damping=0.85)

        # Get authority scores
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, citation_authority
                FROM sources
                WHERE id = ANY($1)
                ORDER BY citation_authority DESC
                """,
                [s.id for s in sources.values()],
            )

        # Pearl should be highest (3 incoming citations)
        # Then DML (1 incoming citation, but from citing source)
        # IV, paper3, textbook2 should have lower scores
        authorities = {row["id"]: row["citation_authority"] for row in rows}

        pearl_auth = authorities[sources["textbook1"].id]
        dml_auth = authorities[sources["paper1"].id]
        iv_auth = authorities[sources["paper2"].id]

        # Pearl should be highest
        assert pearl_auth > dml_auth
        assert pearl_auth > iv_auth


class TestSearchIntegration:
    """Test citation authority integration with search."""

    async def test_search_query_with_citations(self, db_pool):
        """Test SearchQuery supports citation_weight and use_citations."""
        from research_kb_storage.search import SearchQuery

        query = SearchQuery(
            text="instrumental variables",
            embedding=[0.1] * 1024,
            fts_weight=0.2,
            vector_weight=0.4,
            graph_weight=0.2,
            citation_weight=0.2,
            use_graph=True,
            use_citations=True,
        )

        # Weights should be normalized
        total = query.fts_weight + query.vector_weight + query.graph_weight + query.citation_weight
        assert abs(total - 1.0) < 0.01

    async def test_search_query_citations_only(self, db_pool):
        """Test SearchQuery with citations but not graph."""
        from research_kb_storage.search import SearchQuery

        query = SearchQuery(
            text="test",
            embedding=[0.1] * 1024,
            fts_weight=0.3,
            vector_weight=0.5,
            citation_weight=0.2,
            use_citations=True,
        )

        # Should work without graph
        total = query.fts_weight + query.vector_weight + query.citation_weight
        assert abs(total - 1.0) < 0.01
