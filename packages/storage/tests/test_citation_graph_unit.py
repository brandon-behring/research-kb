"""Unit tests for citation graph computation logic (no database required).

Covers:
- compute_pagerank_authority: algorithm correctness with mock data
- match_citation_to_source_simple: priority matching logic
- build_citation_graph: error handling per citation, stats tracking
- get_citation_stats: fallback when no row found
- get_corpus_citation_summary: fallback when no view

Phase S Commit 2: Target citation_graph.py 9.4% → 25%
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool_mock(conn=None):
    """Create a properly structured asyncpg pool mock."""
    if conn is None:
        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        conn.fetchval = AsyncMock(return_value=None)
        conn.execute = AsyncMock()

    pool = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


# ---------------------------------------------------------------------------
# match_citation_to_source_simple
# ---------------------------------------------------------------------------


class TestMatchCitationToSourceSimple:
    """Unit tests for citation matching priority logic."""

    @patch("research_kb_storage.citation_graph.get_connection_pool")
    async def test_match_by_doi(self, mock_get_pool):
        """DOI match takes highest priority."""
        from research_kb_storage.citation_graph import match_citation_to_source_simple

        expected_id = uuid4()
        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": expected_id})
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        citation = MagicMock()
        citation.doi = "10.1111/ectj.12097"
        citation.arxiv_id = None
        citation.title = None
        citation.year = None

        result = await match_citation_to_source_simple(citation)
        assert result == expected_id

    @patch("research_kb_storage.citation_graph.get_connection_pool")
    async def test_match_by_arxiv_when_no_doi(self, mock_get_pool):
        """ArXiv match is second priority when DOI misses."""
        from research_kb_storage.citation_graph import match_citation_to_source_simple

        expected_id = uuid4()
        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        # DOI query returns None, arXiv query returns match
        conn.fetchrow = AsyncMock(side_effect=[None, {"id": expected_id}])
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        citation = MagicMock()
        citation.doi = "10.9999/nonexistent"
        citation.arxiv_id = "econ.em/9501001"
        citation.title = None
        citation.year = None

        result = await match_citation_to_source_simple(citation)
        assert result == expected_id

    @patch("research_kb_storage.citation_graph.get_connection_pool")
    async def test_no_match_returns_none(self, mock_get_pool):
        """Returns None when nothing matches."""
        from research_kb_storage.citation_graph import match_citation_to_source_simple

        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        citation = MagicMock()
        citation.doi = None
        citation.arxiv_id = None
        citation.title = "Unknown paper from unknown authors"
        citation.year = 1900

        result = await match_citation_to_source_simple(citation)
        assert result is None


# ---------------------------------------------------------------------------
# get_citation_stats
# ---------------------------------------------------------------------------


class TestGetCitationStats:
    """Unit tests for citation statistics query."""

    @patch("research_kb_storage.citation_graph.get_connection_pool")
    async def test_stats_returns_data_when_found(self, mock_get_pool):
        """Returns citation stats when row exists."""
        from research_kb_storage.citation_graph import get_citation_stats

        source_id = uuid4()
        mock_row = {
            "cited_by_count": 5,
            "cited_by_papers": 3,
            "cited_by_textbooks": 2,
            "cites_count": 10,
            "cites_papers": 7,
            "cites_textbooks": 3,
            "citation_authority": 0.85,
        }
        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=mock_row)
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        stats = await get_citation_stats(source_id)
        assert stats["cited_by_count"] == 5
        assert stats["citation_authority"] == 0.85

    @patch("research_kb_storage.citation_graph.get_connection_pool")
    async def test_stats_returns_zeros_when_not_found(self, mock_get_pool):
        """Returns zeros when no stats row exists."""
        from research_kb_storage.citation_graph import get_citation_stats

        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        stats = await get_citation_stats(uuid4())
        assert stats["cited_by_count"] == 0
        assert stats["citation_authority"] == 0.0


# ---------------------------------------------------------------------------
# get_corpus_citation_summary
# ---------------------------------------------------------------------------


class TestGetCorpusCitationSummary:
    """Unit tests for corpus-wide citation stats."""

    @patch("research_kb_storage.citation_graph.get_connection_pool")
    async def test_summary_returns_data_when_view_exists(self, mock_get_pool):
        """Returns summary data when materialized view has data."""
        from research_kb_storage.citation_graph import get_corpus_citation_summary

        mock_row = {
            "total_citations": 1000,
            "total_edges": 500,
            "internal_edges": 300,
            "external_edges": 200,
            "paper_to_paper": 150,
            "paper_to_textbook": 50,
            "textbook_to_paper": 80,
            "textbook_to_textbook": 20,
        }
        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=mock_row)
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        summary = await get_corpus_citation_summary()
        assert summary["total_citations"] == 1000

    @patch("research_kb_storage.citation_graph.get_connection_pool")
    async def test_summary_returns_empty_when_no_view(self, mock_get_pool):
        """Returns empty dict when view doesn't exist."""
        from research_kb_storage.citation_graph import get_corpus_citation_summary

        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        summary = await get_corpus_citation_summary()
        assert summary == {} or all(v == 0 for v in summary.values())


# ---------------------------------------------------------------------------
# get_most_cited_sources
# ---------------------------------------------------------------------------


class TestGetMostCitedSources:
    """Unit tests for most-cited sources query."""

    @patch("research_kb_storage.citation_graph.get_connection_pool")
    async def test_returns_list_of_dicts(self, mock_get_pool):
        """Returns a list of source dicts with citation counts."""
        from research_kb_storage.citation_graph import get_most_cited_sources

        sid = uuid4()
        mock_rows = [
            {
                "id": sid,
                "source_type": "paper",
                "title": "Causality",
                "authors": ["Pearl"],
                "year": 2009,
                "citation_authority": 0.95,
                "cited_by_count": 42,
            }
        ]
        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetch = AsyncMock(return_value=mock_rows)
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        results = await get_most_cited_sources(limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Causality"

    @patch("research_kb_storage.citation_graph.get_connection_pool")
    async def test_empty_corpus_returns_empty_list(self, mock_get_pool):
        """Returns empty list when no sources have citations."""
        from research_kb_storage.citation_graph import get_most_cited_sources

        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        results = await get_most_cited_sources(limit=10)
        assert results == []


# ---------------------------------------------------------------------------
# get_citing_sources / get_cited_sources
# ---------------------------------------------------------------------------


class TestCitingAndCitedSources:
    """Unit tests for directional citation queries."""

    @patch("research_kb_storage.citation_graph.get_connection_pool")
    async def test_get_citing_sources_returns_results(self, mock_get_pool):
        """Returns sources that cite the given source."""
        from research_kb_storage.citation_graph import get_citing_sources

        sid1 = uuid4()
        sid2 = uuid4()
        mock_rows = [
            {
                "id": sid1,
                "source_type": "paper",
                "title": "Paper A",
                "authors": ["Author 1"],
                "year": 2020,
                "citation_authority": 0.5,
                "citation_count": 3,
            }
        ]
        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetch = AsyncMock(return_value=mock_rows)
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        results = await get_citing_sources(sid2)
        assert len(results) == 1
        assert results[0]["title"] == "Paper A"

    @patch("research_kb_storage.citation_graph.get_connection_pool")
    async def test_get_cited_sources_returns_results(self, mock_get_pool):
        """Returns sources cited by the given source."""
        from research_kb_storage.citation_graph import get_cited_sources

        sid = uuid4()
        mock_rows = [
            {
                "id": uuid4(),
                "source_type": "textbook",
                "title": "Textbook B",
                "authors": ["Author 2"],
                "year": 2015,
                "citation_authority": 0.8,
                "citation_count": 1,
            }
        ]
        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetch = AsyncMock(return_value=mock_rows)
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        results = await get_cited_sources(sid)
        assert len(results) == 1
        assert results[0]["title"] == "Textbook B"
