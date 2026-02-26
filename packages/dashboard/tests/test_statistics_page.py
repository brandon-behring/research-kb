"""Tests for statistics page.

Tests the corpus statistics page by mocking Streamlit and the API client,
then calling the page function directly to verify metric display, chart
rendering, and error handling.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_stats():
    """Stats API response."""
    return {
        "sources": 495,
        "chunks": 228000,
        "concepts": 312000,
        "relationships": 744000,
        "citations": 10758,
    }


@pytest.fixture
def sample_sources():
    """Sources list API response with varied domains and types."""
    return {
        "sources": [
            {
                "id": str(uuid4()),
                "title": "Causal Inference Paper",
                "source_type": "paper",
                "authors": ["Imbens, G."],
                "year": 2015,
                "metadata": {"domain": "causal_inference", "citation_count": 5000},
            },
            {
                "id": str(uuid4()),
                "title": "ML Textbook",
                "source_type": "textbook",
                "authors": ["Hastie, T."],
                "year": 2009,
                "metadata": {"domain": "machine_learning", "citation_count": 12000},
            },
            {
                "id": str(uuid4()),
                "title": "Time Series Paper",
                "source_type": "paper",
                "authors": ["Hamilton, J."],
                "year": 1994,
                "metadata": {"domain": "time_series"},
            },
            {
                "id": str(uuid4()),
                "title": "Recent DL Paper",
                "source_type": "paper",
                "authors": ["LeCun, Y."],
                "year": 2020,
                "metadata": {"domain": "deep_learning", "citation_count": 3000},
            },
        ],
    }


def _make_mock_client(stats=None, sources_data=None, error=None):
    """Create a mock ResearchKBClient for statistics page testing."""
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()

    if error:
        mock_client.get_stats.side_effect = error
        mock_client.list_sources.side_effect = error
    else:
        mock_client.get_stats.return_value = stats or {}
        mock_client.list_sources.return_value = sources_data or {"sources": []}

    return mock_client


def _make_mock_st():
    """Create a mock streamlit module for statistics page testing."""
    mock_st = MagicMock()

    # columns context manager
    def make_cols(n):
        cols = []
        for _ in range(n):
            col = MagicMock()
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
            cols.append(col)
        return cols

    mock_st.columns.side_effect = make_cols

    # spinner context manager
    mock_st.spinner.return_value.__enter__ = MagicMock()
    mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

    return mock_st


# ---------------------------------------------------------------------------
# Rendering Tests
# ---------------------------------------------------------------------------


class TestStatisticsPageRendering:
    """Test basic page rendering and metrics display."""

    def test_statistics_page_renders(self, sample_stats, sample_sources):
        """Page runs without exception when API is mocked."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.statistics.px") as mock_px,
        ):
            mock_px.bar.return_value = MagicMock()
            mock_px.pie.return_value = MagicMock()

            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        mock_st.header.assert_called_once()
        header_arg = mock_st.header.call_args[0][0]
        assert "Statistics" in header_arg

    def test_top_level_metrics(self, sample_stats, sample_sources):
        """Page displays 5 top-level metric cards."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        # Capture col.metric calls
        metric_calls = []
        original_make_cols = mock_st.columns.side_effect

        def tracking_cols(n):
            cols = original_make_cols(n)
            for col in cols:
                original_metric = col.metric

                def capture_metric(*args, **kwargs):
                    metric_calls.append(args)

                col.metric = capture_metric
            return cols

        mock_st.columns.side_effect = tracking_cols

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.statistics.px") as mock_px,
        ):
            mock_px.bar.return_value = MagicMock()
            mock_px.pie.return_value = MagicMock()

            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        metric_labels = [m[0] for m in metric_calls]
        assert "Sources" in metric_labels
        assert "Chunks" in metric_labels
        assert "Concepts" in metric_labels
        assert "Relationships" in metric_labels
        assert "Citations" in metric_labels

    def test_domain_distribution_chart(self, sample_stats, sample_sources):
        """Domain distribution bar chart is created via plotly."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.statistics.px") as mock_px,
        ):
            mock_fig = MagicMock()
            mock_px.bar.return_value = mock_fig
            mock_px.pie.return_value = MagicMock()

            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        # px.bar called at least once (domain distribution)
        assert mock_px.bar.call_count >= 1
        mock_st.plotly_chart.assert_called()

    def test_source_type_pie_chart(self, sample_stats, sample_sources):
        """Source type pie chart is created."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.statistics.px") as mock_px,
        ):
            mock_px.bar.return_value = MagicMock()
            mock_px.pie.return_value = MagicMock()

            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        mock_px.pie.assert_called_once()

    def test_publication_timeline_chart(self, sample_stats, sample_sources):
        """Publication timeline bar chart is created when year data exists."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.statistics.px") as mock_px,
        ):
            mock_px.bar.return_value = MagicMock()
            mock_px.pie.return_value = MagicMock()

            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        # px.bar called at least twice (domain + timeline)
        assert mock_px.bar.call_count >= 2

    def test_citation_authority_section(self, sample_stats, sample_sources):
        """Sources with citation counts are listed in top-cited section."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.statistics.px") as mock_px,
        ):
            mock_px.bar.return_value = MagicMock()
            mock_px.pie.return_value = MagicMock()

            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        # Check markdown calls for citation entries
        markdown_calls = [c[0][0] for c in mock_st.markdown.call_args_list]
        citation_entries = [m for m in markdown_calls if "citations" in m.lower()]
        assert len(citation_entries) >= 1

    def test_knowledge_graph_stats_section(self, sample_stats, sample_sources):
        """Knowledge graph section shows when concepts > 0."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.statistics.px") as mock_px,
        ):
            mock_px.bar.return_value = MagicMock()
            mock_px.pie.return_value = MagicMock()

            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        subheader_calls = [c[0][0] for c in mock_st.subheader.call_args_list]
        assert any("Knowledge Graph" in s for s in subheader_calls)


# ---------------------------------------------------------------------------
# Empty/Edge Case Tests
# ---------------------------------------------------------------------------


class TestStatisticsEdgeCases:
    """Test edge cases and empty data."""

    def test_empty_sources(self, sample_stats):
        """Empty sources list shows info message."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data={"sources": []})

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        mock_st.info.assert_called()
        info_msg = mock_st.info.call_args[0][0]
        assert "no sources" in info_msg.lower() or "ingest" in info_msg.lower()

    def test_no_citation_counts(self, sample_stats):
        """Sources without citation_count show enrichment hint."""
        sources_no_citations = {
            "sources": [
                {
                    "id": str(uuid4()),
                    "title": "Paper A",
                    "source_type": "paper",
                    "year": 2020,
                    "metadata": {"domain": "causal_inference"},
                }
            ]
        }
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sources_no_citations)

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.statistics.px") as mock_px,
        ):
            mock_px.bar.return_value = MagicMock()
            mock_px.pie.return_value = MagicMock()

            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        info_calls = [c[0][0] for c in mock_st.info.call_args_list]
        assert any("enrich" in msg.lower() or "citation" in msg.lower() for msg in info_calls)

    def test_no_year_data(self, sample_stats):
        """Sources without years show info instead of timeline."""
        sources_no_years = {
            "sources": [
                {
                    "id": str(uuid4()),
                    "title": "Paper A",
                    "source_type": "paper",
                    "metadata": {"domain": "causal_inference"},
                }
            ]
        }
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sources_no_years)

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.statistics.px") as mock_px,
        ):
            mock_px.bar.return_value = MagicMock()
            mock_px.pie.return_value = MagicMock()

            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        info_calls = [c[0][0] for c in mock_st.info.call_args_list]
        assert any("year" in msg.lower() for msg in info_calls)

    def test_zero_concepts_hides_kg_section(self):
        """Knowledge graph section hidden when concepts == 0."""
        zero_stats = {
            "sources": 10,
            "chunks": 100,
            "concepts": 0,
            "relationships": 0,
            "citations": 0,
        }
        sources = {
            "sources": [
                {
                    "id": str(uuid4()),
                    "title": "A",
                    "source_type": "paper",
                    "year": 2020,
                    "metadata": {"domain": "x"},
                }
            ]
        }
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=zero_stats, sources_data=sources)

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.statistics.px") as mock_px,
        ):
            mock_px.bar.return_value = MagicMock()
            mock_px.pie.return_value = MagicMock()

            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        subheader_calls = [c[0][0] for c in mock_st.subheader.call_args_list]
        assert not any("Knowledge Graph" in s for s in subheader_calls)


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestStatisticsErrors:
    """Test error handling."""

    def test_connect_error(self):
        """API connection error shows error and startup hint."""
        import httpx

        mock_st = _make_mock_st()
        mock_client = _make_mock_client(error=httpx.ConnectError("Connection refused"))

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        mock_st.error.assert_called()
        error_msg = mock_st.error.call_args[0][0]
        assert "api" in error_msg.lower() or "connect" in error_msg.lower()
        mock_st.caption.assert_called()

    def test_generic_error(self):
        """Non-connection error shows error message."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(error=RuntimeError("Database timeout"))

        with (
            patch("research_kb_dashboard.pages.statistics.st", mock_st),
            patch(
                "research_kb_dashboard.pages.statistics.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.statistics import statistics_page

            statistics_page()

        mock_st.error.assert_called()
