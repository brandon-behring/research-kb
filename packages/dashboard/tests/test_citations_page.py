"""Tests for citations page.

Tests the citation network page by mocking Streamlit and the API client,
then calling the page function directly to verify widget creation and
data rendering for various states (data loaded, empty, errors).
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
def sample_sources():
    """Sample sources for citation network."""
    return [
        {
            "id": str(uuid4()),
            "title": "Instrumental Variables Paper",
            "source_type": "paper",
            "authors": ["Angrist, J."],
            "year": 1995,
            "metadata": {"citation_authority": 0.85},
        },
        {
            "id": str(uuid4()),
            "title": "Causal Inference Textbook",
            "source_type": "textbook",
            "authors": ["Imbens, G.", "Rubin, D."],
            "year": 2015,
            "metadata": {"citation_authority": 0.92},
        },
    ]


def _make_mock_client(sources=None, edges_map=None, error=None):
    """Create a mock ResearchKBClient for citation page testing.

    Args:
        sources: List of source dicts for list_sources
        edges_map: Dict mapping source_id -> citation response
        error: Exception to raise on API calls
    """
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()

    if error:
        mock_client.list_sources.side_effect = error
    else:
        mock_client.list_sources.return_value = {"sources": sources or []}

        async def get_citations(source_id):
            if edges_map and source_id in edges_map:
                return edges_map[source_id]
            return {"cited_sources": [], "citing_sources": []}

        mock_client.get_source_citations = AsyncMock(side_effect=get_citations)

    return mock_client


def _make_mock_st():
    """Create a mock streamlit module for citation page testing."""
    mock_st = MagicMock()

    # selectbox returns first option
    mock_st.selectbox.side_effect = lambda label, options, **kw: options[kw.get("index", 0)]

    # slider returns the value kwarg
    mock_st.slider.side_effect = lambda label, **kw: kw.get("value", 0)

    # checkbox returns the value kwarg
    mock_st.checkbox.side_effect = lambda label, **kw: kw.get("value", True)

    # columns context manager — return exactly n columns
    def make_cols(n):
        cols = []
        for _ in range(n):
            col = MagicMock()
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
            col.selectbox = mock_st.selectbox
            col.slider = mock_st.slider
            col.checkbox = mock_st.checkbox
            cols.append(col)
        return cols

    mock_st.columns.side_effect = make_cols

    # spinner context manager
    mock_st.spinner.return_value.__enter__ = MagicMock()
    mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

    # expander context manager
    exp = MagicMock()
    exp.__enter__ = MagicMock(return_value=exp)
    exp.__exit__ = MagicMock(return_value=False)
    mock_st.expander.return_value = exp

    return mock_st


# ---------------------------------------------------------------------------
# Rendering Tests
# ---------------------------------------------------------------------------


class TestCitationsPageRendering:
    """Test that the citations page creates expected widgets."""

    def test_citations_page_renders(self):
        """Page runs without exception when API is mocked."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(sources=[])

        with (
            patch("research_kb_dashboard.pages.citations.st", mock_st),
            patch(
                "research_kb_dashboard.pages.citations.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.citations import citation_network_page

            citation_network_page()

        mock_st.header.assert_called_once()

    def test_citations_page_header(self):
        """Citation page shows the Citation Network header."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(sources=[])

        with (
            patch("research_kb_dashboard.pages.citations.st", mock_st),
            patch(
                "research_kb_dashboard.pages.citations.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.citations import citation_network_page

            citation_network_page()

        header_arg = mock_st.header.call_args[0][0]
        assert "Citation Network" in header_arg

    def test_citations_page_source_type_filter(self):
        """Source type selectbox offers All, Paper, Textbook."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(sources=[])

        with (
            patch("research_kb_dashboard.pages.citations.st", mock_st),
            patch(
                "research_kb_dashboard.pages.citations.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.citations import citation_network_page

            citation_network_page()

        selectbox_calls = mock_st.selectbox.call_args_list
        type_call = None
        for c in selectbox_calls:
            if "type" in c[0][0].lower() or "filter" in c[0][0].lower():
                type_call = c
                break

        assert type_call is not None
        options = type_call[0][1]
        assert "All" in options
        assert "Paper" in options
        assert "Textbook" in options

    def test_citations_page_authority_slider(self):
        """Min authority slider has range 0.0-1.0."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(sources=[])

        with (
            patch("research_kb_dashboard.pages.citations.st", mock_st),
            patch(
                "research_kb_dashboard.pages.citations.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.citations import citation_network_page

            citation_network_page()

        slider_calls = mock_st.slider.call_args_list
        authority_call = None
        for c in slider_calls:
            if "authority" in c[0][0].lower():
                authority_call = c
                break

        assert authority_call is not None
        assert authority_call[1]["min_value"] == 0.0
        assert authority_call[1]["max_value"] == 1.0
        assert authority_call[1]["step"] == 0.05

    def test_citations_page_isolated_checkbox(self):
        """Show Isolated Nodes checkbox present with default True."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(sources=[])

        with (
            patch("research_kb_dashboard.pages.citations.st", mock_st),
            patch(
                "research_kb_dashboard.pages.citations.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.citations import citation_network_page

            citation_network_page()

        checkbox_calls = mock_st.checkbox.call_args_list
        isolated_call = None
        for c in checkbox_calls:
            if "isolated" in c[0][0].lower():
                isolated_call = c
                break

        assert isolated_call is not None
        assert isolated_call[1]["value"] is True


# ---------------------------------------------------------------------------
# Data Display Tests
# ---------------------------------------------------------------------------


class TestCitationsData:
    """Test citation data display and error handling."""

    def test_citations_page_with_data(self, sample_sources):
        """Page shows source and edge counts with data."""
        # source[0] cites source[1]
        edges_map = {
            sample_sources[0]["id"]: {
                "cited_sources": [{"id": sample_sources[1]["id"]}],
                "citing_sources": [],
            },
        }

        mock_st = _make_mock_st()
        mock_client = _make_mock_client(sources=sample_sources, edges_map=edges_map)

        with (
            patch("research_kb_dashboard.pages.citations.st", mock_st),
            patch(
                "research_kb_dashboard.pages.citations.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.citations.create_network") as mock_net,
            patch("research_kb_dashboard.pages.citations.render_network"),
        ):
            net = MagicMock()
            mock_net.return_value = net

            from research_kb_dashboard.pages.citations import citation_network_page

            citation_network_page()

        # Should call st.info with counts
        mock_st.info.assert_called()
        info_msg = mock_st.info.call_args[0][0]
        assert "2" in info_msg
        assert "sources" in info_msg.lower()

    def test_citations_page_empty_sources(self):
        """Warning shown when no sources match filters."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(sources=[])

        with (
            patch("research_kb_dashboard.pages.citations.st", mock_st),
            patch(
                "research_kb_dashboard.pages.citations.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.citations import citation_network_page

            citation_network_page()

        mock_st.warning.assert_called()
        warning_msg = mock_st.warning.call_args[0][0]
        assert "no sources" in warning_msg.lower()

    def test_citations_page_connect_error(self):
        """API connection error handled gracefully."""
        import httpx

        mock_st = _make_mock_st()
        mock_client = _make_mock_client(error=httpx.ConnectError("Connection refused"))

        with (
            patch("research_kb_dashboard.pages.citations.st", mock_st),
            patch(
                "research_kb_dashboard.pages.citations.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.citations import citation_network_page

            citation_network_page()

        mock_st.error.assert_called()
        error_msg = mock_st.error.call_args[0][0]
        assert "api" in error_msg.lower() or "connect" in error_msg.lower()

    def test_citations_page_generic_error(self):
        """Non-connection error handled gracefully."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(error=RuntimeError("Database error"))

        with (
            patch("research_kb_dashboard.pages.citations.st", mock_st),
            patch(
                "research_kb_dashboard.pages.citations.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.citations import citation_network_page

            citation_network_page()

        mock_st.error.assert_called()

    def test_citations_page_network_nodes(self, sample_sources):
        """Network nodes are added for each source."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(sources=sample_sources)

        with (
            patch("research_kb_dashboard.pages.citations.st", mock_st),
            patch(
                "research_kb_dashboard.pages.citations.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.citations.create_network") as mock_create,
            patch("research_kb_dashboard.pages.citations.render_network"),
        ):
            net = MagicMock()
            mock_create.return_value = net

            from research_kb_dashboard.pages.citations import citation_network_page

            citation_network_page()

        # Should add 2 nodes (one per source)
        assert net.add_node.call_count == 2
