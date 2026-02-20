"""Tests for search page.

Tests the search page by mocking Streamlit and the API client, then
calling the page function directly to verify widget creation and
result rendering for various states (empty, results, errors).
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
def mock_search_results():
    """Search results matching the API response format."""
    source_id = str(uuid4())
    chunk_id = str(uuid4())
    return [
        {
            "source": {
                "id": source_id,
                "title": "Instrumental Variables Methods",
                "authors": ["Angrist, J.", "Imbens, G."],
                "year": 1995,
                "source_type": "paper",
            },
            "chunk": {
                "id": chunk_id,
                "content": "IV estimation is a technique used when randomization is infeasible."
                * 5,
                "page_start": 10,
                "page_end": 12,
                "section": "Introduction",
            },
            "scores": {
                "fts": 0.45,
                "vector": 0.82,
                "graph": 0.15,
                "citation": 0.08,
            },
            "combined_score": 0.78,
        },
    ]


def _make_mock_client(search_results=None, search_error=None):
    """Create a mock ResearchKBClient for search page testing."""
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()

    if search_error:
        mock_client.search.side_effect = search_error
    else:
        mock_client.search.return_value = {"results": search_results or []}

    return mock_client


def _make_mock_st(query_value=""):
    """Create a mock streamlit module with widgets returning configured values.

    Args:
        query_value: Value returned by text_input (simulates user typing)

    Returns:
        MagicMock configured as streamlit module
    """
    mock_st = MagicMock()

    # text_input returns the query value
    mock_st.text_input.return_value = query_value

    # selectbox returns first option by default
    mock_st.selectbox.side_effect = lambda label, options, **kw: options[kw.get("index", 0)]

    # slider returns the value kwarg
    mock_st.slider.side_effect = lambda label, **kw: kw.get("value", kw.get("min_value", 0))

    # expander context manager
    mock_st.expander.return_value.__enter__ = MagicMock()
    mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

    # columns context manager (returns list of mocks)
    col1, col2, col3, col4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    for col in (col1, col2, col3, col4):
        col.__enter__ = MagicMock(return_value=col)
        col.__exit__ = MagicMock(return_value=False)
        col.selectbox = mock_st.selectbox
        col.slider = mock_st.slider
        col.metric = MagicMock()

    mock_st.columns.side_effect = lambda n: [col1, col2, col3, col4][:n]

    # spinner context manager
    mock_st.spinner.return_value.__enter__ = MagicMock()
    mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

    # container context manager
    mock_st.container.return_value.__enter__ = MagicMock()
    mock_st.container.return_value.__exit__ = MagicMock(return_value=False)

    # button returns False by default
    mock_st.button.return_value = False

    # session_state
    mock_st.session_state = {}

    return mock_st


# ---------------------------------------------------------------------------
# Rendering Tests
# ---------------------------------------------------------------------------


class TestSearchPageRendering:
    """Test that the search page creates expected widgets."""

    def test_search_page_renders(self):
        """Page runs without exception when API is mocked."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(search_results=[])

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        # Verify no crash — function completed
        mock_st.header.assert_called_once()

    def test_search_page_header(self):
        """Search page shows the Hybrid Search header."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(search_results=[])

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        header_args = mock_st.header.call_args[0][0]
        assert "Hybrid Search" in header_args

    def test_search_page_empty_shows_info(self):
        """Empty query shows info message."""
        mock_st = _make_mock_st(query_value="")
        mock_client = _make_mock_client(search_results=[])

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        # Should call st.info with search-related message
        mock_st.info.assert_called()
        info_msg = mock_st.info.call_args[0][0]
        assert "search query" in info_msg.lower()

    def test_search_page_example_buttons(self):
        """Page shows 5 example query buttons when empty."""
        mock_st = _make_mock_st(query_value="")
        mock_client = _make_mock_client(search_results=[])

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        # Should have at least 5 button calls (one per example query)
        button_calls = [c for c in mock_st.button.call_args_list]
        assert len(button_calls) >= 5

        # Check known example queries
        button_labels = [c[0][0] for c in button_calls]
        assert "instrumental variables" in button_labels
        assert "propensity score matching" in button_labels

    def test_search_page_has_text_input(self):
        """Page creates a text input widget."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(search_results=[])

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        mock_st.text_input.assert_called_once()
        call_args = mock_st.text_input.call_args
        assert call_args[0][0] == "Search Query"

    def test_search_page_options_expander(self):
        """Page creates a Search Options expander."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(search_results=[])

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        mock_st.expander.assert_called_with("Search Options")

    def test_search_page_source_type_selectbox(self):
        """Source type selectbox offers All, Paper, Textbook."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(search_results=[])

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        # Find the Source Type selectbox call
        selectbox_calls = mock_st.selectbox.call_args_list
        source_call = None
        for c in selectbox_calls:
            if c[0][0] == "Source Type":
                source_call = c
                break

        assert source_call is not None
        options = source_call[0][1]
        assert options == ["All", "Paper", "Textbook"]

    def test_search_page_context_type_selectbox(self):
        """Context type selectbox offers balanced, building, auditing."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(search_results=[])

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        selectbox_calls = mock_st.selectbox.call_args_list
        context_call = None
        for c in selectbox_calls:
            if c[0][0] == "Search Context":
                context_call = c
                break

        assert context_call is not None
        options = context_call[0][1]
        assert "balanced" in options
        assert "building" in options
        assert "auditing" in options

    def test_search_page_limit_slider(self):
        """Limit slider has range 5-50, default 20."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(search_results=[])

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        slider_calls = mock_st.slider.call_args_list
        limit_call = None
        for c in slider_calls:
            if c[0][0] == "Max Results":
                limit_call = c
                break

        assert limit_call is not None
        assert limit_call[1]["min_value"] == 5
        assert limit_call[1]["max_value"] == 50
        assert limit_call[1]["value"] == 20


# ---------------------------------------------------------------------------
# Search Results Tests
# ---------------------------------------------------------------------------


class TestSearchResults:
    """Test search result display and error handling."""

    def test_search_page_with_results(self, mock_search_results):
        """Submitting a query shows formatted results with success message."""
        mock_st = _make_mock_st(query_value="instrumental variables")
        mock_client = _make_mock_client(search_results=mock_search_results)

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        # Should show success message with count
        mock_st.success.assert_called_once()
        success_msg = mock_st.success.call_args[0][0]
        assert "1" in success_msg

    def test_search_page_result_scores(self, mock_search_results):
        """Result display includes score breakdown metrics."""
        mock_st = _make_mock_st(query_value="instrumental variables")
        mock_client = _make_mock_client(search_results=mock_search_results)

        # Capture metric calls from column mocks
        col_metrics = []
        original_columns = mock_st.columns.side_effect

        def columns_with_tracking(n):
            cols = original_columns(n)
            for col in cols:
                original_metric = col.metric

                def tracking_metric(*args, **kwargs):
                    col_metrics.append(args)

                col.metric = tracking_metric
            return cols

        mock_st.columns.side_effect = columns_with_tracking

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        # Should have 4 metric calls (FTS, Vector, Graph, Citation)
        metric_labels = [m[0] for m in col_metrics]
        assert "FTS" in metric_labels
        assert "Vector" in metric_labels
        assert "Graph" in metric_labels
        assert "Citation" in metric_labels

    def test_search_page_no_results(self):
        """Warning shown when search returns empty."""
        mock_st = _make_mock_st(query_value="xyznonexistent")
        mock_client = _make_mock_client(search_results=[])

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        mock_st.warning.assert_called_once()
        warning_msg = mock_st.warning.call_args[0][0]
        assert "no results" in warning_msg.lower()

    def test_search_page_connect_error(self):
        """API connection error shows error message."""
        import httpx

        mock_st = _make_mock_st(query_value="test query")
        mock_client = _make_mock_client(search_error=httpx.ConnectError("Connection refused"))

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        mock_st.error.assert_called()
        error_msg = mock_st.error.call_args[0][0]
        assert "api" in error_msg.lower() or "connect" in error_msg.lower()

    def test_search_page_generic_error(self):
        """Non-connection error shows error message."""
        mock_st = _make_mock_st(query_value="test query")
        mock_client = _make_mock_client(search_error=RuntimeError("Something went wrong"))

        with (
            patch("research_kb_dashboard.pages.search.st", mock_st),
            patch(
                "research_kb_dashboard.pages.search.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.search import search_page

            search_page()

        mock_st.error.assert_called()
        error_msg = mock_st.error.call_args[0][0]
        assert "error" in error_msg.lower() or "wrong" in error_msg.lower()
