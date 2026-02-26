"""Tests for assumptions page.

Tests the assumption auditing page by mocking Streamlit and the API client,
then calling the page function directly to verify method selection, assumption
display, and error handling.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audit_result():
    """Assumption audit API response for instrumental variables."""
    return {
        "method": "instrumental variables",
        "assumptions": [
            {
                "name": "Relevance Condition",
                "description": "The instrument must be correlated with the endogenous regressor.",
                "importance": "critical",
                "testable": True,
                "sources": [
                    {
                        "source_title": "Econometric Analysis",
                        "content": "The relevance condition requires that Cov(Z,X) != 0...",
                    },
                    {
                        "source_title": "Mostly Harmless Econometrics",
                        "content": "First-stage F-statistic should exceed 10...",
                    },
                ],
            },
            {
                "name": "Exclusion Restriction",
                "description": "The instrument affects the outcome only through the treatment.",
                "importance": "critical",
                "testable": False,
                "sources": [
                    {
                        "source_title": "Causal Inference for Statistics",
                        "content": "The exclusion restriction is fundamentally untestable...",
                    },
                ],
            },
            {
                "name": "Independence",
                "description": "The instrument is independent of unobserved confounders.",
                "importance": "high",
                "testable": False,
                "sources": [],
            },
        ],
    }


@pytest.fixture
def empty_audit_result():
    """Audit response with no cached assumptions."""
    return {
        "method": "novel method",
        "assumptions": [],
    }


def _make_mock_http_client(audit_response=None, error=None):
    """Create a mock httpx client for assumption audit endpoint."""
    mock_http = AsyncMock()

    if error:
        mock_http.get.side_effect = error
    else:
        response = MagicMock()
        response.json.return_value = audit_response or {"method": "", "assumptions": []}
        response.raise_for_status = MagicMock()
        mock_http.get.return_value = response

    return mock_http


def _make_mock_api_client(http_client=None, error=None):
    """Create a mock ResearchKBClient for assumptions page."""
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client._get_client = AsyncMock(return_value=http_client)

    if error:
        mock_client._get_client.side_effect = error

    return mock_client


def _make_mock_st(method_input=""):
    """Create a mock streamlit module for assumptions page testing."""
    mock_st = MagicMock()

    mock_st.text_input.return_value = method_input

    # button returns False by default (no quick-select clicked)
    mock_st.button.return_value = False

    # columns context manager
    def make_cols(spec):
        n = spec if isinstance(spec, int) else len(spec)
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

    # expander context manager
    def make_expander(label, **kwargs):
        exp = MagicMock()
        exp.__enter__ = MagicMock(return_value=exp)
        exp.__exit__ = MagicMock(return_value=False)
        return exp

    mock_st.expander.side_effect = make_expander

    return mock_st


# ---------------------------------------------------------------------------
# Rendering Tests
# ---------------------------------------------------------------------------


class TestAssumptionsPageRendering:
    """Test basic page rendering."""

    def test_assumptions_page_renders(self):
        """Page runs without exception when no method input."""
        mock_st = _make_mock_st(method_input="")

        with patch("research_kb_dashboard.pages.assumptions.st", mock_st):
            from research_kb_dashboard.pages.assumptions import assumptions_page

            assumptions_page()

        mock_st.header.assert_called_once()
        header_arg = mock_st.header.call_args[0][0]
        assert "Assumption" in header_arg

    def test_page_shows_quick_select_buttons(self):
        """Page shows quick-select buttons for common methods."""
        mock_st = _make_mock_st(method_input="")

        with patch("research_kb_dashboard.pages.assumptions.st", mock_st):
            from research_kb_dashboard.pages.assumptions import assumptions_page

            assumptions_page()

        button_calls = mock_st.button.call_args_list
        button_labels = [c[0][0] for c in button_calls]
        assert "instrumental variables" in button_labels
        assert "double machine learning" in button_labels

    def test_empty_input_shows_info(self):
        """No method input shows info message."""
        mock_st = _make_mock_st(method_input="")

        with patch("research_kb_dashboard.pages.assumptions.st", mock_st):
            from research_kb_dashboard.pages.assumptions import assumptions_page

            assumptions_page()

        mock_st.info.assert_called()
        info_msg = mock_st.info.call_args[0][0]
        assert "method" in info_msg.lower()

    def test_common_methods_list(self):
        """COMMON_METHODS contains expected methods."""
        from research_kb_dashboard.pages.assumptions import COMMON_METHODS

        assert "instrumental variables" in COMMON_METHODS
        assert "double machine learning" in COMMON_METHODS
        assert "difference-in-differences" in COMMON_METHODS
        assert len(COMMON_METHODS) == 10


# ---------------------------------------------------------------------------
# Assumption Display Tests
# ---------------------------------------------------------------------------


class TestAssumptionDisplay:
    """Test assumption audit result rendering."""

    def test_audit_with_assumptions(self, sample_audit_result):
        """Successful audit shows assumption count and expanders."""
        mock_st = _make_mock_st(method_input="instrumental variables")
        mock_http = _make_mock_http_client(audit_response=sample_audit_result)
        mock_client = _make_mock_api_client(http_client=mock_http)

        with (
            patch("research_kb_dashboard.pages.assumptions.st", mock_st),
            patch(
                "research_kb_dashboard.pages.assumptions.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.assumptions import assumptions_page

            assumptions_page()

        mock_st.success.assert_called()
        success_msg = mock_st.success.call_args[0][0]
        assert "3" in success_msg

        # Expander created for each assumption
        assert mock_st.expander.call_count == 3

    def test_assumption_names_in_expanders(self, sample_audit_result):
        """Each assumption name appears in its expander label."""
        mock_st = _make_mock_st(method_input="instrumental variables")
        mock_http = _make_mock_http_client(audit_response=sample_audit_result)
        mock_client = _make_mock_api_client(http_client=mock_http)

        with (
            patch("research_kb_dashboard.pages.assumptions.st", mock_st),
            patch(
                "research_kb_dashboard.pages.assumptions.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.assumptions import assumptions_page

            assumptions_page()

        expander_labels = [c[0][0] for c in mock_st.expander.call_args_list]
        assert any("Relevance Condition" in label for label in expander_labels)
        assert any("Exclusion Restriction" in label for label in expander_labels)
        assert any("Independence" in label for label in expander_labels)

    def test_method_subheader(self, sample_audit_result):
        """Method name appears in subheader."""
        mock_st = _make_mock_st(method_input="instrumental variables")
        mock_http = _make_mock_http_client(audit_response=sample_audit_result)
        mock_client = _make_mock_api_client(http_client=mock_http)

        with (
            patch("research_kb_dashboard.pages.assumptions.st", mock_st),
            patch(
                "research_kb_dashboard.pages.assumptions.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.assumptions import assumptions_page

            assumptions_page()

        subheader_calls = [c[0][0] for c in mock_st.subheader.call_args_list]
        assert any("instrumental variables" in s for s in subheader_calls)

    def test_download_button(self, sample_audit_result):
        """Download JSON button is rendered."""
        mock_st = _make_mock_st(method_input="instrumental variables")
        mock_http = _make_mock_http_client(audit_response=sample_audit_result)
        mock_client = _make_mock_api_client(http_client=mock_http)

        with (
            patch("research_kb_dashboard.pages.assumptions.st", mock_st),
            patch(
                "research_kb_dashboard.pages.assumptions.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.assumptions import assumptions_page

            assumptions_page()

        mock_st.download_button.assert_called_once()
        dl_kwargs = mock_st.download_button.call_args
        assert (
            "json" in dl_kwargs[1].get("mime", dl_kwargs[0][0] if dl_kwargs[0] else "").lower()
            or "json" in str(dl_kwargs).lower()
        )

    def test_empty_assumptions_shows_info(self, empty_audit_result):
        """No assumptions shows info with extraction hint."""
        mock_st = _make_mock_st(method_input="novel method")
        mock_http = _make_mock_http_client(audit_response=empty_audit_result)
        mock_client = _make_mock_api_client(http_client=mock_http)

        with (
            patch("research_kb_dashboard.pages.assumptions.st", mock_st),
            patch(
                "research_kb_dashboard.pages.assumptions.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.assumptions import assumptions_page

            assumptions_page()

        info_calls = [c[0][0] for c in mock_st.info.call_args_list]
        assert any("no cached" in msg.lower() or "extract" in msg.lower() for msg in info_calls)

    def test_partial_result_with_error(self):
        """Audit result with error field shows warning."""
        partial_result = {
            "method": "test method",
            "error": "Timeout on Ollama fallback",
            "assumptions": [{"name": "A1", "description": "Some assumption"}],
        }
        mock_st = _make_mock_st(method_input="test method")
        mock_http = _make_mock_http_client(audit_response=partial_result)
        mock_client = _make_mock_api_client(http_client=mock_http)

        with (
            patch("research_kb_dashboard.pages.assumptions.st", mock_st),
            patch(
                "research_kb_dashboard.pages.assumptions.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.assumptions import assumptions_page

            assumptions_page()

        mock_st.warning.assert_called()


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestAssumptionsErrors:
    """Test error handling."""

    def test_connect_error_with_cli_fallback(self):
        """API connection error triggers CLI fallback attempt."""
        import httpx

        mock_st = _make_mock_st(method_input="instrumental variables")

        # Make ResearchKBClient raise ConnectError
        with (
            patch("research_kb_dashboard.pages.assumptions.st", mock_st),
            patch(
                "research_kb_dashboard.pages.assumptions.run_async",
                side_effect=[
                    httpx.ConnectError("Connection refused"),
                    {"method": "instrumental variables", "assumptions": []},
                ],
            ),
        ):
            from research_kb_dashboard.pages.assumptions import assumptions_page

            assumptions_page()

        # Should show connect error
        mock_st.error.assert_called()

    def test_generic_error(self):
        """Non-connection error shows error message."""
        mock_st = _make_mock_st(method_input="test")

        with (
            patch("research_kb_dashboard.pages.assumptions.st", mock_st),
            patch(
                "research_kb_dashboard.pages.assumptions.run_async",
                side_effect=RuntimeError("Database error"),
            ),
        ):
            from research_kb_dashboard.pages.assumptions import assumptions_page

            assumptions_page()

        mock_st.error.assert_called()
