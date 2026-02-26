"""Tests for queue page.

Tests the extraction queue status page by mocking Streamlit and the API client,
then calling the page function directly to verify progress display, source
listing, and error handling.
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
    """Stats response with extraction progress data."""
    return {
        "sources": 495,
        "chunks": 228000,
        "concepts": 312000,
        "relationships": 744000,
        "citations": 10758,
        "chunk_concepts": 600000,
    }


@pytest.fixture
def sample_sources():
    """Sources list for queue display."""
    return {
        "sources": [
            {
                "id": str(uuid4()),
                "title": "Causal Inference Methods",
                "source_type": "paper",
            },
            {
                "id": str(uuid4()),
                "title": "Machine Learning Textbook",
                "source_type": "textbook",
            },
            {
                "id": str(uuid4()),
                "title": "Time Series Analysis",
                "source_type": "paper",
            },
        ],
    }


def _make_mock_client(stats=None, sources_data=None, error=None):
    """Create a mock ResearchKBClient for queue page testing."""
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
    """Create a mock streamlit module for queue page testing."""
    mock_st = MagicMock()

    # selectbox returns first option
    mock_st.selectbox.side_effect = lambda label, options, **kw: options[kw.get("index", 0)]

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
# format_time Tests
# ---------------------------------------------------------------------------


class TestFormatTime:
    """Test the format_time utility function."""

    def test_minutes(self):
        """Short durations shown as minutes."""
        from research_kb_dashboard.pages.queue import format_time

        assert format_time(30) == "30 min"

    def test_hours(self):
        """Durations >= 60 min shown as hours."""
        from research_kb_dashboard.pages.queue import format_time

        result = format_time(120)
        assert "hours" in result
        assert "2.0" in result

    def test_days(self):
        """Durations >= 1440 min shown as days."""
        from research_kb_dashboard.pages.queue import format_time

        result = format_time(2880)
        assert "days" in result
        assert "2.0" in result

    def test_zero(self):
        """Zero minutes."""
        from research_kb_dashboard.pages.queue import format_time

        assert format_time(0) == "0 min"


# ---------------------------------------------------------------------------
# Rendering Tests
# ---------------------------------------------------------------------------


class TestQueuePageRendering:
    """Test basic page rendering and progress display."""

    def test_queue_page_renders(self, sample_stats, sample_sources):
        """Page runs without exception when API is mocked."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.queue.st", mock_st),
            patch(
                "research_kb_dashboard.pages.queue.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.queue.pd") as mock_pd,
        ):
            mock_pd.DataFrame.return_value = MagicMock()

            from research_kb_dashboard.pages.queue import queue_page

            queue_page()

        mock_st.header.assert_called_once()

    def test_overall_progress_metrics(self, sample_stats, sample_sources):
        """Page displays overall progress metric cards."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        # Capture metric calls
        metric_calls = []
        original_make_cols = mock_st.columns.side_effect

        def tracking_cols(n):
            cols = original_make_cols(n)
            for col in cols:

                def capture_metric(*args, **kwargs):
                    metric_calls.append(args)

                col.metric = capture_metric
            return cols

        mock_st.columns.side_effect = tracking_cols

        with (
            patch("research_kb_dashboard.pages.queue.st", mock_st),
            patch(
                "research_kb_dashboard.pages.queue.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.queue.pd") as mock_pd,
        ):
            mock_pd.DataFrame.return_value = MagicMock()

            from research_kb_dashboard.pages.queue import queue_page

            queue_page()

        metric_labels = [m[0] for m in metric_calls]
        assert "Total Chunks" in metric_labels
        assert "Processed (est.)" in metric_labels
        assert "Remaining (est.)" in metric_labels
        assert "Complete (est.)" in metric_labels

    def test_progress_bar(self, sample_stats, sample_sources):
        """Progress bar is rendered."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.queue.st", mock_st),
            patch(
                "research_kb_dashboard.pages.queue.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.queue.pd") as mock_pd,
        ):
            mock_pd.DataFrame.return_value = MagicMock()

            from research_kb_dashboard.pages.queue import queue_page

            queue_page()

        mock_st.progress.assert_called_once()
        progress_value = mock_st.progress.call_args[0][0]
        assert 0 <= progress_value <= 1.0

    def test_eta_caption(self, sample_stats, sample_sources):
        """ETA caption is shown with throughput estimate."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.queue.st", mock_st),
            patch(
                "research_kb_dashboard.pages.queue.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.queue.pd") as mock_pd,
        ):
            mock_pd.DataFrame.return_value = MagicMock()

            from research_kb_dashboard.pages.queue import queue_page

            queue_page()

        caption_calls = [c[0][0] for c in mock_st.caption.call_args_list]
        assert any("chunks/min" in c for c in caption_calls)

    def test_source_type_filter(self, sample_stats, sample_sources):
        """Source type filter selectbox is present."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.queue.st", mock_st),
            patch(
                "research_kb_dashboard.pages.queue.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.queue.pd") as mock_pd,
        ):
            mock_pd.DataFrame.return_value = MagicMock()

            from research_kb_dashboard.pages.queue import queue_page

            queue_page()

        selectbox_calls = mock_st.selectbox.call_args_list
        filter_call = None
        for c in selectbox_calls:
            if "type" in c[0][0].lower() or "filter" in c[0][0].lower():
                filter_call = c
                break

        assert filter_call is not None
        options = filter_call[0][1]
        assert "All" in options

    def test_dataframe_displayed(self, sample_stats, sample_sources):
        """Source dataframe is rendered."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.queue.st", mock_st),
            patch(
                "research_kb_dashboard.pages.queue.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.queue.pd") as mock_pd,
        ):
            mock_df = MagicMock()
            mock_df.__getitem__ = MagicMock(return_value=mock_df)
            mock_df.rename = MagicMock(return_value=mock_df)
            mock_pd.DataFrame.return_value = mock_df

            from research_kb_dashboard.pages.queue import queue_page

            queue_page()

        mock_st.dataframe.assert_called()

    def test_extraction_commands_shown(self, sample_stats, sample_sources):
        """Extraction command code blocks are rendered."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data=sample_sources)

        with (
            patch("research_kb_dashboard.pages.queue.st", mock_st),
            patch(
                "research_kb_dashboard.pages.queue.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.queue.pd") as mock_pd,
        ):
            mock_df = MagicMock()
            mock_df.__getitem__ = MagicMock(return_value=mock_df)
            mock_df.rename = MagicMock(return_value=mock_df)
            mock_pd.DataFrame.return_value = mock_df

            from research_kb_dashboard.pages.queue import queue_page

            queue_page()

        code_calls = [c[0][0] for c in mock_st.code.call_args_list]
        assert any("extract_concepts" in c for c in code_calls)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestQueueEdgeCases:
    """Test edge cases and empty data."""

    def test_empty_sources(self, sample_stats):
        """No sources shows info message."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=sample_stats, sources_data={"sources": []})

        with (
            patch("research_kb_dashboard.pages.queue.st", mock_st),
            patch(
                "research_kb_dashboard.pages.queue.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.queue import queue_page

            queue_page()

        info_calls = [c[0][0] for c in mock_st.info.call_args_list]
        assert any("no sources" in msg.lower() or "filter" in msg.lower() for msg in info_calls)

    def test_zero_chunks(self):
        """Zero chunks handles division correctly."""
        zero_stats = {
            "sources": 0,
            "chunks": 0,
            "concepts": 0,
            "relationships": 0,
            "chunk_concepts": 0,
        }
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(stats=zero_stats, sources_data={"sources": []})

        with (
            patch("research_kb_dashboard.pages.queue.st", mock_st),
            patch(
                "research_kb_dashboard.pages.queue.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.queue import queue_page

            queue_page()

        # Should not crash on zero division
        mock_st.progress.assert_called_once()


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestQueueErrors:
    """Test error handling."""

    def test_connect_error(self):
        """API connection error shows error and startup hint."""
        import httpx

        mock_st = _make_mock_st()
        mock_client = _make_mock_client(error=httpx.ConnectError("Connection refused"))

        with (
            patch("research_kb_dashboard.pages.queue.st", mock_st),
            patch(
                "research_kb_dashboard.pages.queue.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.queue import queue_page

            queue_page()

        mock_st.error.assert_called()
        error_msg = mock_st.error.call_args[0][0]
        assert "api" in error_msg.lower() or "connect" in error_msg.lower()

    def test_generic_error(self):
        """Non-connection error shows error message."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client(error=RuntimeError("Database error"))

        with (
            patch("research_kb_dashboard.pages.queue.st", mock_st),
            patch(
                "research_kb_dashboard.pages.queue.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.queue import queue_page

            queue_page()

        mock_st.error.assert_called()
