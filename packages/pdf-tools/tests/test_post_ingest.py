"""Tests for post-ingestion hooks (Issue #5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from research_kb_pdf.post_ingest import run_post_ingest_hooks

pytestmark = pytest.mark.unit


class TestRunPostIngestHooks:
    """run_post_ingest_hooks routes to the right storage helpers."""

    async def test_build_citations_called_with_source_ids(self):
        """Default run invokes build_citation_graph scoped to the new IDs."""
        ids = [uuid4(), uuid4(), uuid4()]

        with patch(
            "research_kb_storage.build_citation_graph",
            new_callable=AsyncMock,
            return_value={"total_processed": 5, "matched": 3, "unmatched": 2},
        ) as mock_build:
            summary = await run_post_ingest_hooks(ids)

        mock_build.assert_awaited_once_with(source_ids=ids)
        assert summary["source_count"] == 3
        assert summary["citations"]["matched"] == 3
        assert "pagerank" not in summary

    async def test_pagerank_opt_in(self):
        """compute_pagerank=True triggers the authority computation."""
        ids = [uuid4()]

        with (
            patch(
                "research_kb_storage.build_citation_graph",
                new_callable=AsyncMock,
                return_value={"matched": 0},
            ),
            patch(
                "research_kb_storage.compute_pagerank_authority",
                new_callable=AsyncMock,
                return_value={"min": 0.1, "max": 0.9, "mean": 0.3},
            ) as mock_pagerank,
        ):
            summary = await run_post_ingest_hooks(ids, compute_pagerank=True)

        mock_pagerank.assert_awaited_once()
        assert summary["pagerank"]["mean"] == pytest.approx(0.3)

    async def test_build_citations_false_skips_hook(self):
        """build_citations=False short-circuits the citation build."""
        ids = [uuid4()]

        with patch(
            "research_kb_storage.build_citation_graph",
            new_callable=AsyncMock,
        ) as mock_build:
            summary = await run_post_ingest_hooks(ids, build_citations=False)

        mock_build.assert_not_called()
        assert "citations" not in summary
        assert summary["source_count"] == 1

    async def test_empty_source_ids_with_hooks_raises(self):
        """Empty source_ids with hooks requested is a caller bug."""
        with pytest.raises(ValueError, match="at least one source_id"):
            await run_post_ingest_hooks([], build_citations=True)

    async def test_empty_source_ids_no_hooks_noop(self):
        """Empty source_ids with no hooks requested is a clean no-op."""
        summary = await run_post_ingest_hooks([], build_citations=False, compute_pagerank=False)
        assert summary == {}
