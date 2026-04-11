"""Unit tests for scripts/retry_dlq.py metadata resolution.

Issue #6: ensure DLQ retry reads stored metadata + sidecar fallback and
passes the full argument set to ``PDFDispatcher.ingest_pdf``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import retry_dlq
from research_kb_contracts import SourceType
from research_kb_pdf.dlq import DLQEntry

pytestmark = pytest.mark.unit


def _make_entry(file_path: Path, metadata: dict | None = None) -> DLQEntry:
    return DLQEntry(
        id="00000000-0000-0000-0000-000000000001",
        file_path=str(file_path),
        error_type="TestError",
        error_message="simulated failure",
        traceback="...",
        timestamp="2026-04-11T12:00:00+00:00",
        retry_count=0,
        metadata=metadata or {},
    )


class TestResolveMetadata:
    """_resolve_metadata priority: entry.metadata > sidecar > defaults."""

    def test_uses_entry_metadata_when_present(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        entry = _make_entry(
            pdf,
            metadata={
                "source_type": "paper",
                "title": "Metadata Title",
                "domain_id": "econometrics",
            },
        )

        resolved = retry_dlq._resolve_metadata(entry)
        assert resolved["source_type"] == SourceType.PAPER
        assert resolved["title"] == "Metadata Title"
        assert resolved["domain_id"] == "econometrics"
        assert resolved["pdf_path"] == pdf

    def test_falls_back_to_sidecar_when_metadata_missing(self, tmp_path):
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        sidecar = tmp_path / "book.json"
        sidecar.write_text(
            json.dumps(
                {
                    "source_type": "textbook",
                    "title": "Sidecar Title",
                    "domain_id": "linear_algebra",
                    "authors": ["Strang"],
                    "year": 2016,
                }
            )
        )

        entry = _make_entry(pdf, metadata={})
        resolved = retry_dlq._resolve_metadata(entry)
        assert resolved["source_type"] == SourceType.TEXTBOOK
        assert resolved["title"] == "Sidecar Title"
        assert resolved["domain_id"] == "linear_algebra"
        assert resolved["authors"] == ["Strang"]
        assert resolved["year"] == 2016

    def test_sidecar_double_suffix_also_resolves(self, tmp_path):
        pdf = tmp_path / "arxiv_paper.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        # Alternative sidecar location: paper.pdf.json
        sidecar = tmp_path / "arxiv_paper.pdf.json"
        sidecar.write_text(json.dumps({"title": "Double Suffix Title", "domain_id": "ml"}))

        entry = _make_entry(pdf, metadata={})
        resolved = retry_dlq._resolve_metadata(entry)
        assert resolved["title"] == "Double Suffix Title"
        assert resolved["domain_id"] == "ml"

    def test_defaults_when_nothing_found(self, tmp_path):
        pdf = tmp_path / "lonely.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        entry = _make_entry(pdf, metadata={})
        resolved = retry_dlq._resolve_metadata(entry)
        # Default source_type is PAPER, domain_id is causal_inference,
        # title falls back to the PDF stem.
        assert resolved["source_type"] == SourceType.PAPER
        assert resolved["domain_id"] == "causal_inference"
        assert resolved["title"] == "lonely"

    def test_entry_metadata_takes_precedence_over_sidecar(self, tmp_path):
        pdf = tmp_path / "conflict.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        sidecar = tmp_path / "conflict.json"
        sidecar.write_text(json.dumps({"title": "Sidecar Title"}))

        entry = _make_entry(pdf, metadata={"title": "Metadata Title"})
        resolved = retry_dlq._resolve_metadata(entry)
        assert resolved["title"] == "Metadata Title"

    def test_invalid_source_type_normalized_to_paper(self, tmp_path):
        pdf = tmp_path / "weird.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        entry = _make_entry(pdf, metadata={"source_type": "banana"})
        resolved = retry_dlq._resolve_metadata(entry)
        assert resolved["source_type"] == SourceType.PAPER


class TestRetryEntry:
    """retry_entry calls ingest_pdf with resolved metadata and removes on success."""

    async def test_success_removes_entry(self, tmp_path):
        pdf = tmp_path / "ok.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        entry = _make_entry(
            pdf,
            metadata={
                "source_type": "paper",
                "title": "Success Title",
                "domain_id": "causal_inference",
            },
        )

        dlq = MagicMock()
        dlq.remove = MagicMock(return_value=True)

        mock_result = MagicMock()
        mock_result.source = MagicMock(id="00000000-0000-0000-0000-000000000002")

        mock_dispatcher = MagicMock()
        mock_dispatcher.ingest_pdf = AsyncMock(return_value=mock_result)

        with patch(
            "research_kb_pdf.dispatcher.PDFDispatcher",
            return_value=mock_dispatcher,
        ):
            ok = await retry_dlq.retry_entry(dlq, entry)

        assert ok is True
        dlq.remove.assert_called_once_with(entry.id)
        mock_dispatcher.ingest_pdf.assert_awaited_once()
        call_kwargs = mock_dispatcher.ingest_pdf.await_args.kwargs
        assert call_kwargs["pdf_path"] == pdf
        assert call_kwargs["source_type"] == SourceType.PAPER
        assert call_kwargs["title"] == "Success Title"
        assert call_kwargs["domain_id"] == "causal_inference"

    async def test_missing_file_returns_false(self, tmp_path):
        pdf = tmp_path / "gone.pdf"  # deliberately not created

        entry = _make_entry(pdf, metadata={"title": "Ghost", "domain_id": "x"})

        dlq = MagicMock()
        ok = await retry_dlq.retry_entry(dlq, entry)
        assert ok is False
        dlq.remove.assert_not_called()

    async def test_exception_returns_false_and_keeps_entry(self, tmp_path):
        pdf = tmp_path / "boom.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        entry = _make_entry(
            pdf,
            metadata={
                "source_type": "paper",
                "title": "Boom",
                "domain_id": "causal_inference",
            },
        )

        dlq = MagicMock()
        mock_dispatcher = MagicMock()
        mock_dispatcher.ingest_pdf = AsyncMock(side_effect=RuntimeError("kaboom"))

        with patch(
            "research_kb_pdf.dispatcher.PDFDispatcher",
            return_value=mock_dispatcher,
        ):
            ok = await retry_dlq.retry_entry(dlq, entry)

        assert ok is False
        dlq.remove.assert_not_called()
