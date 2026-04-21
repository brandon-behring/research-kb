"""Tests for ``scripts/split_and_reingest.py``.

Unit tests cover the pure-logic pieces (``compute_page_ranges``, CLI
collection) without touching disk or DB. Integration tests use a real
18-page fixture PDF (``fixtures/papers/Black_Scholes_1973_Option_Pricing.pdf``)
to exercise the poppler subprocess wrappers end-to-end.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "split_and_reingest.py"
_spec = importlib.util.spec_from_file_location("split_and_reingest", _SCRIPT_PATH)
split_mod = importlib.util.module_from_spec(_spec)
sys.modules["split_and_reingest"] = split_mod
_spec.loader.exec_module(split_mod)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Unit: compute_page_ranges
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputePageRanges:
    def test_even_split(self):
        assert split_mod.compute_page_ranges(12, 4) == [(1, 3), (4, 6), (7, 9), (10, 12)]

    def test_uneven_split_distributes_remainder_to_early_parts(self):
        """10 pages / 3 parts = 3.33 pp/part. Extra page goes to part 1."""
        ranges = split_mod.compute_page_ranges(10, 3)
        assert ranges == [(1, 4), (5, 7), (8, 10)]
        # Each part differs by at most 1 page from the others.
        sizes = [end - start + 1 for start, end in ranges]
        assert max(sizes) - min(sizes) <= 1

    def test_ranges_are_contiguous_and_cover_full_doc(self):
        ranges = split_mod.compute_page_ranges(1180, 4)
        # First starts at 1, last ends at total, no gaps or overlaps.
        assert ranges[0][0] == 1
        assert ranges[-1][1] == 1180
        for (a_start, a_end), (b_start, b_end) in zip(ranges, ranges[1:]):
            assert b_start == a_end + 1, f"gap or overlap between {a_end} and {b_start}"

    def test_single_part_returns_whole_doc(self):
        assert split_mod.compute_page_ranges(42, 1) == [(1, 42)]

    def test_rejects_non_positive_pages(self):
        with pytest.raises(ValueError, match="total_pages must be positive"):
            split_mod.compute_page_ranges(0, 4)
        with pytest.raises(ValueError, match="total_pages must be positive"):
            split_mod.compute_page_ranges(-1, 4)

    def test_rejects_non_positive_parts(self):
        with pytest.raises(ValueError, match="parts must be positive"):
            split_mod.compute_page_ranges(100, 0)

    def test_rejects_parts_greater_than_pages(self):
        with pytest.raises(ValueError, match="cannot split"):
            split_mod.compute_page_ranges(3, 4)


# ---------------------------------------------------------------------------
# Unit: sha256_file + _collect_source_ids
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSha256File:
    def test_known_vector(self, tmp_path: Path):
        p = tmp_path / "hello.txt"
        p.write_text("hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert split_mod.sha256_file(p) == expected

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert split_mod.sha256_file(p) == hashlib.sha256(b"").hexdigest()


@pytest.mark.unit
class TestCollectSourceIds:
    def _argparse_ns(self, **kw):
        import argparse

        ns = argparse.Namespace(source_id=None, from_dlq=None)
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def test_single_source_id(self):
        args = self._argparse_ns(source_id="abc-123")
        assert split_mod._collect_source_ids(args) == ["abc-123"]

    def test_from_dlq_dedupes_retry_entries(self, tmp_path: Path):
        dlq = tmp_path / "dlq.jsonl"
        dlq.write_text(
            "\n".join(
                [
                    json.dumps({"source_id": "a", "error": "first attempt"}),
                    json.dumps({"source_id": "b", "error": "first b"}),
                    json.dumps({"source_id": "a", "error": "second attempt"}),  # dup
                    json.dumps({"source_id": "c", "error": "first c"}),
                ]
            )
        )
        args = self._argparse_ns(from_dlq=str(dlq))
        assert split_mod._collect_source_ids(args) == ["a", "b", "c"]

    def test_from_dlq_ignores_empty_lines(self, tmp_path: Path):
        dlq = tmp_path / "dlq.jsonl"
        dlq.write_text(
            "\n".join(
                [
                    json.dumps({"source_id": "a"}),
                    "",
                    "   ",
                    json.dumps({"source_id": "b"}),
                ]
            )
        )
        args = self._argparse_ns(from_dlq=str(dlq))
        assert split_mod._collect_source_ids(args) == ["a", "b"]

    def test_raises_when_nothing_specified(self):
        args = self._argparse_ns()
        with pytest.raises(SystemExit):
            split_mod._collect_source_ids(args)


# ---------------------------------------------------------------------------
# Integration: real PDF via poppler-utils
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pdf_18pp() -> Path:
    """An 18-page test fixture PDF. Skip the test if missing."""
    path = _REPO_ROOT / "fixtures" / "papers" / "Black_Scholes_1973_Option_Pricing.pdf"
    if not path.exists():
        pytest.skip(f"fixture PDF not available: {path}")
    return path


@pytest.mark.integration
class TestPopplerWrappers:
    """Exercises pdfseparate + pdfunite + pdfinfo against a real fixture."""

    def test_count_pdf_pages_reports_expected(self, sample_pdf_18pp: Path):
        assert split_mod.count_pdf_pages(sample_pdf_18pp) == 18

    def test_count_pdf_pages_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="pdfinfo failed|does not exist|No such"):
            split_mod.count_pdf_pages(tmp_path / "does_not_exist.pdf")

    def test_split_range_produces_correct_page_count(self, sample_pdf_18pp: Path, tmp_path: Path):
        out = tmp_path / "part_1to6.pdf"
        split_mod.split_pdf_range(sample_pdf_18pp, 1, 6, out)
        assert out.exists()
        assert split_mod.count_pdf_pages(out) == 6

    def test_split_into_3_parts_round_trips(self, sample_pdf_18pp: Path, tmp_path: Path):
        """Split 18 pages into 3 parts of 6 each; verify no page loss."""
        ranges = split_mod.compute_page_ranges(18, 3)
        assert ranges == [(1, 6), (7, 12), (13, 18)]

        part_sizes = []
        for i, (start, end) in enumerate(ranges, 1):
            out = tmp_path / f"part{i}.pdf"
            split_mod.split_pdf_range(sample_pdf_18pp, start, end, out)
            part_sizes.append(split_mod.count_pdf_pages(out))
        assert part_sizes == [6, 6, 6]
        assert sum(part_sizes) == 18

    def test_split_rejects_invalid_range(self, sample_pdf_18pp: Path, tmp_path: Path):
        with pytest.raises(ValueError, match="invalid range"):
            split_mod.split_pdf_range(sample_pdf_18pp, 5, 3, tmp_path / "x.pdf")
        with pytest.raises(ValueError, match="invalid range"):
            split_mod.split_pdf_range(sample_pdf_18pp, 0, 5, tmp_path / "x.pdf")
