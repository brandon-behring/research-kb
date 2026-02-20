"""Tests for PyMuPDF extraction with real PDF fixture."""

import pytest
from pathlib import Path

from research_kb_pdf.pymupdf_extractor import (
    extract_pdf,
    get_text_with_page_numbers,
    get_full_text,
)

pytestmark = pytest.mark.unit


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TEST_PDF = FIXTURES_DIR / "test_simple.pdf"


class TestPyMuPDFRealExtraction:
    """Test PyMuPDF extraction with real PDF."""

    def test_extract_real_pdf(self):
        """Test extracting real PDF."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)

        # Basic validation
        assert doc.total_pages > 0, "Should have at least 1 page"
        assert doc.total_chars > 0, "Should have extracted some text"
        assert len(doc.pages) == doc.total_pages, "Page count mismatch"
        assert doc.file_path == str(TEST_PDF)

        # Page numbering (1-indexed)
        assert doc.pages[0].page_num == 1, "First page should be numbered 1"
        assert doc.pages[-1].page_num == doc.total_pages, "Last page number should match total"

        # Each page should have text
        for page in doc.pages:
            assert page.char_count >= 0, f"Page {page.page_num} has negative char count"
            assert page.char_count == len(page.text), f"Page {page.page_num} char count mismatch"

        print(f"\n✅ Extracted {doc.total_pages} pages, {doc.total_chars} chars")
        print(f"First page preview: {doc.pages[0].text[:200]}...")

    def test_get_text_with_page_numbers_real(self):
        """Test getting text with page numbers from real PDF."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)
        text_with_pages = get_text_with_page_numbers(doc)

        assert len(text_with_pages) == doc.total_pages
        assert all(isinstance(page_num, int) for page_num, _ in text_with_pages)
        assert all(isinstance(text, str) for _, text in text_with_pages)

        # Page numbers should be sequential starting from 1
        page_nums = [page_num for page_num, _ in text_with_pages]
        assert page_nums == list(range(1, doc.total_pages + 1))

    def test_get_full_text_real(self):
        """Test getting complete document text from real PDF."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)
        full_text = get_full_text(doc)

        # Full text should contain all page text
        assert len(full_text) > 0
        assert len(full_text) >= doc.total_chars - (doc.total_pages * 2)  # Allow for separators

        # Spot check: first page text should be in full text
        if doc.pages[0].text.strip():
            assert doc.pages[0].text[:50] in full_text

        print(f"\n✅ Full text length: {len(full_text)} chars")

    def test_page_content_quality(self):
        """Test extracted content quality (no excessive whitespace)."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)

        for page in doc.pages:
            # No lines should be just whitespace
            lines = page.text.split("\n")
            assert all(
                line.strip() or not line for line in lines
            ), f"Page {page.page_num} has whitespace-only lines"

            # No excessive consecutive newlines (>2)
            assert "\n\n\n\n" not in page.text, f"Page {page.page_num} has excessive newlines"
