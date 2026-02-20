"""Tests for PyMuPDF extraction."""

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
EMPTY_PDF = FIXTURES_DIR / "test_empty.pdf"
ENCRYPTED_PDF = FIXTURES_DIR / "test_encrypted.pdf"
CORRUPTED_PDF = FIXTURES_DIR / "test_corrupted.pdf"


class TestPyMuPDFExtractor:
    """Test PyMuPDF-based PDF extraction."""

    def test_extract_pdf_not_found(self):
        """Test extracting non-existent PDF raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_pdf("nonexistent.pdf")

    def test_extract_pdf_basic(self):
        """Test basic PDF extraction with test PDF."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)

        # Basic validation
        assert doc.total_pages > 0, "Should have at least 1 page"
        assert doc.total_chars > 0, "Should have extracted some text"
        assert len(doc.pages) == doc.total_pages, "Page count mismatch"

    def test_get_text_with_page_numbers(self):
        """Test extracting text with page number tracking."""
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

    def test_get_full_text(self):
        """Test extracting complete document text."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)
        full_text = get_full_text(doc)

        # Full text should contain all page text
        assert len(full_text) > 0
        # First page content should be in full text
        if doc.pages[0].text.strip():
            assert doc.pages[0].text[:50] in full_text


class TestPyMuPDFEdgeCases:
    """Test edge cases and error handling."""

    def test_encrypted_pdf(self):
        """Test encrypted PDF raises ValueError."""
        if not ENCRYPTED_PDF.exists():
            pytest.skip(f"Encrypted PDF fixture not found: {ENCRYPTED_PDF}")

        # Encrypted PDF should raise ValueError when trying to extract without password
        with pytest.raises(ValueError, match="encrypted|password"):
            extract_pdf(ENCRYPTED_PDF)

    def test_corrupted_pdf(self):
        """Test corrupted PDF handling (PyMuPDF attempts recovery)."""
        if not CORRUPTED_PDF.exists():
            pytest.skip(f"Corrupted PDF fixture not found: {CORRUPTED_PDF}")

        # PyMuPDF is lenient with corrupted files - it attempts recovery
        # Rather than raising, it returns what it can extract (usually nothing)
        doc = extract_pdf(CORRUPTED_PDF)

        # Corrupted PDF recovers but with minimal/no content
        assert doc.total_chars == 0, "Corrupted PDF should have no extractable text"

    def test_empty_pdf(self):
        """Test empty PDF (blank page, no text)."""
        if not EMPTY_PDF.exists():
            pytest.skip(f"Empty PDF fixture not found: {EMPTY_PDF}")

        doc = extract_pdf(EMPTY_PDF)

        # Should have 1 page but no text
        assert doc.total_pages == 1, "Empty PDF should have 1 page"
        assert doc.total_chars == 0, "Empty PDF should have no text"
