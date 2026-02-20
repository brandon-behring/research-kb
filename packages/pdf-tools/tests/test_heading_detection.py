"""Tests for heading detection and section tracking."""

import pytest
from pathlib import Path

from research_kb_pdf import (
    extract_pdf,
    detect_headings,
    extract_with_headings,
    chunk_with_sections,
    Heading,
)

pytestmark = pytest.mark.unit


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TEST_PDF = FIXTURES_DIR / "test_simple.pdf"


class TestHeadingDataclass:
    """Test Heading dataclass."""

    def test_heading_creation(self):
        """Test creating a Heading object."""
        heading = Heading(
            text="Introduction",
            level=1,
            page_num=1,
            font_size=18.0,
            char_offset=100,
        )

        assert heading.text == "Introduction"
        assert heading.level == 1
        assert heading.page_num == 1
        assert heading.font_size == 18.0
        assert heading.char_offset == 100


class TestHeadingDetection:
    """Test heading detection via font-size heuristics."""

    def test_detect_headings_returns_list(self):
        """Test detect_headings returns a list."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        headings = detect_headings(TEST_PDF)

        assert isinstance(headings, list)
        # May be empty if no headings detected (uniform font size)

    def test_detect_headings_structure(self):
        """Test detected headings have correct structure."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        headings = detect_headings(TEST_PDF)

        # If headings detected, verify structure
        for heading in headings:
            assert isinstance(heading, Heading)
            assert isinstance(heading.text, str)
            assert 1 <= heading.level <= 3  # H1, H2, or H3
            assert heading.page_num >= 1
            assert heading.font_size > 0
            assert heading.char_offset >= 0
            assert 3 <= len(heading.text) <= 100  # Filter criteria
            assert len(heading.text.split()) <= 15  # ≤15 words

    def test_detect_headings_file_not_found(self):
        """Test detect_headings raises FileNotFoundError for missing PDF."""
        with pytest.raises(FileNotFoundError):
            detect_headings("nonexistent.pdf")

    def test_detect_headings_levels_ordered(self):
        """Test heading levels are consistent with font sizes."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        headings = detect_headings(TEST_PDF)

        if len(headings) < 2:
            pytest.skip("Not enough headings to test ordering")

        # H1 should have larger font than H2, H2 larger than H3
        h1_sizes = [h.font_size for h in headings if h.level == 1]
        h2_sizes = [h.font_size for h in headings if h.level == 2]
        h3_sizes = [h.font_size for h in headings if h.level == 3]

        if h1_sizes and h2_sizes:
            assert min(h1_sizes) >= max(h2_sizes), "H1 should have larger font than H2"

        if h2_sizes and h3_sizes:
            assert min(h2_sizes) >= max(h3_sizes), "H2 should have larger font than H3"


class TestExtractWithHeadings:
    """Test combined extraction + heading detection."""

    def test_extract_with_headings_returns_tuple(self):
        """Test extract_with_headings returns (doc, headings) tuple."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc, headings = extract_with_headings(TEST_PDF)

        assert doc is not None
        assert isinstance(headings, list)
        assert doc.total_pages > 0

    def test_extract_with_headings_doc_matches_extract_pdf(self):
        """Test document from extract_with_headings matches extract_pdf."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc_standard = extract_pdf(TEST_PDF)
        doc_with_headings, _ = extract_with_headings(TEST_PDF)

        assert doc_standard.total_pages == doc_with_headings.total_pages
        assert doc_standard.total_chars == doc_with_headings.total_chars


class TestTextChunkMetadata:
    """Test TextChunk metadata field."""

    def test_textchunk_metadata_default(self):
        """Test TextChunk initializes metadata to empty dict."""
        from research_kb_pdf import TextChunk

        chunk = TextChunk(
            content="Test content",
            start_page=1,
            end_page=1,
            token_count=5,
            char_count=12,
            chunk_index=0,
        )

        assert chunk.metadata == {}
        assert isinstance(chunk.metadata, dict)

    def test_textchunk_metadata_custom(self):
        """Test TextChunk accepts custom metadata."""
        from research_kb_pdf import TextChunk

        chunk = TextChunk(
            content="Test content",
            start_page=1,
            end_page=1,
            token_count=5,
            char_count=12,
            chunk_index=0,
            metadata={"section": "Introduction", "heading_level": 1},
        )

        assert chunk.metadata["section"] == "Introduction"
        assert chunk.metadata["heading_level"] == 1


class TestSectionTracking:
    """Test chunk_with_sections function."""

    def test_chunk_with_sections_basic(self):
        """Test chunk_with_sections returns chunks."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc, headings = extract_with_headings(TEST_PDF)
        chunks = chunk_with_sections(doc, headings)

        assert len(chunks) > 0
        assert all(hasattr(chunk, "metadata") for chunk in chunks)

    def test_chunk_with_sections_no_headings(self):
        """Test chunk_with_sections handles documents without headings."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)
        chunks = chunk_with_sections(doc, headings=[])

        assert len(chunks) > 0
        # Chunks should exist but have no section metadata

    def test_chunk_with_sections_metadata_populated(self):
        """Test chunks have section metadata when headings provided."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc, headings = extract_with_headings(TEST_PDF)

        if not headings:
            pytest.skip("No headings detected in test PDF")

        chunks = chunk_with_sections(doc, headings)

        # At least some chunks should have section metadata
        chunks_with_sections = [c for c in chunks if c.metadata.get("section")]

        # If headings exist, some chunks should be assigned to sections
        assert len(chunks_with_sections) > 0 or len(headings) == 0

    def test_chunk_with_sections_fields(self):
        """Test section metadata has correct fields."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc, headings = extract_with_headings(TEST_PDF)

        if not headings:
            pytest.skip("No headings detected in test PDF")

        chunks = chunk_with_sections(doc, headings)

        for chunk in chunks:
            # metadata should always exist
            assert "section" in chunk.metadata
            assert "heading_level" in chunk.metadata

            # Values can be None (before first heading) or populated
            if chunk.metadata["section"] is not None:
                assert isinstance(chunk.metadata["section"], str)
                assert isinstance(chunk.metadata["heading_level"], int)
                assert 1 <= chunk.metadata["heading_level"] <= 3


class TestIntegration:
    """Integration tests for heading detection + section tracking pipeline."""

    def test_full_pipeline(self):
        """Test complete pipeline: extract → detect headings → chunk with sections."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        # Step 1: Extract with headings
        doc, headings = extract_with_headings(TEST_PDF)

        # Step 2: Chunk with section tracking
        chunks = chunk_with_sections(doc, headings)

        # Validation
        assert doc.total_pages > 0
        assert len(chunks) > 0

        # All chunks should have metadata field
        for chunk in chunks:
            assert hasattr(chunk, "metadata")
            assert isinstance(chunk.metadata, dict)

        print("\n✅ Pipeline complete:")
        print(f"  Pages: {doc.total_pages}")
        print(f"  Headings detected: {len(headings)}")
        print(f"  Chunks created: {len(chunks)}")

        if headings:
            print("  Sample headings:")
            for h in headings[:3]:
                print(f"    H{h.level}: {h.text}")

            chunks_with_sections = [c for c in chunks if c.metadata.get("section")]
            print(f"  Chunks with section metadata: {len(chunks_with_sections)}/{len(chunks)}")
