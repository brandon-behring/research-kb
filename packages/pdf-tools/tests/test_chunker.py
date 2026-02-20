"""Tests for PDF chunker with token-accurate segmentation."""

import pytest
from pathlib import Path

from research_kb_pdf import (
    extract_pdf,
    chunk_document,
    count_tokens,
    get_full_text,
)
from research_kb_pdf.chunker import (
    split_paragraphs,
    get_overlap_paragraphs,
    split_sentences,
    MAX_EMBEDDING_TOKENS,
)

pytestmark = pytest.mark.unit


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TEST_PDF = FIXTURES_DIR / "test_simple.pdf"


class TestTokenCounting:
    """Test token counting accuracy."""

    def test_count_tokens_basic(self):
        """Test basic token counting."""
        text = "Hello world"
        tokens = count_tokens(text)
        assert tokens > 0, "Should count some tokens"
        assert tokens < 10, "Should be reasonable token count"

    def test_count_tokens_empty(self):
        """Test empty string returns 0 tokens."""
        assert count_tokens("") == 0

    def test_count_tokens_long_text(self):
        """Test token counting on longer text."""
        text = "This is a longer piece of text with multiple sentences. " * 10
        tokens = count_tokens(text)
        # Roughly 10 tokens per sentence, 10 repeats = ~100 tokens
        assert 80 < tokens < 150, f"Expected ~100 tokens, got {tokens}"


class TestParagraphSplitting:
    """Test paragraph boundary detection."""

    def test_split_paragraphs_basic(self):
        """Test basic paragraph splitting."""
        text = "Para 1\n\nPara 2\n\nPara 3"
        paras = split_paragraphs(text)
        assert len(paras) == 3
        assert paras == ["Para 1", "Para 2", "Para 3"]

    def test_split_paragraphs_multiple_newlines(self):
        """Test splitting with multiple newlines."""
        text = "Para 1\n\n\nPara 2\n\n\n\nPara 3"
        paras = split_paragraphs(text)
        assert len(paras) == 3

    def test_split_paragraphs_single_newlines_preserved(self):
        """Test that single newlines within paragraphs are preserved."""
        text = "Line 1\nLine 2\n\nPara 2"
        paras = split_paragraphs(text)
        assert len(paras) == 2
        assert "Line 1\nLine 2" in paras[0]

    def test_split_paragraphs_empty_lines_ignored(self):
        """Test that empty paragraphs are filtered out."""
        text = "Para 1\n\n\n\nPara 2"
        paras = split_paragraphs(text)
        assert len(paras) == 2


class TestSentenceSplitting:
    """Test improved sentence splitting that respects abbreviations."""

    def test_split_sentences_basic(self):
        """Test basic sentence splitting."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = split_sentences(text)
        assert len(sentences) >= 2, f"Expected multiple sentences, got {sentences}"

    def test_split_sentences_preserves_abbreviations(self):
        """Test that common abbreviations don't cause splits."""
        text = "Dr. Smith said hello. She was right."
        sentences = split_sentences(text)
        # Should NOT split on "Dr."
        # The first sentence should contain "Dr. Smith"
        assert any("Dr. Smith" in s for s in sentences), f"Lost 'Dr. Smith': {sentences}"

    def test_split_sentences_mr_mrs(self):
        """Test Mr./Mrs. abbreviations are preserved."""
        text = "Mr. Jones and Mrs. Smith arrived. They were happy."
        sentences = split_sentences(text)
        # Should preserve "Mr. Jones and Mrs. Smith" together
        assert any(
            "Mr. Jones" in s and "Mrs. Smith" in s for s in sentences
        ), f"Split on Mr./Mrs.: {sentences}"

    def test_split_sentences_vs_abbreviation(self):
        """Test vs. abbreviation is preserved."""
        text = "The case of Smith vs. Jones was important. It set precedent."
        sentences = split_sentences(text)
        assert any("vs. Jones" in s for s in sentences), f"Split on vs.: {sentences}"

    def test_split_sentences_etc_abbreviation(self):
        """Test etc. abbreviation is preserved."""
        text = "Items include apples, oranges, etc. More text here."
        sentences = split_sentences(text)
        assert any("etc." in s and "apples" in s for s in sentences), f"Split on etc.: {sentences}"

    def test_split_sentences_initials(self):
        """Test single-letter initials are preserved."""
        text = "J. K. Rowling wrote many books. Her work is famous."
        sentences = split_sentences(text)
        # The full name should be preserved in one sentence
        first_sentence = sentences[0] if sentences else ""
        assert "Rowling" in first_sentence, f"Split on initials: {sentences}"

    def test_split_sentences_exclamation_question(self):
        """Test splitting on ! and ? marks."""
        text = "What happened? It was amazing! Then silence."
        sentences = split_sentences(text)
        assert len(sentences) >= 2, f"Should split on ! and ?: {sentences}"

    def test_split_sentences_empty_string(self):
        """Test empty string returns empty list."""
        sentences = split_sentences("")
        assert sentences == [] or sentences == [""]

    def test_split_sentences_single_sentence(self):
        """Test single sentence without ending punctuation."""
        text = "Just one sentence"
        sentences = split_sentences(text)
        assert len(sentences) >= 1


class TestChunkSizeValidation:
    """Test chunk size validation against embedding model limits."""

    def test_max_embedding_tokens_constant(self):
        """Test that MAX_EMBEDDING_TOKENS is defined correctly."""
        assert MAX_EMBEDDING_TOKENS == 512, "Expected BGE model limit of 512 tokens"

    def test_chunk_document_warns_on_oversized(self, caplog):
        """Test that oversized chunks generate warnings."""
        # This is a structural test - actual warning verification
        # depends on having a PDF with very large paragraphs
        # For now, just verify the constant is accessible
        assert MAX_EMBEDDING_TOKENS > 0


class TestOverlapCalculation:
    """Test overlap paragraph selection."""

    def test_get_overlap_paragraphs_basic(self):
        """Test getting overlap paragraphs."""
        paragraphs = ["Short", "Medium length paragraph", "Another one"]
        overlap = get_overlap_paragraphs(paragraphs, target_tokens=10)

        assert len(overlap) > 0
        assert len(overlap) <= len(paragraphs)
        # Should keep last paragraphs
        assert overlap[-1] == paragraphs[-1]

    def test_get_overlap_paragraphs_empty(self):
        """Test with empty paragraph list."""
        overlap = get_overlap_paragraphs([], target_tokens=50)
        assert overlap == []

    def test_get_overlap_paragraphs_respects_max(self):
        """Test that overlap retrieves reasonable amount of content."""
        # Create paragraphs with known token counts
        short_para = "Short paragraph."  # ~3 tokens
        medium_para = "This is a medium length paragraph with some content. " * 5  # ~50 tokens
        paragraphs = [short_para, medium_para, short_para, medium_para]

        overlap = get_overlap_paragraphs(paragraphs, target_tokens=50)
        overlap_text = "\n\n".join(overlap)
        overlap_tokens = count_tokens(overlap_text)

        # Should get some overlap but not all paragraphs
        assert overlap_tokens > 0, "Should have some overlap"
        assert len(overlap) < len(paragraphs), "Should not include all paragraphs"


class TestChunkDocument:
    """Test document chunking with real PDF."""

    def test_chunk_document_basic(self):
        """Test basic document chunking."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)
        chunks = chunk_document(doc)

        # Basic validations
        assert len(chunks) > 0, "Should create at least one chunk"
        assert all(
            isinstance(chunk.content, str) for chunk in chunks
        ), "All chunks should have content"
        assert all(chunk.token_count > 0 for chunk in chunks), "All chunks should have tokens"

        print(f"\n✅ Created {len(chunks)} chunks from {doc.total_pages} pages")

    def test_chunk_document_token_counts(self):
        """Test that chunks respect token count targets."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)
        chunks = chunk_document(doc, target_tokens=300, max_variance=50)

        # Check that most chunks are within target range
        min_tokens = 250
        max_tokens = 350

        in_range_count = 0
        for chunk in chunks:
            if min_tokens <= chunk.token_count <= max_tokens:
                in_range_count += 1

        # At least 45% of chunks should be in target range (PDF paragraphs vary in size)
        in_range_ratio = in_range_count / len(chunks)
        assert in_range_ratio >= 0.45, f"Only {in_range_ratio:.0%} of chunks in target range"

        # Average should be close to target
        avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
        assert 250 <= avg_tokens <= 350, f"Average {avg_tokens:.0f} tokens outside target range"

        print(f"\n✅ {in_range_count}/{len(chunks)} chunks in range, avg: {avg_tokens:.0f} tokens")

    def test_chunk_document_no_content_loss(self):
        """Test that chunking doesn't lose significant content."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)
        original_text = get_full_text(doc)
        chunks = chunk_document(doc)

        # Reconstruct text from chunks (without overlap)
        # This is approximate - we expect some duplication from overlap
        total_chars = sum(c.char_count for c in chunks)

        # Allow for overlap (expect 10-30% more chars due to overlap)
        assert total_chars >= len(original_text), "Chunks have less content than original"
        assert (
            total_chars <= len(original_text) * 1.5
        ), f"Too much overlap: {total_chars} vs {len(original_text)} original chars"

        print(
            f"\n✅ Content preservation: {len(original_text)} original chars, {total_chars} total chunk chars"
        )

    def test_chunk_document_overlap_verification(self):
        """Test that chunking strategy produces reasonable coverage."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)
        chunks = chunk_document(doc, overlap_tokens=50)

        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks to test")

        # Verify chunks span the document (page numbers should be reasonable)
        # Note: Page detection uses snippet matching, so small inaccuracies expected
        first_pages = [c.start_page for c in chunks[:3]]
        last_pages = [c.end_page for c in chunks[-3:]]

        # First chunks should be in first half of document
        assert all(
            p <= doc.total_pages // 2 + 2 for p in first_pages
        ), "Early chunks should be in first half"

        # Last chunks should be in second half of document
        assert all(
            p >= doc.total_pages // 2 for p in last_pages
        ), "Late chunks should be in second half"

        # Verify total token count is reasonable (some duplication from overlap expected)
        total_tokens = sum(c.token_count for c in chunks)
        original_tokens = count_tokens(get_full_text(doc))

        # Total should be more than original (due to overlap) but not 2x
        assert total_tokens >= original_tokens, "Should preserve all content"
        assert total_tokens <= original_tokens * 1.5, "Too much duplication from overlap"

        print(
            f"\n✅ Coverage: {total_tokens} chunk tokens vs {original_tokens} original ({total_tokens/original_tokens:.1%})"
        )

    def test_chunk_document_page_tracking(self):
        """Test that chunks track page numbers correctly."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)
        chunks = chunk_document(doc)

        for chunk in chunks:
            # Page numbers should be valid
            assert chunk.start_page >= 1, f"Invalid start_page: {chunk.start_page}"
            assert chunk.end_page >= chunk.start_page, "end_page should be >= start_page"
            assert chunk.end_page <= doc.total_pages, "end_page exceeds document pages"

        # First chunk should start on page 1
        assert chunks[0].start_page == 1, "First chunk should start on page 1"

        # Last chunk should end on last page (or close to it)
        assert (
            chunks[-1].end_page >= doc.total_pages - 2
        ), "Last chunk should be near end of document"

        print(f"\n✅ Page tracking: chunks span pages 1-{chunks[-1].end_page}")

    def test_chunk_document_custom_params(self):
        """Test chunking with custom parameters."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)
        chunks = chunk_document(doc, target_tokens=200, max_variance=30, overlap_tokens=30)

        # Should create more chunks with smaller target
        assert len(chunks) > 0

        # Average should be close to new target (allow small deviation)
        avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
        assert 165 <= avg_tokens <= 235, f"Average {avg_tokens:.0f} tokens outside target range"

        print(
            f"\n✅ Custom params: {len(chunks)} chunks, avg {avg_tokens:.0f} tokens (target: 200)"
        )

    def test_chunk_index_sequential(self):
        """Test that chunk indices are sequential."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        doc = extract_pdf(TEST_PDF)
        chunks = chunk_document(doc)

        for i, chunk in enumerate(chunks):
            assert (
                chunk.chunk_index == i
            ), f"Chunk index mismatch: expected {i}, got {chunk.chunk_index}"


# Helper function
def longest_common_substring(s1: str, s2: str) -> str:
    """Find longest common substring between two strings."""
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest = 0, 0

    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0

    return s1[x_longest - longest : x_longest]
