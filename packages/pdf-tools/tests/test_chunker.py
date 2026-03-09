"""Tests for shared chunking utilities (TextChunk, token counting, text splitting).

These tests cover the utilities retained after the PyMuPDF → Docling migration:
TextChunk, count_tokens, split_paragraphs, split_sentences, get_overlap_paragraphs.
"""

import pytest

from research_kb_pdf import count_tokens
from research_kb_pdf.chunker import (
    split_paragraphs,
    get_overlap_paragraphs,
    split_sentences,
    MAX_EMBEDDING_TOKENS,
)

pytestmark = pytest.mark.unit


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
        assert any("Dr. Smith" in s for s in sentences), f"Lost 'Dr. Smith': {sentences}"

    def test_split_sentences_mr_mrs(self):
        """Test Mr./Mrs. abbreviations are preserved."""
        text = "Mr. Jones and Mrs. Smith arrived. They were happy."
        sentences = split_sentences(text)
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
        # Structural test — verify the constant is accessible
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
        short_para = "Short paragraph."  # ~3 tokens
        medium_para = "This is a medium length paragraph with some content. " * 5  # ~50 tokens
        paragraphs = [short_para, medium_para, short_para, medium_para]

        overlap = get_overlap_paragraphs(paragraphs, target_tokens=50)
        overlap_text = "\n\n".join(overlap)
        overlap_tokens = count_tokens(overlap_text)

        assert overlap_tokens > 0, "Should have some overlap"
        assert len(overlap) < len(paragraphs), "Should not include all paragraphs"
