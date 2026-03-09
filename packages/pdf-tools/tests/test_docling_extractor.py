"""Tests for Docling-based PDF extraction and chunking.

Tests cover:
- extract_and_chunk() returns correct types
- TextChunk fields are populated correctly
- Section metadata from headings
- LaTeX preservation in content
- Page number extraction
- Sequential chunk indices
- Token count within limits
- File not found handling
- Chunking method metadata
- DoclingExtractionResult fields
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_kb_pdf.chunker import TextChunk, count_tokens
from research_kb_pdf.docling_extractor import (
    DoclingExtractionResult,
    _get_page_from_prov,
    extract_and_chunk,
)

pytestmark = pytest.mark.unit


def _make_mock_chunk(text, headings=None, page=None):
    """Create a mock DocChunk with the expected interface."""
    chunk = MagicMock()
    chunk.text = text

    meta = MagicMock()
    meta.headings = headings or []
    meta.doc_items = []

    if page is not None:
        # Create provenance with page number
        prov = MagicMock()
        prov.page_no = page
        item = MagicMock()
        item.prov = [prov]
        meta.doc_items = [item]

    chunk.meta = meta
    return chunk


def _make_mock_document(markdown_text="# Title\n\nSome content", pages=None):
    """Create a mock DoclingDocument."""
    doc = MagicMock()
    doc.export_to_markdown.return_value = markdown_text

    if pages is None:
        pages = {1: MagicMock(), 2: MagicMock()}
    doc.pages = pages

    # Mock body for heading counting
    body = MagicMock()
    body.children = []
    doc.body = body

    return doc


class TestExtractAndChunk:
    """Test the main extract_and_chunk function."""

    @patch("docling.document_converter.DocumentConverter")
    @patch("docling.chunking.HybridChunker")
    @patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
    @patch("transformers.AutoTokenizer")
    def test_returns_textchunks(
        self, mock_auto_tok, mock_hf_tok, mock_chunker_cls, mock_converter_cls, tmp_path
    ):
        """extract_and_chunk returns (DoclingExtractionResult, list[TextChunk])."""
        # Create test PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        # Mock converter
        mock_doc = _make_mock_document()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter_cls.return_value.convert.return_value = mock_result

        # Mock chunker
        mock_chunks = [
            _make_mock_chunk("This is chunk one content.", headings=["Introduction"], page=1),
            _make_mock_chunk("This is chunk two content.", headings=["Methods"], page=2),
        ]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = mock_chunks
        mock_chunker.contextualize.side_effect = lambda c: c.text
        mock_chunker_cls.return_value = mock_chunker

        result, chunks = extract_and_chunk(pdf_file)

        assert isinstance(result, DoclingExtractionResult)
        assert isinstance(chunks, list)
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert len(chunks) == 2

    @patch("docling.document_converter.DocumentConverter")
    @patch("docling.chunking.HybridChunker")
    @patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
    @patch("transformers.AutoTokenizer")
    def test_textchunk_fields_populated(
        self, mock_auto_tok, mock_hf_tok, mock_chunker_cls, mock_converter_cls, tmp_path
    ):
        """All TextChunk fields are correctly populated."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_doc = _make_mock_document()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter_cls.return_value.convert.return_value = mock_result

        content_text = "Statistical inference for causal effects."
        mock_chunks = [_make_mock_chunk(content_text, headings=["Causal Inference"], page=3)]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = mock_chunks
        mock_chunker.contextualize.side_effect = lambda c: c.text
        mock_chunker_cls.return_value = mock_chunker

        _, chunks = extract_and_chunk(pdf_file)

        chunk = chunks[0]
        assert chunk.content == content_text
        assert chunk.start_page == 3
        assert chunk.end_page == 3
        assert chunk.token_count == count_tokens(content_text)
        assert chunk.char_count == len(content_text)
        assert chunk.chunk_index == 0

    @patch("docling.document_converter.DocumentConverter")
    @patch("docling.chunking.HybridChunker")
    @patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
    @patch("transformers.AutoTokenizer")
    def test_section_metadata_from_headings(
        self, mock_auto_tok, mock_hf_tok, mock_chunker_cls, mock_converter_cls, tmp_path
    ):
        """Section metadata is extracted from chunk headings."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_doc = _make_mock_document()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter_cls.return_value.convert.return_value = mock_result

        mock_chunks = [
            _make_mock_chunk("Content", headings=["Chapter 1", "Section 1.1"], page=1),
            _make_mock_chunk("More content", headings=[], page=2),
        ]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = mock_chunks
        mock_chunker.contextualize.side_effect = lambda c: c.text
        mock_chunker_cls.return_value = mock_chunker

        _, chunks = extract_and_chunk(pdf_file)

        # First chunk: last heading is "Section 1.1", level 2
        assert chunks[0].metadata["section"] == "Section 1.1"
        assert chunks[0].metadata["heading_level"] == 2

        # Second chunk: no headings
        assert chunks[1].metadata["section"] is None
        assert chunks[1].metadata["heading_level"] == 0

    @patch("docling.document_converter.DocumentConverter")
    @patch("docling.chunking.HybridChunker")
    @patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
    @patch("transformers.AutoTokenizer")
    def test_latex_preserved_in_content(
        self, mock_auto_tok, mock_hf_tok, mock_chunker_cls, mock_converter_cls, tmp_path
    ):
        """LaTeX equations are preserved in chunk content."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        latex_content = "The regression model is $$E[Y|X] = X\\beta + \\epsilon$$"
        mock_doc = _make_mock_document(
            markdown_text=f"# Model\n\n{latex_content}"
        )
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter_cls.return_value.convert.return_value = mock_result

        mock_chunks = [_make_mock_chunk(latex_content, page=1)]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = mock_chunks
        mock_chunker.contextualize.side_effect = lambda c: c.text
        mock_chunker_cls.return_value = mock_chunker

        result, chunks = extract_and_chunk(pdf_file)

        assert "$$" in chunks[0].content
        assert "\\beta" in chunks[0].content
        assert result.has_equations is True

    @patch("docling.document_converter.DocumentConverter")
    @patch("docling.chunking.HybridChunker")
    @patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
    @patch("transformers.AutoTokenizer")
    def test_chunk_index_sequential(
        self, mock_auto_tok, mock_hf_tok, mock_chunker_cls, mock_converter_cls, tmp_path
    ):
        """Chunk indices are sequential starting from 0."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_doc = _make_mock_document()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter_cls.return_value.convert.return_value = mock_result

        mock_chunks = [
            _make_mock_chunk(f"Chunk {i} content", page=i + 1) for i in range(5)
        ]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = mock_chunks
        mock_chunker.contextualize.side_effect = lambda c: c.text
        mock_chunker_cls.return_value = mock_chunker

        _, chunks = extract_and_chunk(pdf_file)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    @patch("docling.document_converter.DocumentConverter")
    @patch("docling.chunking.HybridChunker")
    @patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
    @patch("transformers.AutoTokenizer")
    def test_chunking_method_is_docling(
        self, mock_auto_tok, mock_hf_tok, mock_chunker_cls, mock_converter_cls, tmp_path
    ):
        """All chunks have chunking_method='docling' in metadata."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_doc = _make_mock_document()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter_cls.return_value.convert.return_value = mock_result

        mock_chunks = [_make_mock_chunk("Content", page=1)]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = mock_chunks
        mock_chunker.contextualize.side_effect = lambda c: c.text
        mock_chunker_cls.return_value = mock_chunker

        _, chunks = extract_and_chunk(pdf_file)

        assert chunks[0].metadata["chunking_method"] == "docling"

    def test_file_not_found_raises(self):
        """extract_and_chunk raises FileNotFoundError for missing PDF."""
        with pytest.raises(FileNotFoundError):
            extract_and_chunk("/nonexistent/path/missing.pdf")


class TestDoclingExtractionResult:
    """Test DoclingExtractionResult dataclass."""

    def test_fields(self):
        """All fields are accessible."""
        result = DoclingExtractionResult(
            file_path="/data/test.pdf",
            total_pages=42,
            total_chars=100000,
            has_equations=True,
            heading_count=15,
        )

        assert result.file_path == "/data/test.pdf"
        assert result.total_pages == 42
        assert result.total_chars == 100000
        assert result.has_equations is True
        assert result.heading_count == 15

    def test_has_equations_false(self):
        """has_equations is False when no LaTeX markers found."""
        result = DoclingExtractionResult(
            file_path="/data/test.pdf",
            total_pages=10,
            total_chars=5000,
            has_equations=False,
        )
        assert result.has_equations is False
        assert result.heading_count == 0  # default

    @patch("docling.document_converter.DocumentConverter")
    @patch("docling.chunking.HybridChunker")
    @patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
    @patch("transformers.AutoTokenizer")
    def test_extraction_result_pages_from_document(
        self, mock_auto_tok, mock_hf_tok, mock_chunker_cls, mock_converter_cls, tmp_path
    ):
        """total_pages comes from document.pages."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_doc = _make_mock_document(pages={1: MagicMock(), 2: MagicMock(), 3: MagicMock()})
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter_cls.return_value.convert.return_value = mock_result

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []
        mock_chunker_cls.return_value = mock_chunker

        result, _ = extract_and_chunk(pdf_file)

        assert result.total_pages == 3


class TestGetPageFromProv:
    """Test page number extraction from chunk provenance."""

    def test_page_from_doc_items_prov(self):
        """Extracts page from doc_items provenance."""
        chunk = _make_mock_chunk("text", page=5)
        start, end = _get_page_from_prov(chunk)
        assert start == 5
        assert end == 5

    def test_page_from_multiple_provs(self):
        """Extracts min/max page from multiple provenance items."""
        chunk = MagicMock()
        meta = MagicMock()

        prov1 = MagicMock()
        prov1.page_no = 3
        prov2 = MagicMock()
        prov2.page_no = 7

        item1 = MagicMock()
        item1.prov = [prov1]
        item2 = MagicMock()
        item2.prov = [prov2]

        meta.doc_items = [item1, item2]
        chunk.meta = meta

        start, end = _get_page_from_prov(chunk)
        assert start == 3
        assert end == 7

    def test_default_page_when_no_prov(self):
        """Returns (1, 1) when no provenance available."""
        chunk = MagicMock()
        chunk.meta = MagicMock()
        chunk.meta.doc_items = []
        chunk.meta.page = None

        start, end = _get_page_from_prov(chunk)
        assert start == 1
        assert end == 1

    def test_fallback_to_meta_page(self):
        """Falls back to chunk.meta.page when doc_items unavailable."""
        chunk = MagicMock()
        meta = MagicMock()
        meta.doc_items = []
        meta.page = 4
        chunk.meta = meta

        start, end = _get_page_from_prov(chunk)
        assert start == 4
        assert end == 4


class TestContractCompatibility:
    """Verify TextChunk format matches storage layer expectations."""

    @patch("docling.document_converter.DocumentConverter")
    @patch("docling.chunking.HybridChunker")
    @patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
    @patch("transformers.AutoTokenizer")
    def test_chunk_has_required_storage_fields(
        self, mock_auto_tok, mock_hf_tok, mock_chunker_cls, mock_converter_cls, tmp_path
    ):
        """TextChunk has all fields required by ChunkStore.batch_create."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_doc = _make_mock_document()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter_cls.return_value.convert.return_value = mock_result

        mock_chunks = [_make_mock_chunk("Test content for storage.", page=1)]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = mock_chunks
        mock_chunker.contextualize.side_effect = lambda c: c.text
        mock_chunker_cls.return_value = mock_chunker

        _, chunks = extract_and_chunk(pdf_file)

        chunk = chunks[0]

        # These fields are used by dispatcher._store_chunks_with_embeddings
        assert hasattr(chunk, "content")
        assert hasattr(chunk, "start_page")
        assert hasattr(chunk, "end_page")
        assert hasattr(chunk, "metadata")
        assert isinstance(chunk.content, str)
        assert isinstance(chunk.start_page, int)
        assert isinstance(chunk.end_page, int)
        assert isinstance(chunk.metadata, dict)
        assert chunk.start_page >= 1
        assert chunk.end_page >= chunk.start_page

    @patch("docling.document_converter.DocumentConverter")
    @patch("docling.chunking.HybridChunker")
    @patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
    @patch("transformers.AutoTokenizer")
    def test_metadata_has_chunking_method(
        self, mock_auto_tok, mock_hf_tok, mock_chunker_cls, mock_converter_cls, tmp_path
    ):
        """Chunk metadata includes chunking_method for rechunk filtering."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_doc = _make_mock_document()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter_cls.return_value.convert.return_value = mock_result

        mock_chunks = [_make_mock_chunk("Content", page=1)]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = mock_chunks
        mock_chunker.contextualize.side_effect = lambda c: c.text
        mock_chunker_cls.return_value = mock_chunker

        _, chunks = extract_and_chunk(pdf_file)

        # rechunk_corpus.py filters on this field
        assert chunks[0].metadata["chunking_method"] == "docling"
