"""PDF extraction and chunking for research-kb system.

This package provides:
- PDF text extraction (PyMuPDF for textbooks, GROBID for papers)
- Token-based chunking with overlap
- Embedding generation via BGE-large-en-v1.5
- Integration with research-kb storage layer
"""

__version__ = "1.0.0"

from research_kb_pdf.pymupdf_extractor import (
    ExtractedDocument,
    ExtractedPage,
    Heading,
    extract_pdf,
    get_text_with_page_numbers,
    get_full_text,
    detect_headings,
    extract_with_headings,
)

from research_kb_pdf.chunker import (
    TextChunk,
    chunk_document,
    chunk_with_sections,
    count_tokens,
)

from research_kb_pdf.embedding_client import (
    EmbeddingClient,
    embed_text,
    embed_texts,
)

from research_kb_pdf.grobid_client import (
    GrobidClient,
    ExtractedPaper,
    PaperMetadata,
    PaperSection,
    parse_tei_xml,
)

from research_kb_pdf.dlq import (
    DLQEntry,
    DeadLetterQueue,
)

from research_kb_pdf.dispatcher import (
    PDFDispatcher,
    IngestResult,
)

from research_kb_pdf.bibtex_generator import (
    citation_to_bibtex,
    source_to_bibtex,
    generate_bibliography,
    generate_bibtex_key,
    escape_bibtex,
)

from research_kb_pdf.reranker import (
    CrossEncoderReranker,
    RerankResult,
)

from research_kb_pdf.rerank_client import (
    RerankClient,
    rerank_texts,
)

__all__ = [
    # Extraction
    "ExtractedDocument",
    "ExtractedPage",
    "Heading",
    "extract_pdf",
    "get_text_with_page_numbers",
    "get_full_text",
    "detect_headings",
    "extract_with_headings",
    # Chunking
    "TextChunk",
    "chunk_document",
    "chunk_with_sections",
    "count_tokens",
    # Embedding
    "EmbeddingClient",
    "embed_text",
    "embed_texts",
    # GROBID
    "GrobidClient",
    "ExtractedPaper",
    "PaperMetadata",
    "PaperSection",
    "parse_tei_xml",
    # DLQ & Dispatcher
    "DLQEntry",
    "DeadLetterQueue",
    "PDFDispatcher",
    "IngestResult",
    # BibTeX
    "citation_to_bibtex",
    "source_to_bibtex",
    "generate_bibliography",
    "generate_bibtex_key",
    "escape_bibtex",
    # Reranking (Phase 3)
    "CrossEncoderReranker",
    "RerankResult",
    "RerankClient",
    "rerank_texts",
]
