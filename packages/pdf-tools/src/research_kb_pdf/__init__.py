"""PDF extraction and chunking for research-kb system.

This package provides:
- PDF extraction via Docling (LaTeX-preserving, Granite-Docling-258M)
- Structure-aware chunking via Docling's HybridChunker
- Embedding generation via BGE-large-en-v1.5
- Citation metadata via GROBID (academic papers)
- Integration with research-kb storage layer
"""

__version__ = "1.0.0"

from research_kb_pdf.docling_extractor import (
    DoclingExtractionResult,
    extract_and_chunk,
)

from research_kb_pdf.chunker import (
    TextChunk,
    count_tokens,
    split_paragraphs,
    split_sentences,
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
    # Extraction (Docling)
    "DoclingExtractionResult",
    "extract_and_chunk",
    # Chunking utilities
    "TextChunk",
    "count_tokens",
    "split_paragraphs",
    "split_sentences",
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
    # Reranking
    "CrossEncoderReranker",
    "RerankResult",
    "RerankClient",
    "rerank_texts",
]
