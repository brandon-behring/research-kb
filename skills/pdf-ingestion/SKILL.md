# PDF Ingestion Skill

This skill teaches agents how to ingest PDF documents into the research knowledge base.

## When to Use

- Adding new textbooks or papers to the knowledge base
- Updating existing documents with new versions
- Bulk ingestion of document collections

## Ingestion Pipeline Overview

```
PDF → Docling (Granite-258M) → HybridChunker → Embedding → Storage
      ↓ (papers only)
      GROBID → Citation metadata
```

### 1. Extraction Methods

**Docling (All PDFs)**
- Best for: All document types — textbooks, papers, technical manuals
- Uses: IBM Granite-Docling-258M for document understanding
- Preserves: Equations as LaTeX (`$$...$$` or `\(...\)`), headings, tables
- Output: `DoclingExtractionResult` + `list[TextChunk]`

**GROBID (Papers — metadata only)**
- Used for: Citation extraction from academic papers
- Extracts: Authors, title, abstract, citations, DOI/arXiv IDs
- Output: `ExtractedPaper` with bibliography and BibTeX

### 2. Structure-Aware Chunking

Docling's HybridChunker provides structure-aware chunking:
- Respects heading boundaries (no mid-section splits)
- Contextualizes chunks with parent heading hierarchy
- Uses BGE-large-en-v1.5 tokenizer for consistent token counts
- Target: 300 tokens per chunk (configurable via `max_tokens`)

### 3. Embedding Generation

- **Model**: BGE-large-en-v1.5 (1024 dimensions)
- **Server**: Unix socket for low-latency inference
- **Start server**: `python -m research_kb_pdf.embed_server`

### 4. Citation Extraction & Storage

GROBID extracts citations from `<listBibl>` in TEI-XML:
- Authors, title, year, venue
- DOI and arXiv IDs when available
- Raw string for fallback
- BibTeX entry generation

Citations are automatically stored via `CitationStore.batch_create()` during ingestion.

```python
from research_kb_storage import CitationStore

# Query stored citations
citations = await CitationStore.list_by_source(source.id)
citation = await CitationStore.find_by_doi("10.1017/CBO9780511803161")
citation = await CitationStore.find_by_arxiv("1706.03762")
count = await CitationStore.count_by_source(source.id)
```

## Usage Examples

### Extract and Chunk a PDF

```python
from research_kb_pdf import extract_and_chunk, EmbeddingClient

# Extract + chunk in one call (Docling + HybridChunker)
result, chunks = extract_and_chunk("textbook.pdf", max_tokens=300)

print(f"{result.total_pages} pages, {len(chunks)} chunks")
print(f"Has equations: {result.has_equations}")

# Embed and store
client = EmbeddingClient()
for chunk in chunks:
    embedding = client.embed(chunk.content)
    await ChunkStore.create(source_id=source.id, content=chunk.content, embedding=embedding, ...)
```

### Use the Dispatcher (Recommended)

```python
from research_kb_pdf import PDFDispatcher, IngestResult
from research_kb_contracts import SourceType

dispatcher = PDFDispatcher()
result: IngestResult = await dispatcher.ingest_pdf(
    pdf_path="paper.pdf",
    source_type=SourceType.PAPER,
    title="Attention Is All You Need",
    domain_id="deep_learning",
    authors=["Vaswani", "Shazeer"],
    year=2017,
    metadata={"arxiv_id": "1706.03762"},
)

# Result contains:
# - result.source: Created Source record
# - result.chunk_count: Number of chunks created
# - result.citations_extracted: Number of citations stored
# - result.headings_detected: Detected heading count
# - result.extraction_method: "grobid+docling" or "docling"
```

### Generate BibTeX

```python
from research_kb_pdf import source_to_bibtex, citation_to_bibtex, generate_bibliography
from research_kb_storage import SourceStore, CitationStore
from research_kb_contracts import SourceType

# Get sources and citations
sources = await SourceStore.list_by_type(SourceType.PAPER, limit=100)
citations = await CitationStore.list_by_source(source_id)

# Generate bibliography
bibtex = generate_bibliography(sources, citations)

# Or generate single entry
entry = source_to_bibtex(source)
```

## Dead Letter Queue (DLQ)

Failed ingestions go to the DLQ:
- Path: `.dlq/` in project root
- Contains: Original PDF + error details
- Retry: Manual review and re-ingestion

## Metadata Best Practices

**Source metadata** (stored in `sources.metadata`):
```python
{
    "publisher": "Cambridge University Press",
    "edition": "2nd",
    "domain": "causal inference",
    "authority": "canonical",  # canonical | survey | frontier
    "arxiv_id": "2402.13023",
}
```

**Chunk metadata** (stored in `chunks.metadata`):
```python
{
    "section": "3.3 The Backdoor Criterion",
    "heading_level": 2,
    "chunking_method": "docling",
}
```

## Performance Notes

- Textbook (500 pages): ~5-15 minutes (Docling model inference)
- Paper (20 pages): ~30-60 seconds
- Embedding: ~50ms per chunk on GPU
- First PDF loads Granite-Docling-258M model (~5s, ~2.5 GB VRAM)

## GPU VRAM Management

RTX 2070 (7.6 GB) runs both Docling (~2.5-3.5 GB) and embed_server (~2.5 GB).
During ingestion, these run **sequentially per source** (not simultaneously).

If VRAM is tight:
- Kill daemon before ingesting: `systemctl --user stop research-kb-daemon`
- Force Docling to CPU: `CUDA_VISIBLE_DEVICES="" python scripts/ingest_missing_textbooks.py`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Embedding server not running | `python -m research_kb_pdf.embed_server` |
| GROBID not available | `docker-compose up grobid` |
| PostgreSQL connection failed | `docker start research-kb-postgres` |
| Duplicate source | Check `file_hash` — same PDF already ingested |
| Docling GPU OOM | Set `CUDA_VISIBLE_DEVICES=""` to force CPU mode |
| Slow extraction | Expected: Docling is ~10-30s/PDF (GPU), ~30-60s (CPU) |
