# PDF Ingestion Skill

This skill teaches agents how to ingest PDF documents into the research knowledge base.

## When to Use

- Adding new textbooks or papers to the knowledge base
- Updating existing documents with new versions
- Bulk ingestion of document collections

## Ingestion Pipeline Overview

```
PDF → Extraction → Chunking → Embedding → Storage
```

### 1. Extraction Methods

**PyMuPDF (Textbooks)**
- Best for: Textbooks with complex formatting, figures, tables
- Preserves: Page numbers, font sizes for heading detection
- Output: `ExtractedDocument` with pages and detected headings

**GROBID (Papers)**
- Best for: Academic papers with structured sections
- Extracts: Title, authors, abstract, sections (IMRAD format), citations
- Output: `ExtractedPaper` with metadata, sections, and bibliography

### 2. Heading Detection

Font-size based heuristics detect headings:
- H1: font_size > median + 2σ
- H2: font_size > median + 1.5σ
- H3: font_size > median + σ

### 3. Chunking Strategy

- **Target size**: 300 tokens (±50)
- **Overlap**: Sentence boundary aware
- **Section tracking**: Each chunk knows its parent heading hierarchy
- **Metadata preserved**: page numbers, section name, heading level

### 4. Embedding Generation

- **Model**: BGE-large-en-v1.5 (1024 dimensions)
- **Server**: Unix socket for low-latency inference
- **Start server**: `python -m research_kb_pdf.embed_server`

### 5. Citation Extraction & Storage

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

### Ingest a Single PDF

```python
from research_kb_pdf import extract_with_headings, chunk_with_sections, EmbeddingClient
from research_kb_storage import SourceStore, ChunkStore, get_connection_pool

# Extract
doc, headings = extract_with_headings("textbook.pdf")

# Chunk
chunks = chunk_with_sections(doc, headings)

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
    authors=["Vaswani", "Shazeer"],
    year=2017,
    metadata={"arxiv_id": "1706.03762"},
)

# Result contains:
# - result.source: Created Source record
# - result.chunk_count: Number of chunks created
# - result.citations_extracted: Number of citations stored
# - result.headings_detected: Detected heading count
# - result.extraction_method: "grobid+pymupdf" or "pymupdf"
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
    "chunk_type": "theorem",  # theorem | definition | example | proof
}
```

## Performance Notes

- Textbook (500 pages): ~5 minutes
- Paper (20 pages): ~30 seconds
- Embedding: ~50ms per chunk on GPU

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Embedding server not running | `python -m research_kb_pdf.embed_server` |
| GROBID not available | `docker-compose up grobid` |
| PostgreSQL connection failed | `docker start research-kb-postgres` |
| Duplicate source | Check `file_hash` - same PDF already ingested |
