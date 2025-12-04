# research-kb-pdf

PDF extraction and chunking for the research-kb system.

## Features

- **PyMuPDF extraction**: Fast textbook extraction with page number tracking
- **GROBID integration**: Academic paper parsing (IMRAD structure) [Coming in Day 5-6]
- **Smart chunking**: 300±50 tokens with 50-token overlap, sentence-aware
- **BGE embeddings**: 1024-dim vectors via BGE-large-en-v1.5 (Unix socket server)
- **Error handling**: DLQ for failed extractions [Coming in Day 8]

## Installation

```bash
cd packages/pdf-tools
poetry install
```

## Usage

```python
from research_kb_pdf import (
    extract_pdf,
    chunk_document,
    EmbeddingClient,
)

# 1. Extract PDF
document = extract_pdf("paper.pdf")
print(f"Extracted {document.total_pages} pages")

# 2. Chunk document
chunks = chunk_document(document, target_tokens=300, overlap_tokens=50)
print(f"Created {len(chunks)} chunks")

# 3. Embed chunks (requires running server)
# Start server: python -m research_kb_pdf.embed_server &
client = EmbeddingClient()
embeddings = client.embed_chunks(chunks)
print(f"Generated {len(embeddings)} embeddings")
```

## Testing

```bash
poetry run pytest tests/
```

## Week 2 Implementation Plan

- [x] Day 1: Package setup + PyMuPDF extraction foundation
- [x] Day 2-3: Chunking logic with BGE tokenizer
- [x] Day 4: Embedding service (BGE-large-en-v1.5) + client
- [ ] Day 5-6: GROBID paper extraction
- [ ] Day 7: PyMuPDF hierarchy detection
- [ ] Day 8: Dispatcher + DLQ
- [ ] Day 9-10: End-to-end integration + golden tests
