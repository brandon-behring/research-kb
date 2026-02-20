# research-kb-pdf

PDF extraction, chunking, and embedding for the research-kb system.

## Features

- **PyMuPDF extraction**: Fast textbook extraction with page number tracking
- **GROBID integration**: Academic paper parsing with IMRAD structure detection
- **Smart chunking**: 300 +/- 50 tokens with 50-token overlap, sentence-aware boundaries
- **BGE embeddings**: 1024-dim vectors via BGE-large-en-v1.5 (Unix socket server)
- **Cross-encoder reranking**: Optional reranking with BGE-reranker-v2-m3

## Installation

```bash
pip install -e packages/pdf-tools
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

# 3. Embed chunks (requires running embed server)
# Start server: python -m research_kb_pdf.embed_server &
client = EmbeddingClient()
embeddings = client.embed_chunks(chunks)
print(f"Generated {len(embeddings)} embeddings")
```

## Embedding Server

The embedding server runs as a separate process, communicating via Unix socket:

```bash
# Start embedding server
python -m research_kb_pdf.embed_server &

# Socket default: /tmp/research_kb_embed.sock
```

## Reranker Server

Optional cross-encoder reranker for improved search precision:

```bash
# Start reranker (GPU recommended)
python -m research_kb_pdf.rerank_server &
```

## Testing

```bash
pytest packages/pdf-tools/tests/ -v
```
