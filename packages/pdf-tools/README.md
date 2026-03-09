# research-kb-pdf

PDF extraction, chunking, and embedding for the research-kb system.

## Features

- **Docling extraction**: LaTeX-preserving extraction via Granite-Docling-258M
- **GROBID integration**: Academic paper parsing with IMRAD structure detection
- **Structure-aware chunking**: Docling HybridChunker (~300 tokens, heading-boundary-aware)
- **BGE embeddings**: 1024-dim vectors via BGE-large-en-v1.5 (Unix socket server)
- **Cross-encoder reranking**: Optional reranking with BGE-reranker-v2-m3

## Installation

```bash
pip install -e packages/pdf-tools
```

## Usage

```python
from research_kb_pdf import extract_and_chunk, EmbeddingClient

# 1. Extract + chunk with Docling (LaTeX-preserving)
result, chunks = extract_and_chunk("paper.pdf", max_tokens=300)
print(f"{result.total_pages} pages, {len(chunks)} chunks, equations: {result.has_equations}")

# 2. Embed chunks (requires running embed server)
# Start server: python -m research_kb_pdf.embed_server &
client = EmbeddingClient()
for chunk in chunks:
    embedding = client.embed(chunk.content)
    print(f"Chunk {chunk.chunk_index}: {len(embedding)} dims")
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
