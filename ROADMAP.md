# Research KB Roadmap

A semantic search system for research literature with graph-boosted retrieval across 36 domains. Combines full-text search, vector similarity (BGE-large-en-v1.5, 1024 dimensions), citation authority scoring, and knowledge graph signals (KuzuDB).

## Current Status

See [`docs/status/CURRENT_STATUS.md`](docs/status/CURRENT_STATUS.md) for live database metrics (auto-generated).

See [`docs/STRATEGIC_ASSESSMENT.md`](docs/STRATEGIC_ASSESSMENT.md) for:
- Strategic diagnosis and value delivery roadmap
- Complete phase history (Phases 1-4, D through bulk ingestion)
- Knowledge graph reconnection decision
- Sprint history and prioritized next steps

## Architecture

12 packages: contracts, common, storage, cli, api, daemon, dashboard, mcp-server, client, pdf-tools, extraction, s2-client

22 MCP tools | 36 populated domains | 2,815+ tests | CI coverage gate 70%

---

## Future Work (Contributions Welcome)

### Retrieval Quality
- **Learned weight optimization**: Tune FTS/vector/graph/citation weights on golden dataset
- **Multi-vector retrieval**: ColBERT-style late interaction for fine-grained matching
- **Adaptive chunking**: Use document structure (sections, paragraphs) instead of fixed token windows

### New Domains
- **Biology/Genomics**: Pathway analysis literature
- **Climate science**: Climate modeling papers
- **Your domain**: See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a new domain

### Infrastructure
- **Kubernetes deployment**: Helm chart for production deployment
- **Streaming search**: Server-sent events for real-time result streaming
- **Multi-user**: Authentication and per-user corpora

### Knowledge Graph
- **Temporal reasoning**: Track how assumptions/methods evolve across publications
- **Contradiction detection**: Flag conflicting claims across papers

### Dashboard
- **Screenshot/GIF capture**: Visual showcase for README and docs
- **Performance page**: Benchmark visualization from `fixtures/benchmarks/`
- **Comparison mode**: Side-by-side search results with different weight profiles
