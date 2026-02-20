---
layout: default
title: research-kb
---

# research-kb

**Graph-boosted semantic search for research literature.**

Combines full-text search (BM25), vector similarity (BGE-large 1024d), knowledge graph traversal (KuzuDB), and citation authority scoring (PageRank) into a unified retrieval system.

## Key Features

- **4-signal hybrid search** -- BM25 + vector + knowledge graph + citation authority
- **19-tool MCP server** -- conversational access via Claude Code
- **Knowledge graph** -- 307K concepts and 742K relationships served by KuzuDB
- **Multi-domain** -- causal inference, time series, RAG/LLM, and extensible
- **Interactive dashboard** -- Streamlit UI with graph visualization

## Architecture

```
Query
  |
  +--> Embed (BGE-large-en-v1.5, 1024d)
  |
  +--> Execute in parallel:
  |      FTS (PostgreSQL ts_rank)
  |      Vector (pgvector cosine similarity)
  |      Graph (KuzuDB concept traversal)
  |      Citation (PageRank authority)
  |
  +--> Weighted fusion
  |
  +--> Return top-K results
```

## Quick Start

```bash
# Start infrastructure
docker-compose up -d

# Install
pip install -e packages/cli

# Set up demo corpus (25 open-access arXiv papers)
python scripts/setup_demo.py

# Search
research-kb query "instrumental variables"
```

## Pages

- [Features](features) -- Detailed feature overview
- [Architecture](architecture) -- System design and package structure
- [Getting Started](getting-started) -- Installation and setup guide

## Links

- [GitHub Repository](https://github.com/brandonmbehring-dev/research-kb)
- [Contributing Guide](https://github.com/brandonmbehring-dev/research-kb/blob/main/CONTRIBUTING.md)
