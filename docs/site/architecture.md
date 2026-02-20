---
layout: default
title: Architecture
---

# Architecture

## Package Dependency Graph

```
contracts (pure Pydantic models)
    |
common (logging, retry, instrumentation)
    |
    +---> storage (PostgreSQL + pgvector + KuzuDB)
    |       +---> cli
    |       +---> pdf-tools
    |       +---> extraction
    |       +---> api
    |       +---> dashboard
    |       +---> daemon
    |       +---> mcp-server
    |       +---> client
    +---> pdf-tools
    +---> extraction
    +---> s2-client
```

## Packages

| Package | Purpose | Technology |
|---------|---------|------------|
| **contracts** | Shared data models | Pydantic |
| **common** | Logging, retry, instrumentation | structlog, tenacity, OpenTelemetry |
| **storage** | Database + search orchestration | asyncpg, pgvector, KuzuDB |
| **pdf-tools** | PDF extraction + embedding | PyMuPDF, GROBID, BGE-large |
| **extraction** | Concept extraction from text | Ollama, Anthropic |
| **cli** | Command-line interface | Typer |
| **api** | REST endpoints + health checks | FastAPI |
| **daemon** | Low-latency query service | asyncio, Unix sockets |
| **mcp-server** | MCP tool server for Claude Code | MCP SDK |
| **dashboard** | Visual search + graph explorer | Streamlit, PyVis, Plotly |
| **s2-client** | Semantic Scholar API client | httpx |
| **client** | Python client library | JSON-RPC 2.0 |

## Database Schema

**Core tables**: `sources`, `chunks`, `citations`

**Knowledge graph**: `concepts`, `concept_relationships`, `chunk_concepts`, `methods`, `assumptions`

**Caching**: `method_assumption_cache`, `method_aliases`

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| BGE-large-en-v1.5 (1024d) | Single model ensures embedding consistency across all chunks |
| KuzuDB embedded graph | Solved O(N*M) recursive CTE bottleneck: 85s to 2.1s |
| Weighted sum over RRF | Validated 5:1 superiority on 47-query golden dataset |
| asyncpg connection pooling | Handles concurrent MCP + API + CLI requests |
| JSONB metadata columns | Extensible without schema migrations |

## Performance

| Operation | p50 Latency |
|-----------|-------------|
| Health check | 20ms |
| Vector search (fast path) | 208ms |
| Graph-boosted search (warm) | 2.1s |
| Graph path query (KuzuDB) | 3.1s |

The graph-boosted warm latency of 2.1s represents a **40x improvement** from the pre-KuzuDB architecture (85s).
