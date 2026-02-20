---
layout: default
title: Features
---

# Features

## Hybrid Search (4 Signals)

research-kb combines four independent retrieval signals:

| Signal | Engine | What It Captures |
|--------|--------|------------------|
| **FTS** | PostgreSQL `ts_rank` | Exact keyword matching |
| **Vector** | pgvector (BGE-large 1024d) | Semantic similarity |
| **Graph** | KuzuDB concept traversal | Knowledge structure |
| **Citation** | PageRank over citation network | Academic authority |

Weights adapt to search context:
- **Building** (20% FTS, 80% Vector) -- broad semantic exploration
- **Auditing** (50% FTS, 50% Vector) -- precise fact-checking
- **Balanced** (30% FTS, 70% Vector) -- general research

## Knowledge Graph

307K concepts and 742K relationships extracted from research literature via LLM-powered concept extraction.

**Concept types**: METHOD, ASSUMPTION, PROBLEM, DEFINITION, THEOREM, CONCEPT, PRINCIPLE, TECHNIQUE, MODEL

**Relationship types**: REQUIRES, USES, ADDRESSES, GENERALIZES, SPECIALIZES, ALTERNATIVE_TO, EXTENDS

## Assumption Auditing

The "North Star" feature: for any causal inference method, enumerate the assumptions required for valid conclusions.

```bash
research-kb audit-assumptions "instrumental variables"
```

Returns structured output with assumption definitions, importance levels, testability, and source references.

## MCP Server (19 Tools)

Plug into Claude Code for conversational access to the entire system:

- **Search** -- hybrid search with domain filtering
- **Graph** -- concept neighborhoods and path finding
- **Citations** -- upstream/downstream influence analysis
- **Assumptions** -- method assumption extraction

## Citation Network

15K+ citation links with PageRank authority scoring and bibliographic coupling for related-work discovery.

## Multi-Domain Support

Extensible domain system with LLM-powered concept extraction prompts tailored per domain:

| Domain | Sources |
|--------|---------|
| Causal Inference | 88 |
| RAG/LLM | 75 |
| Time Series | 49 |
| Econometrics | 35 |
| Deep Learning | 32 |
| ... and 14 more | |

## Interactive Dashboard

Streamlit-based UI with 6 pages:
- **Search** -- hybrid search with score breakdown
- **Citation Network** -- interactive PyVis visualization
- **Concept Graph** -- neighborhood exploration
- **Assumption Audit** -- interactive assumption checking
- **Statistics** -- corpus composition charts
- **Extraction Queue** -- pipeline monitoring
