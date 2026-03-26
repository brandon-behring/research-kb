---
paths:
  - "packages/**"
---

# Architecture Details

## KuzuDB Graph Engine

KuzuDB serves as the primary graph traversal engine, with PostgreSQL recursive CTEs as fallback:

- **Data**: `~/.research_kb/kuzu/research_kb.kuzu` (~110MB, mirrors PostgreSQL concepts/relationships)
- **Sync**: `python scripts/sync_kuzu.py` (run after ingestion/extraction)
- **Performance**: ~150ms batch scoring vs ~96s PostgreSQL CTEs
- **Fallback**: 2-second timeout on PostgreSQL path (`GRAPH_SCORE_TIMEOUT = 2.0`)
- **Code**: `packages/storage/src/research_kb_storage/kuzu_store.py` (749 lines)
- **Integration**: `graph_queries.py` tries KuzuDB first, PostgreSQL on failure

Key enums:
- `ConceptType`: METHOD, ASSUMPTION, PROBLEM, DEFINITION, THEOREM, CONCEPT, PRINCIPLE, TECHNIQUE, MODEL
- `RelationshipType`: REQUIRES, USES, ADDRESSES, GENERALIZES, SPECIALIZES, ALTERNATIVE_TO, EXTENDS, RELATED_TO

## Hybrid Search Weight Tuning

Default (3-way): FTS + vector + citation authority. Graph disabled pending re-extraction (~$250).

```
score = fts_weight × fts + vector_weight × vector + citation_weight × citation
```

**Signals:**
- **FTS**: PostgreSQL full-text search (keyword matching)
- **Vector**: BGE-large cosine similarity (semantic matching)
- **Graph**: Concept co-occurrence boost — **disabled by default** (stale chunk IDs)
- **Citation**: PageRank-style authority score — **enabled by default**

Context types adjust FTS + vector weights (citation adds 15% on top, normalized):
- **building**: 20% FTS, 80% vector (favor semantic breadth)
- **auditing**: 50% FTS, 50% vector (favor precision)
- **balanced**: 30% FTS, 70% vector (default)

Enable graph with `--graph` flag when KG data is re-extracted.

## HyDE (Hypothetical Document Embeddings)

Optional query expansion — generates a hypothetical document to improve embedding quality for terse queries.

```python
from research_kb_storage import HydeConfig, get_hyde_embedding

config = HydeConfig(
    enabled=True,
    backend="ollama",  # or "anthropic"
    model="llama3.1:8b",  # or "claude-3-5-haiku-20241022"
    max_length=200,
)

embedding = await get_hyde_embedding("IV assumptions", config)
```

Benefits: 5-10% improvement on terse queries, graceful fallback if LLM unavailable, configurable backend (Ollama dev / Anthropic prod).
