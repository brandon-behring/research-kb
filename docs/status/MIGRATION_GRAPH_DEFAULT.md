# Migration: Graph Search Defaults

**Status**: Complete (graph disabled by default since 2026-03-25)

---

## Summary

Graph-boosted search is **disabled by default** across all surfaces (CLI, MCP, API, dashboard, daemon, client SDK). The `--graph` flag enables it when KG data is current.

Default search uses 3 signals:
- **FTS** (BM25 keyword matching)
- **Vector** (BGE-large cosine similarity)
- **Citation** (PageRank authority)

Graph signal is opt-in via `--graph` flag. Currently returns empty results because `chunk_concepts = 0` (KG re-extraction deferred -- see `docs/STRATEGIC_ASSESSMENT.md` Section 7).

---

## CLI Flags

```bash
# Default: FTS + vector + citation (3-way)
research-kb search query "instrumental variables"

# Enable graph signal (returns empty until KG re-extracted)
research-kb search query "instrumental variables" --graph

# Customize citation weight
research-kb search query "instrumental variables" --citation-weight 0.25

# Disable citation signal
research-kb search query "instrumental variables" --no-citations
```

---

## Graceful Fallback

When graph is enabled but no chunk_concepts exist, graph search silently returns 0 graph scores. The other signals (FTS, vector, citation) still produce results. No error, no configuration change needed.

---

## History

- **Phase D** (2025-12): Graph search enabled by default via KuzuDB
- **2026-03-25**: Graph defaults flipped OFF across all 7 surfaces after North Star validation revealed `chunk_concepts = 0` (stale KG produces cross-contaminated results)
- **Deferred**: KG re-extraction blocked by Anthropic API credit exhaustion. See `docs/STRATEGIC_ASSESSMENT.md` Section 7 for cost estimates and trigger conditions

---

## CI Validation

- **PR checks** (automated): Unit + integration tests, coverage gate 70%, doc freshness
- **Integration test** (manual, `workflow_dispatch`): Search pipeline with mocked data
- **Weekly full rebuild** (manual, `workflow_dispatch`): End-to-end with real embeddings, retrieval eval (MRR >= 0.85 gate on core domains)
