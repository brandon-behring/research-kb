# Migration: Graph Search as Default

**Status**: Complete

Graph-boosted search has been the default CLI behavior since Phase D. The `--no-graph` flag disables it.

---

## Summary

The `research-kb query` command uses 4-signal hybrid search by default:
- **FTS** (BM25 keyword matching)
- **Vector** (BGE-large cosine similarity)
- **Graph** (KuzuDB concept traversal)
- **Citation** (PageRank authority)

Graph and citation signals are opt-in via `--graph` / `--citations` flags (graph is enabled by default).

---

## CLI Flags

```bash
# Default: FTS + vector + graph
research-kb query "instrumental variables"

# Disable graph signal
research-kb query "instrumental variables" --no-graph

# Customize graph weight
research-kb query "instrumental variables" --graph-weight 0.3
```

---

## Graceful Fallback

When no concepts are extracted (fresh database), graph search silently falls back to FTS + vector. No error, no configuration change needed.

---

## CI Validation

Both CI workflows validate the search pipeline:

- **`integration-test.yml`** (manual trigger): Tests search code paths with mocked data
- **`weekly-full-rebuild.yml`** (manual trigger): End-to-end validation with real embeddings and golden dataset eval (MRR >= 0.5 gate)

---

## Performance

| Path | Latency |
|------|---------|
| KuzuDB graph scoring (warm) | ~150ms |
| PostgreSQL CTE fallback | max 2s (timeout) |
| FTS + vector only | ~200ms |
| Full 4-signal (warm) | ~2.1s |

KuzuDB pre-warming runs on daemon startup to avoid cold-start latency.

---

## References

- CLI implementation: `packages/cli/src/research_kb_cli/main.py`
- Graph scoring: `packages/storage/src/research_kb_storage/graph_queries.py`
- KuzuDB store: `packages/storage/src/research_kb_storage/kuzu_store.py`
- Full pipeline CI: `.github/workflows/weekly-full-rebuild.yml`

---

**Last Updated**: 2026-02-20
