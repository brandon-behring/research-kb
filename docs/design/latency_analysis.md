# Latency Analysis: Graph Signal Bottleneck

**Date**: 2026-01-15 (original benchmarks)
**Updated**: 2026-02-05 (post-KuzuDB migration)
**Author**: Audit remediation (Phase C)

---

## Executive Summary

**Pre-KuzuDB** (Jan 2026): Graph-boosted search added **85+ seconds** to query latency due to O(N×M) recursive CTE queries in `compute_weighted_graph_score()`.

**Post-KuzuDB** (current): KuzuDB serves as the primary graph engine with ~150ms batch scoring. PostgreSQL recursive CTEs remain as fallback with a 2-second timeout (`GRAPH_SCORE_TIMEOUT = 2.0`). This reduces worst-case graph overhead from 85s to 2s.

---

## Benchmark Results

### Pre-KuzuDB (January 2026 — Historical)

| Configuration | Latency | Delta |
|--------------|---------|-------|
| Baseline (FTS + vector) | 10.8s | - |
| + Citations only | 10.4s | -0.4s |
| + Graph only | **96.6s** | +85.8s |
| Full (all signals) | **94.5s** | +83.7s |

### Post-KuzuDB Architecture — Measured (Phase D, 2026-02-05)

**Corpus**: 439 sources, 170K chunks, 284K concepts, 726K relationships

| Operation | p50 | p95 | Notes |
|-----------|-----|-----|-------|
| Health check (baseline) | 20ms | 22ms | Socket round-trip + DB ping |
| `fast_search` (vector-only) | 208ms | 212ms | ~150ms embed + ~30ms pgvector |
| Hybrid search (FTS + vector) | 3.4s | 3.4s | Normalization overhead dominates |
| Graph path (KuzuDB) | 3.1s | 5.8s | Concept resolution (ILIKE) adds variance; warm cache ~1.7s |
| Graph-boosted search (warm) | **2.1s** | — | After first query warms KuzuDB |
| Graph-boosted search (cold) | **60s** | 60s | First query after daemon restart; primary optimization target |

**Source**: `fixtures/benchmarks/graph_latency_2026-02-05.json`

### Key Findings

1. **`fast_search` meets target**: 208ms p50 vs 500ms target — 2.4x margin
2. **Hybrid search bottleneck**: 3.4s is dominated by FTS+vector normalization, not graph
3. **Graph-boosted cold start**: 60s first query is the main pain point — KuzuDB graph scoring across 170K chunks needs warm cache
4. **Warm graph queries**: 2.1s when KuzuDB cache is hot — within acceptable range for interactive use
5. **Graph path variance**: Concept resolution via ILIKE adds 1-4s overhead; indexed exact-match lookups would help

---

## Root Cause Analysis

### The Problem: O(N×M) Recursive Queries

**File**: `packages/storage/src/research_kb_storage/graph_queries.py:604-698`

```python
async def compute_weighted_graph_score(...):
    for q_id in query_concept_ids:      # N = query concepts
        for c_id in chunk_concept_ids:   # M = chunk concepts per result
            path = await find_shortest_path(q_id, c_id, max_hops)  # 2 recursive CTEs
```

**Complexity per search**:
- Query concepts: ~3-5 (extracted from search text)
- Chunk concepts per result: ~10-30
- Results: 10 (limit)
- Total recursive queries: 3 × 20 × 10 × 2 = **1,200 recursive CTEs**

Each recursive CTE traverses 726,009 relationships with max_hops=2.

### Why Citations Are Fast

Citation authority uses a single batch query:
```python
source_authorities = await SourceStore.get_citation_authority_batch(source_ids)
```

This is O(1) with respect to result count — one query returns all authority scores.

---

## Optimization Options (Status as of 2026-02-05)

### Option 1: Batch Path Queries — Not Implemented

Replace per-pair calls with a single batch CTE. Superseded by KuzuDB migration.

**Expected improvement**: 60-80% latency reduction (PostgreSQL-only path)

### Option 2: Precompute Co-occurrence Matrix — Not Implemented

Materialize concept → concept distances. Superseded by KuzuDB.

### Option 3: Graph-Native Engine — IMPLEMENTED (KuzuDB)

KuzuDB embedded graph database serves as primary graph engine:

- `graph_queries.py:31-44` — Optional KuzuDB imports
- `graph_queries.py:54-126` — KuzuDB readiness checks with caching
- `graph_queries.py:762-774` — Batch graph scoring via KuzuDB (~129ms)
- `graph_queries.py:984-1014` — Weighted scoring via KuzuDB (~150ms)
- `scripts/sync_kuzu.py` — Syncs PostgreSQL → KuzuDB

Data lives at `~/.research_kb/kuzu/research_kb.kuzu` (~110MB).

### Option 4: Timeout with Partial Results — IMPLEMENTED

Timeout fallback protects against KuzuDB unavailability:

```python
GRAPH_SCORE_TIMEOUT = 2.0  # seconds (graph_queries.py:50)
PATH_QUERY_TIMEOUT = 0.5   # seconds (graph_queries.py:51)
```

PostgreSQL recursive CTEs fire only when KuzuDB is unavailable, capped at 2s.

---

## Current Architecture

```
Query → KuzuDB batch scoring (~150ms)
         ↓ (on failure)
       PostgreSQL recursive CTEs (2s timeout → score=0.0)
```

## Remaining Optimization Opportunities

1. ~~**Validate production latency**~~ — DONE (Phase D, 2026-02-05). See table above.
2. **KuzuDB sync freshness** — Nightly sync timer installed (`scripts/systemd/research-kb-sync.timer`)
3. **Cold-start optimization** — Pre-warm KuzuDB cache on daemon start by running a dummy graph query
4. **Concept resolution** — Add index on `concepts.canonical_name` for faster ILIKE resolution
5. **Normalization overhead** — Profile the 3.4s hybrid search to identify FTS/vector normalization bottleneck
6. **Prometheus observability** — `daemon_request_duration_seconds` histogram now tracks per-method latency

---

## Files Involved

| File | Role |
|------|------|
| `packages/storage/src/research_kb_storage/graph_queries.py` | Graph scoring + KuzuDB integration |
| `packages/storage/src/research_kb_storage/kuzu_store.py` | KuzuDB operations (749 lines) |
| `packages/storage/src/research_kb_storage/search.py` | Search orchestration |
| `scripts/sync_kuzu.py` | PostgreSQL → KuzuDB sync |
| `docs/archive/gemini_audit_report_2026-01-08.md` | Strategic recommendations |

---

## References

- Gemini Audit Report §6.1: "The Performance Bottleneck (Postgres vs. Proactive)"
- Master Plan Lines 616-673: Phase 2 knowledge graph
