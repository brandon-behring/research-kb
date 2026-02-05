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

### Post-KuzuDB Architecture (Current)

| Component | Expected Latency | Source |
|-----------|-----------------|--------|
| KuzuDB batch graph scoring | ~129-150ms | `graph_queries.py:762` comment |
| KuzuDB path finding | ~74ms | `graph_queries.py:194` comment |
| KuzuDB neighborhood | ~20ms | `graph_queries.py:469` comment |
| PostgreSQL fallback (timeout) | max 2.0s | `GRAPH_SCORE_TIMEOUT` at line 50 |

**Note**: These are code-documented estimates from development. Run Phase D benchmarking for validated production numbers.

**Key finding**: KuzuDB migration (Option 3 from original analysis) was implemented, reducing graph signal from the sole bottleneck to a minor contributor.

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

1. **Validate production latency** — Run Phase D benchmarks to get real p50/p95/p99 numbers
2. **KuzuDB sync freshness** — Ensure nightly sync after ingestion runs
3. **Consider Option 1 (batch CTE)** as improved PostgreSQL fallback path

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
