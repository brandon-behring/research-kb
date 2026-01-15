# Latency Analysis: Graph Signal Bottleneck

**Date**: 2026-01-15
**Author**: Audit remediation (Phase C)

---

## Executive Summary

Graph-boosted search adds **85+ seconds** to query latency due to O(N×M) recursive CTE queries in `compute_weighted_graph_score()`. The root cause is per-pair shortest path computation.

---

## Benchmark Results

| Configuration | Latency | Delta |
|--------------|---------|-------|
| Baseline (FTS + vector) | 10.8s | - |
| + Citations only | 10.4s | -0.4s |
| + Graph only | **96.6s** | +85.8s |
| Full (all signals) | **94.5s** | +83.7s |

**Key finding**: Citations add negligible overhead. Graph signal is the sole bottleneck.

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

## Proposed Optimizations (Future Phases)

### Option 1: Batch Path Queries (Medium Effort)

Replace per-pair calls with a single batch CTE:

```sql
WITH RECURSIVE paths AS (
    -- Start from ALL query concepts at once
    SELECT source_id, target_id, depth, path
    FROM query_concepts, target_concepts
    ...
)
SELECT source_id, target_id, MIN(depth), path_weight
FROM paths
GROUP BY source_id, target_id;
```

**Expected improvement**: 60-80% latency reduction

### Option 2: Precompute Co-occurrence Matrix (High Effort)

Materialize concept → concept distances for frequent pairs:

```sql
CREATE MATERIALIZED VIEW concept_distances AS
SELECT
    c1.id AS source_id,
    c2.id AS target_id,
    compute_shortest_path_length(c1.id, c2.id, 2) AS distance
FROM concepts c1, concepts c2
WHERE c1.id != c2.id;
```

**Expected improvement**: 95%+ latency reduction
**Tradeoff**: High storage cost, refresh complexity

### Option 3: Graph-Native Engine (Strategic)

Migrate graph traversal to KuzuDB or NetworkX (in-memory):

- **KuzuDB**: Embedded graph DB, ~10ms for 4-hop queries
- **NetworkX**: In-memory Python, <1ms for 4-hop on 726K edges

**Expected improvement**: 99%+ latency reduction
**Tradeoff**: Architectural complexity, sync with PostgreSQL

### Option 4: Timeout with Partial Results (Quick Fix)

Add timeout to graph scoring with graceful fallback:

```python
try:
    async with asyncio.timeout(5.0):  # 5 second budget
        graph_score = await compute_weighted_graph_score(...)
except TimeoutError:
    graph_score = 0.0  # Fall back to FTS+vector only
```

**Expected improvement**: Predictable worst-case latency
**Tradeoff**: May miss graph signal on complex queries

---

## Recommendation

**Immediate (Phase 3.1)**: Implement Option 4 (timeout fallback) for usability.

**Short-term (Phase 3.2)**: Implement Option 1 (batch CTE) for 60-80% improvement.

**Strategic (Phase 5)**: Evaluate KuzuDB migration per Gemini audit recommendation.

---

## Files Involved

| File | Role |
|------|------|
| `packages/storage/src/research_kb_storage/graph_queries.py` | Bottleneck location |
| `packages/storage/src/research_kb_storage/search.py:366-391` | Graph score integration |
| `docs/archive/gemini_audit_report_2026-01-08.md` | Strategic recommendations |

---

## References

- Gemini Audit Report §6.1: "The Performance Bottleneck (Postgres vs. Proactive)"
- Master Plan Lines 616-673: Phase 2 knowledge graph
