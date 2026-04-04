# Evaluation System Design Spec

**Date**: 2026-04-01 (original) | **Updated**: 2026-04-03
**Status**: Active — Phases 1-3 complete, Phase 4+ in progress
**Implementation plan**: `.claude/plans/harmonic-brewing-lollipop.md` (27 decisions locked)
**Context**: Current eval_retrieval.py has 7 methodology flaws discovered during critical
review. Decision: start fresh with proper IR evaluation rather than patching.

---

## 1. Problems With Current Evaluation

### 1.1 Metric Errors

| Issue | Current Behavior | Correct Behavior |
|-------|-----------------|------------------|
| **MRR inflated** | Divides by successful query count | Divide by total query count (include 0 for failures) |
| **NDCG non-standard** | Single-result binary relevance | Graded relevance across all positions (or drop metric) |
| **Hit@K window** | `limit=expected_in_top_k` (never finds results beyond K) | Always fetch top-20+, check within K |

### 1.2 Ground Truth Weaknesses

| Issue | Current State | Required |
|-------|--------------|----------|
| **Source-level only** | Matches any chunk from right source | Need chunk-level AND source-level metrics |
| **Circular UUIDs** | UUIDs captured from current search results | Need independent ground truth (manual annotation) |
| **Non-discriminative grades** | 95% of test cases have grade=3 (max) | Meaningful graded relevance (0-3) |
| **107 test cases** | Too few for statistical significance per domain | ~150-200 with >=5 per domain |

### 1.3 Missing Capabilities

- No statistical significance testing (can't tell if change is real or noise)
- No chunk quality metrics (coherence, boundary quality)
- No diversity metrics (are results from multiple sources or one?)
- No latency tracking alongside quality metrics
- No cross-domain query evaluation

---

## 2. Requirements

### 2.1 Metrics (Minimum Viable)

**Must have (standard IR, binary relevance):**
- **MRR** — Mean Reciprocal Rank (over ALL queries, failed = 0)
- **Recall@K** — Fraction of relevant docs found in top K (for K=5, 10, 20)
- **Hit@K** — Binary: is ANY relevant doc in top K? (what we call "hit rate")
- **MAP** — Mean Average Precision (position-weighted, over all queries)

**Should have (requires graded relevance):**
- **NDCG@K** — Only if we invest in proper relevance annotations
- **Precision@K** — Fraction of top-K results that are relevant

**Must report at two granularities:**
- **Source-level**: Any chunk from expected source found (current behavior)
- **Chunk-level**: Expected CHUNK found (requires chunk-level ground truth)

### 2.2 Ground Truth Format

```yaml
test_cases:
  - id: "ci-001"
    query: "backdoor criterion"
    domain: causal_inference
    difficulty: easy
    
    # Source-level ground truth (lenient)
    relevant_sources:
      - source_id: "de63dc79-..."
        relevance: 3  # 0=irrelevant, 1=marginally, 2=relevant, 3=highly relevant
      - source_id: "abc12345-..."
        relevance: 2  # Another valid source, lower relevance
    
    # Chunk-level ground truth (strict, optional)
    relevant_chunks:
      - chunk_id: "fed98765-..."
        relevance: 3
    
    # Page range (proxy for chunk quality when chunk IDs not available)
    expected_page_range: [75, 375]
    
    # Expected concepts (for concept recall)
    expected_concepts: ["backdoor criterion", "d-separation"]
```

Key changes from current format:
- **Multiple relevant sources per query** (not just one)
- **Graded relevance per source** (not global per test case)
- **Chunk-level annotations** (optional but important)
- **Stable IDs** (not dependent on current search results)

### 2.3 Evaluation Pipeline

```
1. Load test cases (YAML)
2. For each query:
   a. Generate embedding
   b. Run search with FIXED top-K (always 20)
   c. Record ranked result list (source IDs + chunk IDs + scores)
3. Compute metrics:
   a. Source-level: MRR, Recall@K, Hit@K, MAP
   b. Chunk-level: same metrics against chunk ground truth
   c. Per-domain breakdown
   d. Per-difficulty breakdown
4. Output:
   a. JSON results (for comparison tool)
   b. Human-readable report
   c. Statistical summary
```

### 2.4 Comparison & Significance

- Paired comparison between two runs (same queries, different configs)
- Per-query deltas (which queries improved/regressed)
- Statistical significance: paired bootstrap or randomization test
- Domain-level significance (only report domain improvement if p < 0.05)

### 2.5 Integration Points

- `eval_compare.py` — existing tool, update to work with new format
- CI gate — weekly-full-rebuild uses `--fail-below` threshold
- Ablation workflow — run same queries with different configs, compare

---

## 3. Framework Decision: ranx

**Decided 2026-04-01.** See `docs/research/ir_eval_frameworks_2026.md` for full comparison.

**ranx** selected because:
1. All 14 standard IR metrics (MRR, MAP, NDCG, Recall, Precision, Hit Rate, etc.)
2. Per-query breakdown via `return_mean=False`
3. Built-in statistical significance (paired t-test, Fisher's randomization, Tukey HSD)
4. TREC qrels format (dict-of-dicts) maps directly to our YAML
5. Comparison tables with significance markers
6. Active maintenance (ECIR 2022, SIGIR 2023 papers)
7. `pip install ranx` — no C compiler needed

**Rejected:**
- pytrec_eval: No built-in comparison/significance, aging docs
- RAGAS: LLM-required metrics, targets generation not retrieval
- DeepEval: All metrics require LLM calls, not retrieval-focused
- Custom: Why reimplement what ranx already validates against trec_eval?

---

## 4. Implementation Plan

### Step 1: Install ranx, build eval_v2.py

New script `scripts/eval_v2.py` that:
- Loads YAML test cases (new format with multiple relevant docs)
- Runs search with fixed limit=100 (not expected_in_top_k)
- Builds ranx Qrels and Run objects
- Computes: NDCG@10, Recall@10, MRR@10, MAP@10, Hit@10
- Per-domain and per-difficulty breakdowns
- Outputs JSON compatible with eval_compare.py

Keep `eval_retrieval.py` as-is (legacy, v1 methodology). New script is v2.

### Step 2: Create ground truth (TREC pooling)

1. Select 50-75 queries (stratify across domains and query types):
   - 30 from existing 107 test cases (best-covered domains)
   - 10-15 new cross-domain queries (the North Star)
   - 10-15 new queries for thin domains
2. For each query, run search with multiple configs:
   - FTS-only, vector-only, hybrid, hybrid+citations
   - Pool top-20 from each = ~40-60 unique chunks per query
3. Manually judge each pooled chunk on 4-point scale (0-3)
4. Store as TREC qrels file + YAML index

**Effort estimate:** ~2,500 judgments at 30 sec each = ~20 hours.
Can be done incrementally (10 queries per session).

### Step 3: Corrected baseline

- Run eval_v2.py with current search config
- This becomes the v2 baseline
- All v1 results (evaluation_runs/*.json) archived as "methodology v1"

### Step 4: Re-validate ablations with corrected methodology

- Citation ON/OFF: re-run with ranx, check if +14.3% MRR holds
- Reranking ON/OFF: CRITICAL re-validation — the limit=expected_in_top_k bug
  may have caused the -14.7% MRR regression. With limit=100, reranking may
  actually help.

### Step 5: Update eval_compare.py

- Support ranx output format alongside v1 JSON
- Add significance reporting (from ranx compare())

---

## 5. Resolved Questions

1. **Framework**: ranx (decided 2026-04-01)
2. **LLM-as-judge**: NO — Soboroff 2025 argues persuasively against it. Human judgments only.
3. **Chunk-level vs source-level**: BOTH — chunk-level primary, source-level derived.

## 6. Mitigation: Multiple Relevant Docs Per Query

**Risk:** If each query has only 1 relevant doc, NDCG degrades to binary relevance —
the same mistake we just found in eval_retrieval.py.

**Mitigation:** TREC pooling is designed to produce MULTIPLE relevant docs per query.
The pooling process (union of top-20 from each variant) typically surfaces 40-60
unique chunks per query. Judging all of them at 4 grades produces a rich relevance
distribution.

**Minimum target:** Each query should have:
- 2-5 chunks graded 3 (highly relevant)
- 3-8 chunks graded 2 (relevant)
- 5-10 chunks graded 1 (marginally relevant)
- Rest graded 0

If TREC pooling doesn't produce enough relevant material for a query, that query
reveals a genuine corpus gap (not an annotation problem).

**Validation check:** After annotation, verify each query has >= 3 relevant docs
(grade >= 2). Queries with < 3 are flagged as "shallow ground truth" in metrics.

---

## 7. Mitigation: Re-Chunking Resilience

**Risk:** Phase 2 plans to improve chunking. Every re-chunk changes chunk IDs,
invalidating all chunk-level annotations.

**Mitigation: Dual-layer ground truth.**

For each relevant judgment, store:

```yaml
judgments:
  - query_id: "q_iv_001"
    # Layer 1: Chunk-level (precise, fragile)
    chunk_id: "fed98765-..."
    relevance: 3
    
    # Layer 2: Source+page (stable across re-chunking)
    source_id: "abc12345-..."
    page_range: [142, 148]
    section_path: "Chapter 7 > Instrumental Variables > Assumptions"
    relevance: 3
    
    # Layer 3: Content snippet (survives everything)
    evidence_text: "The exclusion restriction requires that the instrument..."
```

**After re-chunking:**
1. Layer 1 (chunk_id) → stale, needs re-mapping
2. Layer 2 (source+page) → still valid, re-map to new chunks by page overlap
3. Layer 3 (evidence_text) → re-map by fuzzy text matching to new chunks

**Re-mapping script:** Build `scripts/remap_annotations.py` that:
- Takes old qrels + new chunk IDs
- Matches by source_id + page_range overlap
- Falls back to evidence_text fuzzy matching
- Reports unmapped judgments for manual review

**Cost:** Small upfront overhead (store 3 layers instead of 1). Saves 20+ hours
of re-annotation when chunking changes.

---

## 8. Implementation Progress (2026-04-03)

**Full implementation plan**: `.claude/plans/harmonic-brewing-lollipop.md` (27 decisions)

### Completed

| Phase | Date | Deliverable | Key Finding |
|-------|------|-------------|-------------|
| 1 | 2026-04-01 | `--fetch-limit` flag, rank threshold | Reranking confirmed harmful: -44% true MRR |
| 1.2 | 2026-04-03 | Normalization pool-size verification | Min-max normalization is pool-size-dependent; irrelevant for v2/ranx |
| 2 | 2026-04-03 | 3 scripts: pool, prelabel, annotate | Smoke tested on 3 queries; all functional |
| 3a | 2026-04-03 | `fixtures/eval/v2_queries.yaml` (71 queries) | 3-tier protocol: 30 low-MRR + 25 sampled + 16 new |
| 3b | 2026-04-03 | `fixtures/eval/v2_pool_candidates.yaml` (3138 candidates) | TREC pooling across 4 configs, avg 44.2/query |
| 3c | 2026-04-03 | `fixtures/eval/v2_pool_prelabeled.yaml` | Percentile grading: 11.5% g3, 19.6% g2, 29.6% g1, 39.4% g0 |

### Remaining

| Phase | Status | Description |
|-------|--------|-------------|
| 4 | Not started | Build `eval_v2.py` with ranx (NDCG@10 primary metric) |
| 5 | Not started | Annotate ground truth (~8 hrs, Claude Code pre-labeled) |
| 6 | Not started | Corrected v2 baseline |
| 7 | Not started | Citation ablation with statistical significance |
| 8 | Not started | CI integration (2-4 week parallel period) |

### Key Decisions

- **Primary v2 metric**: NDCG@10 (position-weighted + graded relevance)
- **Reranking**: DROPPED from Phase 7 — conclusively harmful (-44% true MRR)
- **ranx doc_id**: chunk_id (system retrieves chunks; source-level derived by aggregation)
- **Pre-labeling**: Percentile-based per-query with absolute floor safety net
- **Cross-domain relevance**: "Does this chunk help answer the question as asked?"
- **Legacy 107 test cases**: Archived, not migrated; v2 ground truth built fresh

---

## 9. Resolved Questions (updated 2026-04-03)

1. ~~**Annotation effort**~~: ~8 hrs with Claude Code pre-labeling + human review (not 20 hrs)
2. ~~**Cross-domain queries**~~: Grade by helpfulness to the cross-domain question. Single-domain chunks → 1-2. Bridging chunks → 3
3. ~~**Reranking re-validation**~~: CLOSED. -44% true MRR. Cross-encoder destroys citation authority signal
4. ~~**Annotation ordering**~~: Annotate now with triple-layer (chunk_id, source+page, evidence_text)
5. **Framework**: ranx 0.3.21 (installed, Python 3.13 compatible)
6. **LLM-as-judge**: NO — human judgments only (Soboroff 2025)
7. **Chunk-level vs source-level**: BOTH — chunk-level primary, source-level derived
