# RRF vs Weighted Sum: Empirical Validation

**Date:** 2026-02-06
**Verdict:** Weighted sum wins. Keep as default.

## Background

RRF (Reciprocal Rank Fusion) was implemented in `search_hybrid_v2` (`search.py:497-511`) as an alternative to weighted-sum scoring. This document records the empirical comparison.

## Methodology

- **Golden dataset**: 47 queries across 4 domains (causal_inference, time_series, rag_llm, interview_prep)
- **Target matching**: Chunk-ID targeting (exact answer chunks identified per query)
- **Search function**: `search_hybrid_v2` with all 4 signals (FTS + vector + graph + citation)
- **Weights**: fts=0.2, vector=0.4, graph=0.2, citation=0.2
- **Graph engine**: PostgreSQL CTE fallback (KuzuDB locked by daemon during evaluation)

## Results

| Metric | Weighted | RRF | Winner |
|--------|----------|-----|--------|
| Hit Rate@5 | 27.7% | 23.4% | Weighted |
| Hit Rate@10 | 34.0% | 31.9% | Weighted |
| MRR | 0.194 | 0.156 | Weighted |
| NDCG@5 | 0.207 | 0.168 | Weighted |
| NDCG@10 | 0.228 | 0.195 | Weighted |
| Avg Latency | 15455ms | 15448ms | RRF (negligible) |

**Score: Weighted 5, RRF 1**

## Per-Domain Breakdown

| Domain | Method | Hit@5 | MRR | NDCG@5 |
|--------|--------|-------|-----|--------|
| causal_inference | Weighted | 35.3% | 0.250 | 0.262 |
| causal_inference | RRF | 29.4% | 0.214 | 0.217 |
| time_series | Weighted | 40.0% | 0.333 | 0.350 |
| time_series | RRF | 30.0% | 0.263 | 0.263 |
| rag_llm | Weighted | 20.0% | 0.133 | 0.139 |
| rag_llm | RRF | 20.0% | 0.075 | 0.106 |
| interview_prep | Weighted | 10.0% | 0.020 | 0.039 |
| interview_prep | RRF | 10.0% | 0.033 | 0.050 |

## Analysis

1. **Weighted sum wins across all domains** except interview_prep MRR (marginal)
2. **Hit rates are low overall** (~28-34%) because the golden dataset uses exact chunk-ID targeting against a 178K-chunk corpus. The target chunks were selected via FTS within a specific source, but `search_hybrid_v2` searches the entire corpus — many queries find relevant content from other sources first
3. **RRF performs worse** likely because the rank-based aggregation loses the magnitude information that weighted sum preserves. When FTS and vector strongly agree, weighted sum amplifies this; RRF treats a rank-1 with score 0.99 the same as rank-1 with score 0.51
4. **Latency is identical** — RRF reranking is O(n) and negligible compared to graph scoring

## Decision

**Keep `scoring_method="weighted"` as default** (`search.py:79`).

RRF remains available via `scoring_method="rrf"` for users who want parameter-free ranking, but empirical evidence does not support changing the default.

## Bug Fix: search_hybrid now supports RRF

Prior to this validation, `search_hybrid` (the 2-way FTS+vector path) ignored `scoring_method`. Fixed in `search.py:258-265` so that `eval_retrieval.py --scoring rrf` now works correctly.

## Reproduction

```bash
python scripts/eval_scoring_methods.py --verbose --output fixtures/benchmarks/rrf_vs_weighted.json
```

Data: `fixtures/benchmarks/rrf_vs_weighted.json`, `fixtures/eval/golden_dataset.json`
