# Search Quality Analysis: Domain Filtering + Reranker + Expansion

**Date**: 2026-02-06
**Golden Dataset**: 47 queries across 4 domains (causal_inference=17, time_series=10, rag_llm=10, interview_prep=10)
**Scoring Method**: Weighted sum (default) — see per-config notes for RRF comparison

---

## Executive Summary

Domain filtering alone delivers the biggest quality improvement (+56% Hit@10 relative to baseline).
Reranking and expansion add marginal gains but introduce CUDA OOM risks on GPUs with limited VRAM (8GB).
**Recommended default**: Domain filtering ON, reranker OFF, expansion OFF.

---

## Configuration Comparison (Weighted Scoring)

| Metric | Baseline | +Domain Filter | +Reranker | +Expand+Rerank |
|--------|----------|----------------|-----------|----------------|
| **Hit Rate@5** | 27.7% | **42.6%** | 31.9% | 31.9% |
| **Hit Rate@10** | 34.0% | **53.2%** | 48.9% | 48.9% |
| **MRR** | 0.194 | **0.281** | 0.225 | 0.222 |
| **NDCG@5** | 0.207 | **0.306** | 0.231 | 0.229 |
| **NDCG@10** | 0.228 | **0.340** | 0.286 | 0.284 |
| **Avg Latency** | 16,787ms | 12,643ms | 5,537ms | 6,840ms |

### Key Observations

1. **Domain filtering is the clear winner**: +15pp Hit@5, +19pp Hit@10 vs baseline
2. **Reranker+domain actually regresses from domain-only**: Hit@5 drops from 42.6% to 31.9%
3. **Expansion adds no measurable improvement** over reranker-only (identical numbers)
4. **Latency improves with reranker/expansion** because KuzuDB was available (vs PostgreSQL fallback for baseline/domain)
5. **CUDA OOM**: Reranker intermittently fails on causal_inference queries (50 candidates × long text), falling back to unranked results

---

## Per-Domain Breakdown (Weighted Scoring)

| Domain | Config | Hit@5 | Hit@10 | MRR | NDCG@5 |
|--------|--------|-------|--------|-----|--------|
| **causal_inference** | Baseline | 35.3% | 47.1% | 0.250 | 0.262 |
| | +Domain | 41.2% | 52.9% | 0.269 | 0.291 |
| | +Reranker | 41.2% | 52.9% | 0.301 | 0.315 |
| | +Expand+Rerank | 41.2% | 52.9% | 0.301 | 0.315 |
| **rag_llm** | Baseline | 20.0% | 30.0% | 0.133 | 0.139 |
| | +Domain | **70.0%** | **90.0%** | 0.397 | 0.454 |
| | +Reranker | 40.0% | 70.0% | 0.257 | 0.263 |
| | +Expand+Rerank | 40.0% | 70.0% | 0.244 | 0.252 |
| **time_series** | Baseline | 40.0% | 40.0% | 0.333 | 0.350 |
| | +Domain | 40.0% | 40.0% | 0.333 | 0.350 |
| | +Reranker | 30.0% | 50.0% | 0.179 | 0.186 |
| | +Expand+Rerank | 30.0% | 50.0% | 0.179 | 0.186 |
| **interview_prep** | Baseline | 10.0% | 10.0% | 0.020 | 0.039 |
| | +Domain | 20.0% | 30.0% | 0.134 | 0.139 |
| | +Reranker | 10.0% | 20.0% | 0.110 | 0.100 |
| | +Expand+Rerank | 10.0% | 20.0% | 0.110 | 0.100 |

### Domain-Level Insights

1. **rag_llm sees dramatic improvement from domain filtering** (30% → 90% Hit@10), but reranking *hurts* it (90% → 70%)
   - Hypothesis: Domain filtering removes cross-domain noise. Reranker's cross-encoder reorders on surface-level text similarity, pushing relevant but differently-worded chunks lower.

2. **interview_prep remains the weakest domain** across all configs (max 30% Hit@10)
   - Root cause: Small corpus (19K chunks), golden targets may be too specific
   - The synonym expansions (ab_testing, sample_size, etc.) did not trigger — expansion_count=0 in logs

3. **causal_inference is stable**: Domain filtering adds modest gain, reranker improves MRR (+0.03) but not hit rate

4. **time_series unchanged by domain filtering** — likely already dominated by correct-domain results in baseline

---

## Weighted vs RRF Comparison

| Config | Weighted Wins | RRF Wins | Verdict |
|--------|--------------|----------|---------|
| Domain filter only | 6 | 0 | **Weighted dominant** |
| +Reranker | 3 | 3 | Tie |
| +Expand+Rerank | 3 | 3 | Tie |
| No domain filter | 5 | 1 | Weighted wins |

**Conclusion**: Keep weighted sum as default. RRF only competitive when reranking is active.

---

## Reranker Issues

1. **CUDA OOM on GPUs with limited VRAM (8GB)**:
   - BGE-reranker-v2-m3 (278M params) shares GPU with embedding model + pytest processes
   - Fails intermittently on batches of 50 candidates with long text
   - Graceful fallback works (returns unranked results), but silently degrades quality

2. **Reranker hurts rag_llm domain**: Cross-encoder text similarity may not align with domain-specific relevance for textbook content

3. **Fix options**:
   - Reduce `fetch_multiplier` from 5 to 3 (fewer candidates)
   - Use CPU-only reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`, 22M params)
   - Allocate dedicated GPU memory budget via `PYTORCH_CUDA_ALLOC_CONF`

---

## Query Expansion Issues

1. **Synonym expansion didn't fire for interview_prep queries**: The expansion logs show `expansion_count=0` for queries like "A/B testing", "statistical significance", "experiment design"
   - Root cause: `search_with_expansion` keys on the *original query text*, not the golden entry query text. The synonym map keys (`ab_testing`, `statistical_significance`) use underscores, but queries use natural language
   - Fix: Update QueryExpander to match natural language phrases, not just underscore-delimited keys

2. **No LLM expansion was used** (`use_llm_expansion=False`): This was intentional to keep evals deterministic, but may explain why expansion added zero value

---

## Recommendations

### Immediate (deploy now)
- **Enable domain filtering as default** in SearchQuery when domain is known
- **Keep reranker as opt-in** — not default, due to CUDA OOM and rag_llm regression

### Short-term (next sprint)
- Fix synonym map matching to handle natural language queries (not just `_`-delimited keys)
- Reduce reranker batch size or switch to lightweight model
- Add more interview_prep golden entries with broader targets

### Medium-term
- Investigate HyDE for interview_prep (terse queries → hypothetical doc → better embeddings)
- Profile reranker on CPU vs GPU with 22M model for latency comparison
- Consider domain-specific weight tuning (rag_llm may want higher vector weight)

---

## Benchmark Files

| File | Config | Notes |
|------|--------|-------|
| `fixtures/benchmarks/no_domain_filter.json` | Baseline (no domain filter) | 16.8s avg latency (PostgreSQL graph fallback) |
| `fixtures/benchmarks/domain_filtered.json` | Domain filter only | **Best quality** — recommended default |
| `fixtures/benchmarks/with_reranker.json` | Domain + reranker | CUDA OOM on some queries |
| `fixtures/benchmarks/full_pipeline.json` | Domain + expand + rerank | No improvement over reranker-only |
| `fixtures/benchmarks/rrf_vs_weighted.json` | Original RRF validation (Phase 4.3) | Pre-domain-filter baseline |

---

## Methodology Notes

- All evals used `search_hybrid_v2` (4-way: FTS + vector + graph + citation) with weights: FTS=0.2, Vector=0.4, Graph=0.2, Citation=0.2
- KuzuDB was available for reranker and full-pipeline evals (daemon stopped during eval). Baseline and domain-filter evals ran with PostgreSQL graph fallback (slower but same quality)
- Reranker: `BAAI/bge-reranker-v2-m3` via Unix socket, `fetch_multiplier=5`, `rerank_top_k=10`
- Expansion: `use_synonyms=True`, `use_graph_expansion=True`, `use_llm_expansion=False`
