# Evaluation Runs

Historical retrieval evaluation snapshots. Each run captures aggregate and per-domain
metrics from `scripts/eval_retrieval.py`.

## Usage

```bash
# Run eval and save snapshot
python scripts/eval_retrieval.py --use-citations --per-domain \
  --output evaluation_runs/<name>_YYYY-MM-DD.json

# Compare two runs
python scripts/eval_compare.py evaluation_runs/baseline.json evaluation_runs/candidate.json --per-domain
```

## Naming Convention

- `baseline_YYYY-MM-DD.json` — Reference baselines (don't delete)
- `ablation_<signal>_<on|off>_YYYY-MM-DD.json` — Ablation study results
- `post_<change>_YYYY-MM-DD.json` — After a specific change
- `experiment_<name>_YYYY-MM-DD.json` — Experimental configurations

## Run Index

| Date | File | Description | MRR | Hit@K | Methodology |
|------|------|-------------|-----|-------|-------------|
| 2026-03-30 | baseline_2026-03-30.json | First baseline (pre-guide-removal, regex patterns) | 0.787 | 92.5% | v1 (limit=K) |
| 2026-03-30 | post_guide_removal_2026-03-30.json | After removing 29 self-written guides | 0.800 | 92.5% | v1 (limit=K) |
| 2026-03-30 | post_fixes_2026-03-30.json | After pattern fixes + guide removal | 0.793 | 98.1% | v1 (limit=K) |
| 2026-03-31 | ablation_citations_off_2026-03-31.json | 2-way search (FTS+vector only) | 0.684 | 97.2% | v1 (limit=K) |
| 2026-03-31 | ablation_citations_on_2026-03-31.json | 3-way search (FTS+vector+citation) | 0.781 | 98.1% | v1 (limit=K) |
| 2026-04-01 | ablation_rerank_on_2026-04-01.json | 3-way + reranking (BUGGY: limit=K) | 0.667 | 91.6% | v1 (limit=K) |
| 2026-04-01 | ablation_rerank_corrected_2026-04-01.json | 3-way + reranking (corrected: limit=100) | 0.685 | 62.6% | v1.1 (limit=100) |
| 2026-04-03 | baseline_corrected_2026-04-03.json | 3-way, limit=100 (Phase 1.2 verification) | 0.769 | 95.3% | v1.1 (limit=100) |

### Methodology Note

**v1 (limit=K)**: Search fetched only `expected_in_top_k` results. All retrieved results were
within top-K by construction, inflating Hit@K and masking rank degradation from reranking.

**v1.1 (limit=100)**: Search fetches 100 results, evaluates whether expected result appears
within top-K. Reranker gets a fair 500→100 pool. This corrects the evaluation window bug
but retains the v1 MRR inflation (divides by successful queries, not total).

## Ablation Results

### Citation Signal (2026-03-31)

| Metric | Citations OFF | Citations ON | Delta | Change |
|--------|-------------|------------|-------|--------|
| Hit@K | 97.2% | 98.1% | +0.9% | +1.0% |
| MRR | 0.684 | 0.781 | +0.098 | **+14.3%** |
| NDCG@5 | 0.728 | 0.814 | +0.086 | **+11.9%** |
| NDCG@10 | 0.740 | 0.819 | +0.079 | **+10.7%** |

**Conclusion:** Citation authority signal is proven valuable. Keep ON by default.
15 domains improved, 5 regressed (analysis, finance, RL, signal_processing, sql).
Biggest gains: deep_learning (+0.42), recommender_systems (+0.44), machine_learning (+0.39).

### Reranking Signal (2026-04-01)

Baseline: citations ON, no reranking. Candidate: citations ON + cross-encoder reranking (bge-reranker-v2-m3).

| Metric | Rerank OFF | Rerank ON | Delta | Change |
|--------|-----------|----------|-------|--------|
| Hit@K | 98.1% | 91.6% | -6.5% | **-6.7%** |
| MRR | 0.781 | 0.667 | -0.115 | **-14.7%** |
| NDCG@5 | 0.814 | 0.676 | -0.138 | **-17.0%** |
| NDCG@10 | 0.819 | 0.685 | -0.134 | **-16.4%** |

**Conclusion:** Cross-encoder reranking HURTS retrieval quality. 16/31 domains regressed.
The reranker reorders results in ways that push expected sources out of top-K.
**Do NOT enable reranking.** Investigate root cause before reconsidering.

### Reranking Signal — Corrected (2026-04-01)

Corrected methodology: `--fetch-limit 100` (reranker gets 500→100 pool instead of K→K).
Baseline: citations ON, no reranking. Candidate: citations ON + cross-encoder reranking.

| Metric | Rerank OFF | Rerank ON (corrected) | Delta | Change |
|--------|-----------|----------------------|-------|--------|
| Hit@K | 98.1% | 62.6% | -35.5% | **-36.2%** |
| MRR | 0.781 | 0.685 | -0.096 | **-12.3%** |
| NDCG@5 | 0.814 | 0.474 | -0.340 | **-41.8%** |
| NDCG@10 | 0.819 | 0.477 | -0.342 | **-41.8%** |

19/31 domains regressed. Only 6 improved (algebra, biology_neuroscience, physics, rag_llm,
reinforcement_learning, software_engineering).

**Corrected conclusion:** Reranking is WORSE than previously measured. The v1 limit=K bug
actually MASKED the full severity — with a fair pool, reranking pushes expected sources
to ranks 20-53 in many cases. The bge-reranker-v2-m3 cross-encoder optimizes for
semantic similarity but destroys the citation-authority ranking signal.

**Root cause hypothesis:** The cross-encoder reranker scores by query-chunk textual relevance,
ignoring the citation authority and FTS signals that hybrid search uses. Sources with high
PageRank but moderate text overlap get demoted. This is a fundamental mismatch between
the two-stage retrieval paradigm and our multi-signal hybrid approach.

**Verdict: CONFIRMED — do NOT enable reranking.** Investigation closed.

### Phase 1.2: Normalization Pool-Size Effect (2026-04-03)

Baseline: `ablation_citations_on_2026-03-31.json` (v1, limit=K).
Candidate: `baseline_corrected_2026-04-03.json` (v1.1, limit=100).

| Metric | limit=K | limit=100 | Delta | Change |
|--------|---------|-----------|-------|--------|
| Hit@K | 98.1% | 95.3% | -2.8% | -2.9% |
| MRR | 0.781 | 0.769 | -0.013 | **-1.6%** |
| NDCG@10 | 0.819 | 0.787 | -0.032 | **-4.0%** |

25/31 domains identical. 5 regressed (mathematics, deep_learning, statistics,
causal_inference, recommender_systems). 1 improved (econometrics).

**Root cause:** Min-max normalization in `_hybrid_search()` is pool-size-dependent.
Fetching 100 results (instead of K) changes the normalization window for FTS/vector
scores. When citation authority (not pool-dependent) is blended in, the relative
weighting shifts, moving some results in/out of top-K.

**Conclusion:** The v1 limit=K metrics were inflated by an artificially narrow
normalization window. limit=100 is more honest. Does not block v2 pipeline —
all v1 absolute metrics are unreliable regardless. v2/ranx uses fixed limit=100.

---

## v2 Evaluation Pipeline (in progress)

**Design spec**: `docs/research/eval_design_spec.md`
**Implementation plan**: `.claude/plans/harmonic-brewing-lollipop.md`

### v2 Query Set

`fixtures/eval/v2_queries.yaml` — 71 queries across 35 domains (incl. cross_domain):
- Tier 1 (30): All queries from 10 low-MRR domains (MRR < 0.65)
- Tier 2 (25): 1-4 sampled per healthy domain
- Tier 3 (16): 6 gap-domain (healthcare, dynamical_systems, portfolio_management) + 10 cross-domain

### v2 Pooled Candidates

`fixtures/eval/v2_pool_candidates.yaml` — 3138 candidates (avg 44.2/query) from 4 configs:
- FTS-only, vector-only, hybrid (FTS+vector), hybrid+citations
- Top-20 per config, deduplicated by chunk_id

### v2 Pre-labeled Candidates

`fixtures/eval/v2_pool_prelabeled.yaml` — percentile-based suggested grades:
- Grade 3: 361 (11.5%), Grade 2: 614 (19.6%), Grade 1: 927 (29.6%), Grade 0: 1236 (39.4%)

### Next Steps

1. **Phase 4**: Build `eval_v2.py` using ranx (NDCG@10 primary metric)
2. **Phase 5**: Annotate ground truth (~8 hrs, Claude Code pre-labeled + human review)
3. **Phase 6**: Corrected v2 baseline
4. **Phase 7**: Citation ablation with Fisher's randomization test
5. **Phase 8**: CI integration (2-4 week parallel period, then promote v2)
