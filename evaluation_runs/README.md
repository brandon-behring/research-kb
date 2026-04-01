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

| Date | File | Description | MRR | Hit@K |
|------|------|-------------|-----|-------|
| 2026-03-30 | baseline_2026-03-30.json | First baseline (pre-guide-removal, regex patterns) | 0.787 | 92.5% |
| 2026-03-30 | post_guide_removal_2026-03-30.json | After removing 29 self-written guides | 0.800 | 92.5% |
| 2026-03-30 | post_fixes_2026-03-30.json | After pattern fixes + guide removal | 0.793 | 98.1% |
| 2026-03-31 | ablation_citations_off_2026-03-31.json | 2-way search (FTS+vector only) | 0.684 | 97.2% |
| 2026-03-31 | ablation_citations_on_2026-03-31.json | 3-way search (FTS+vector+citation) | 0.781 | 98.1% |
| 2026-04-01 | ablation_rerank_on_2026-04-01.json | 3-way + cross-encoder reranking | 0.667 | 91.6% |

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
