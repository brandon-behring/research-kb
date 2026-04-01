# IR Evaluation Frameworks for RAG Systems (2025-2026)

Research compiled 2026-04-01 for research-kb retrieval evaluation redesign.
Full research output from agent — see detailed findings below.

## Decision Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Evaluation library** | **ranx** | Best API, built-in comparison + significance tests, TREC-compatible, actively maintained |
| **Metric suite** | NDCG@10, Recall@10, MRR@10, MAP@10 | BEIR-standard primary + completeness + first-hit + all-relevant |
| **Relevance format** | TREC qrels, 4-point graded (0-3) | Industry standard, supports graded NDCG |
| **Evaluation level** | Chunk-level (primary), source-level (derived) | Chunk-level is the RAG standard |
| **Significance testing** | Paired t-test via ranx `compare()` | Standard, adequate for 50+ queries |
| **Ground truth** | Human judgments with TREC pooling | LLM judgments not recommended (Soboroff 2025) |

## What NOT to Use

- **RAGAS / DeepEval**: Target end-to-end RAG with generation, not retrieval-only
- **pytrec_eval alone**: Functional but no comparison/significance built-in
- **LLM-as-judge for ground truth**: Creates ceiling effects (Soboroff 2025)

## Key Sources

- [ranx GitHub](https://github.com/AmenRa/ranx) — 14 metrics, significance tests, TREC format
- [Soboroff 2025](https://arxiv.org/abs/2409.15133) — Don't Use LLMs for Relevance Judgments
- [Smucker et al. 2007](https://dl.acm.org/doi/10.1145/1321440.1321528) — Statistical Significance Tests for IR
- [BEIR Benchmark](https://github.com/beir-cellar/beir) — NDCG@10 as standard
- [Stanford IR Book Ch.8](https://nlp.stanford.edu/IR-book/html/htmledition/information-retrieval-system-evaluation-1.html) — 50+ queries minimum
