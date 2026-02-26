# Domain Coverage Gap Report

> Cross-references research-kb corpus domains against interview preparation volumes and identifies strategic gaps.
>
> Last updated: 2026-02-26 (Phase X: Domain Identity Fix)

---

## Coverage Matrix

Maps interview preparation topic areas to research-kb domains, with source counts and qualitative coverage assessments.

| Interview Topic | KB Domain | Sources | Coverage | Notes |
|----------------|-----------|---------|----------|-------|
| Causal Inference | `causal_inference` | 89 | Good | Pearl, Imbens/Rubin, DML + papers (econometrics split out) |
| GenAI / LLM | `rag_llm` | 76 | Excellent | RAG pipelines, prompt engineering, evaluation, KG |
| Time Series | `time_series` | 48 | Excellent | Hamilton, Hyndman, Box-Jenkins + domain papers |
| Econometrics | `econometrics` | 35 | Good | Angrist/Pischke, Wooldridge, panel data, IV methods |
| Deep Learning | `deep_learning` | 35 | Good | Neural nets, transformers, optimization |
| Software Engineering | `software_engineering` | 30 | Good | Design patterns, testing, architecture, DevOps |
| Mathematics | `mathematics` | 28 | Good | Linear algebra, calculus, physics, optimization |
| Interview Prep | `interview_prep` | 23 | Good | Multi-domain interview preparation |
| Finance / Credit Risk | `finance` | 23 | Good | Quantitative finance, CFA, risk management |
| Statistics | `statistics` | 18 | Good | Theory and methods (Wasserman, Efron, Bayesian) |
| MLOps / ML Engineering | `ml_engineering` | 17 | Good | ML systems, production ML |
| ML Foundations | `machine_learning` | 14 | Moderate | Core algorithms and theory |
| Python / Algorithms | `algorithms` | 12 | Moderate | Roughgarden 1-3, Grokking, Wengrow, Kochenderfer |
| Product Analytics | `data_science` | 12 | Moderate | Data analysis, A/B testing overlap |
| Portfolio Management | `portfolio_management` | 11 | Moderate | MPT, CAPM, factor models, risk management |
| Functional Programming | `functional_programming` | 8 | Moderate | Haskell, Scala, FP patterns |
| Forecasting | `forecasting` | 5 | Partial | Overlaps time_series but distinct methods |
| Recommender Systems | `recommender_systems` | 3 | Partial | Collaborative filtering, recsys (Phase T arXiv) |
| SQL | `sql` | 2 | Partial | SQL databases, query optimization (Phase T arXiv) |
| Ads / AdTech | `adtech` | 2 | Partial | Advertising technology, auction mechanisms (Phase T arXiv) |
| Fitness | `fitness` | 2 | Partial | Strength training |
| Economics | `economics` | 2 | Partial | Quantitative economics |

**Note**: Source counts updated in Phase X after fixing `sources.domain_id` to match canonical metadata domains (209 sources retagged). Key changes: `causal_inference` 272 -> 89 (econometrics, rag_llm, deep_learning, etc. split out), `finance` 28 -> 23 (portfolio_management split out).

---

## Phase N Changes (2026-02-25)

### New Domains Registered
- `sql` — SQL & Databases (domain + extraction prompt config)
- `recommender_systems` — Recommender Systems (domain + extraction prompt config)
- `adtech` — Ads & AdTech (domain + extraction prompt config)

### New Extraction Prompt Configs
- `algorithms` — Algorithm design, complexity analysis, data structures
- `forecasting` — Time-series forecasting methods, evaluation, uncertainty

### Sidecar Audit
- 97 out of 98 migrated/ sidecar JSONs corrected (legacy labels → valid domains)
- Legacy labels eliminated: `other`, `programming`, `ml_stats`, `math`, `nlp`, `causal`

### DB Retagging
- 88 sources retagged from `causal_inference` fallback to correct domains
- 27,965 chunks domain_id propagated via `sync_chunk_domains.py`

---

## Remaining Gaps

### Thin Domains (need expansion)

| Domain | Sources | Target | Strategy |
|--------|---------|--------|----------|
| `economics` | 2 | 5+ | Macroeconomics, microeconomics textbooks |
| `sql` | 2 | 5+ | Acquire SQL textbooks (window functions, query optimization, database internals) |
| `adtech` | 2 | 5+ | Acquire adtech/auction theory sources; overlap with causal_inference for incrementality |
| `fitness` | 2 | 5+ | Exercise science, nutrition, programming textbooks |
| `recommender_systems` | 3 | 5+ | Acquire recsys textbooks (Aggarwal, Ricci) or papers |
| `forecasting` | 5 | 5+ | At threshold; consider demand forecasting, energy forecasting sources |

---

## How to Add a Missing Domain

See [`docs/tutorial_new_domain.md`](tutorial_new_domain.md) for the full 5-step process:

1. Register domain in PostgreSQL (`scripts/add_missing_domains.py`)
2. Configure extraction prompts (`domain_prompts.py`)
3. Ingest source documents
4. Extract concepts + sync KuzuDB
5. Verify with search + build golden eval queries
