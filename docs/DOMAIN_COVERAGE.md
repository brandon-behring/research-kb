# Domain Coverage Gap Report

> Cross-references research-kb corpus domains against interview preparation volumes and identifies strategic gaps.
>
> Last updated: 2026-02-25 (Phase N: Domain Gap Expansion)

---

## Coverage Matrix

Maps interview preparation topic areas to research-kb domains, with source counts and qualitative coverage assessments.

| Interview Topic | KB Domain | Sources | Coverage | Notes |
|----------------|-----------|---------|----------|-------|
| Causal Inference | `causal_inference` | 299 | Excellent | Core strength. Pearl, Angrist/Pischke, Imbens/Rubin + papers |
| Time Series | `time_series` | 47 | Excellent | Hamilton, Hyndman, Box-Jenkins + domain papers |
| Mathematics | `mathematics` | 27 | Good | Linear algebra, calculus, physics, optimization |
| Interview Prep | `interview_prep` | 22 | Good | Multi-domain interview preparation |
| GenAI / LLM | `rag_llm` | 17 | Good | RAG pipelines, prompt engineering, evaluation |
| Software Engineering | `software_engineering` | 15 | Good | Design patterns, testing, architecture, DevOps |
| Statistics | `statistics` | 10 | Moderate | Theory and methods (Wasserman, Efron, Bayesian) |
| ML Foundations | `machine_learning` | 10 | Moderate | Core algorithms and theory |
| Deep Learning | `deep_learning` | 9 | Moderate | Neural nets, transformers, optimization |
| Python / Algorithms | `algorithms` | 9 | Moderate | Roughgarden 1-3, Grokking, Wengrow, Kochenderfer |
| Functional Programming | `functional_programming` | 8 | Moderate | Haskell, Scala, FP patterns |
| Product Analytics | `data_science` | 4 | Partial | Data analysis, A/B testing overlap |
| MLOps / ML Engineering | `ml_engineering` | 3 | Partial | ML systems, production ML |
| Fitness | `fitness` | 2 | Partial | Strength training |
| Forecasting | `forecasting` | 1 | **Thin** | Overlaps time_series but distinct methods |
| Economics | `economics` | 1 | **Thin** | Quantitative economics |
| Finance / Credit Risk | `finance` | 1 | **Thin** | Quantitative finance |
| SQL | `sql` | 0 | **Registered, empty** | Domain + prompt config ready; no local PDFs yet |
| Recommender Systems | `recommender_systems` | 0 | **Registered, empty** | Domain + prompt config ready; no local PDFs yet |
| Ads / AdTech | `adtech` | 0 | **Registered, empty** | Domain + prompt config ready; no local PDFs yet |

**Note**: Source counts changed significantly in Phase N due to sidecar audit + DB retagging. Many books previously defaulting to `causal_inference` are now correctly assigned to `mathematics`, `algorithms`, `functional_programming`, `software_engineering`, etc.

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

### Empty Domains (registered, need sources)

| Domain | Priority | Source Strategy |
|--------|----------|-----------------|
| `sql` | High | Acquire SQL textbooks (window functions, query optimization, database internals) |
| `recommender_systems` | Medium | Acquire recsys textbooks (Aggarwal, Ricci) or papers |
| `adtech` | Low | Acquire adtech/auction theory sources; overlap with causal_inference for incrementality |

### Thin Domains (need expansion)

| Domain | Sources | Target | Strategy |
|--------|---------|--------|----------|
| `forecasting` | 1 | 5+ | More forecasting-specific textbooks |
| `economics` | 1 | 3+ | Macroeconomics, microeconomics textbooks |
| `finance` | 1 | 5+ | Re-check Manning finance titles; some may have been absorbed elsewhere |

---

## How to Add a Missing Domain

See [`docs/tutorial_new_domain.md`](tutorial_new_domain.md) for the full 5-step process:

1. Register domain in PostgreSQL (`scripts/add_missing_domains.py`)
2. Configure extraction prompts (`domain_prompts.py`)
3. Ingest source documents
4. Extract concepts + sync KuzuDB
5. Verify with search + build golden eval queries
