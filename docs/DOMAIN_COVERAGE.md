# Domain Coverage Gap Report

> Cross-references research-kb corpus domains against interview preparation volumes and identifies strategic gaps.
>
> Last updated: 2026-02-24

---

## Coverage Matrix

Maps interview preparation topic areas to research-kb domains, with source counts and qualitative coverage assessments.

| Interview Topic | KB Domain | Sources | Coverage | Notes |
|----------------|-----------|---------|----------|-------|
| Causal Inference | `causal_inference` | 89 | Excellent | Core strength. Pearl, Angrist/Pischke, Imbens/Rubin + 80 papers |
| Time Series | `time_series` | 49 | Excellent | Hamilton, Hyndman, Box-Jenkins + domain papers |
| GenAI / LLM | `rag_llm` | 74 | Excellent | RAG pipelines, prompt engineering, evaluation |
| Software Engineering | `software_engineering` | 37 | Good | Design patterns, testing, architecture, DevOps |
| Deep Learning | `deep_learning` | 35 | Good | Neural nets, transformers, optimization |
| Econometrics | `econometrics` | 35 | Good | Theory and estimation methods |
| ML Foundations | `machine_learning` | 20 | Good | Core algorithms and theory |
| Statistics | `statistics` | 18 | Moderate | Theory and methods |
| MLOps / ML Engineering | `ml_engineering` | 14 | Moderate | ML systems, production ML |
| Product Analytics | `data_science` | 12 | Moderate | Data analysis, A/B testing overlap |
| Portfolio / Quant Finance | `portfolio_management` | 11 | Moderate | Portfolio theory and optimization |
| Finance / Credit Risk | `finance` | 23 | Partial | Quantitative finance; credit risk is a subset |
| Pricing / Demand | `econometrics` | 35 | Partial overlap | Pricing models partially covered by econometrics |
| Python / Algorithms | `algorithms` | 6 | **Thin** | Needs expansion: algorithm design, complexity, data structures |
| Forecasting | `forecasting` | 4 | **Thin** | Overlaps time_series but distinct methods |
| SQL | -- | 0 | **Missing** | No domain exists. No SQL-specific sources |
| Ads / AdTech | -- | 0 | **Missing** | No domain exists. Auction theory, attribution, bid optimization |
| Recommender Systems | -- | 0 | **Missing** | No domain exists. Collaborative filtering, content-based, hybrid |

---

## Gap Analysis

### Critical Gaps (no coverage)

1. **SQL** -- High-frequency interview topic with zero corpus representation. Would require:
   - Domain registration (`sql` or `databases`)
   - Sources: query optimization, window functions, database internals
   - Extraction prompt config for SQL-specific concepts

2. **Recommender Systems** -- Core ML interview topic, especially for tech companies. Would require:
   - Domain registration (`recommender_systems`)
   - Sources: collaborative filtering, matrix factorization, neural recommenders, evaluation metrics
   - High overlap potential with `machine_learning` and `deep_learning` concepts

3. **Ads / AdTech** -- Specialized but important for ad-tech roles. Would require:
   - Domain registration (`adtech`)
   - Sources: auction theory, attribution modeling, bid optimization, incrementality testing
   - Partial overlap with `causal_inference` (incrementality) and `econometrics` (demand estimation)

### Thin Coverage (needs expansion)

4. **Algorithms** (6 sources) -- Foundational CS topic under-represented. Priority additions:
   - Algorithm design and analysis textbooks (CLRS, Skiena)
   - Data structures, complexity theory, dynamic programming
   - Graph algorithms (separate from knowledge graph concepts)

5. **Forecasting** (4 sources) -- Distinct from time_series in methods and evaluation. Priority additions:
   - Forecasting-specific evaluation (MAPE, SMAPE, calibration)
   - Hierarchical forecasting, probabilistic forecasting
   - Business forecasting applications

---

## Manning Library Gap Analysis

114 of 125 Manning books have been ingested (Tier 1+2+3 complete). The 11 remaining books and their potential domain contributions:

| Potential Domain Impact | Count | Notes |
|------------------------|-------|-------|
| Could fill `algorithms` gap | 2-3 | Algorithm-focused Manning titles |
| Could fill `recommender_systems` gap | 1 | If recommendation-focused title exists |
| Incremental to existing domains | 7-8 | Marginal additions to well-covered domains |

> To identify specific titles: cross-reference `fixtures/textbooks/` against the ingestion log.

---

## Strategic Priority Ordering

For domain expansion, prioritized by interview frequency and gap severity:

| Priority | Domain | Action | Effort |
|----------|--------|--------|--------|
| 1 | `algorithms` | Expand existing (6 → 15+ sources) | Low -- domain exists, add textbooks |
| 2 | `sql` | Create new domain + ingest | Medium -- new domain registration + prompt config |
| 3 | `recommender_systems` | Create new domain + ingest | Medium -- new domain + cross-domain linking |
| 4 | `forecasting` | Expand existing (4 → 10+ sources) | Low -- domain exists, add sources |
| 5 | `adtech` | Create new domain + ingest | High -- specialized, fewer open sources |

---

## Related ROADMAP Items

The following "Future Work" items in [`ROADMAP.md`](../ROADMAP.md) relate to domain expansion:

- **New Domains** section lists biology/genomics and climate science as community suggestions
- **Adaptive chunking** would improve ingestion quality for algorithm textbooks (heavy on pseudocode)
- **Temporal reasoning** would benefit forecasting domain (method evolution tracking)
- **Automated literature review** could generate domain gap reports programmatically

---

## How to Add a Missing Domain

See [`docs/tutorial_new_domain.md`](tutorial_new_domain.md) for the full 5-step process:

1. Register domain in PostgreSQL (SQL migration)
2. Configure extraction prompts (optional, in `domain_prompts.py`)
3. Ingest source documents
4. Extract concepts + sync KuzuDB
5. Verify with search + build golden eval queries
