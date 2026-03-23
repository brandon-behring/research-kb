# Research-KB Documentation

## Current Status

- **Phase**: All core phases ✅ Complete | Catalog ingestion ✅ Complete (2026-03-23)
- **Status**: [→ Full Status (auto-generated)](status/CURRENT_STATUS.md)
- **Corpus**: 997 sources, 857K chunks, 100% embedded, 41,852 citation edges (updated 2026-03-23)
- **Domains**: 35 tagged across causal_inference, rag_llm, time_series, finance, mathematics, physics, algebra, optimization, and 27 more
- **Search**: 3-way hybrid (FTS + vector + citation). MRR 0.771, Hit Rate 94.4% on 107 eval test cases
- **MCP Server**: 22 tools available (9 with JSON output_format)
- **KuzuDB**: 310K concepts, 744K relationships (stale chunk IDs — graph search disabled, re-extraction deferred)
- **ProactiveContext**: Integrated via `fast_search` (~200ms latency)

---

## Quick Navigation

| I want to... | Go to... |
|--------------|----------|
| Understand the architecture | [System Design](SYSTEM_DESIGN.md) |
| See current status | [Current Status](status/CURRENT_STATUS.md) |
| Understand integration surfaces | [Integration Guide](INTEGRATION.md) |
| Run the CLI | [CLAUDE.md](../CLAUDE.md#cli-usage) |
| Set up locally | [Local Development](guides/LOCAL_DEVELOPMENT.md) |
| See domain coverage gaps | [Domain Coverage](DOMAIN_COVERAGE.md) |
| See what to buy/download next | [Acquisition List](ACQUISITION_LIST.md) |
| Find a script's usage | [Script Utilities](scripts/UTILITIES.md) |
| Update status docs | `python scripts/generate_status.py` |

---

## Phase Overview

| Phase | Status | Key Deliverables | Doc |
|-------|--------|------------------|-----|
| 1. Foundation | ✅ Complete | PostgreSQL, contracts, storage | [→](phases/phase1/FOUNDATION.md) |
| 1.5 PDF Ingestion | ✅ Complete | Dispatcher, citations, embeddings | [→](phases/phase1.5/PDF_INGESTION.md) |
| 2. Knowledge Graph | ✅ Complete | Concept extraction, graph queries, KuzuDB | [→](phases/phase2/KNOWLEDGE_GRAPH.md) |
| 3. Enhanced Retrieval | ✅ Complete | Re-ranking, query expansion, citation authority | [→](phases/phase3/ENHANCED_RETRIEVAL.md) |
| 4. Production | ✅ Complete | FastAPI, dashboard, metrics, daemon | [→](phases/phase4/PRODUCTION.md) |
| 4.3 ProactiveContext | ✅ Complete | Context injection hook integration | [→](status/REMEDIATION_LOG.md#phase-43-proactivecontext-integration--complete--2026-02-06) |
| D. Observability | ✅ Complete | Prometheus metrics, Grafana dashboard, KuzuDB benchmarks | [→](../CLAUDE.md#daemon-service) |
| E. RAG/LLM Extraction | ✅ Complete | 23K concepts from 13 RAG/LLM textbooks (Haiku 4.5) | [→](status/REMEDIATION_LOG.md) |
| F. Cross-Repo Integration | ✅ Complete | Lever health monitoring, interview readiness scorer | [→](INTEGRATION.md) |
| G. Repository Hygiene | ✅ Complete | Pytest consolidation, test markers, pre-commit, scripts archive | [→](../CLAUDE.md) |
| H. Multi-Domain Extraction | ✅ Complete | 9 new domain prompt configs (14 total), 162 tests | [→](../ROADMAP.md) |
| I. CI Hardening | ✅ Complete | pytest-cov in PR checks, doc freshness gate, ROADMAP rewrite | [→](../ROADMAP.md) |
| J. Eval Expansion | ✅ Complete | 55 retrieval test cases, 29 assumption methods, per-domain eval | [→](../ROADMAP.md) |
| K. Doc Consolidation | ✅ Complete | CLAUDE.md, README, MEMORY.md refresh, zero audit warnings | [→](../ROADMAP.md) |
| M. Code Quality Hardening | ✅ Complete | Coverage gate, MCP HyDE+fast_search, daemon timeout, dashboard tests | [→](../ROADMAP.md) |
| P. Audit Remediation | ✅ Complete | Integration fixtures, doc drift, CI schedule alignment | [→](../ROADMAP.md) |
| Q. Type Safety | ✅ Complete | PEP 561 (12/12), mypy 74→18 baseline, strictness for 4 packages | [→](../ROADMAP.md) |
| R. CI Quality Gates | ✅ Complete | Coverage config, threshold 40%→60%, e2e in CI, black 26.1.0 | [→](../ROADMAP.md) |
| S. Coverage Hardening | ✅ Complete | 85 new unit tests, coverage gate 60%→66%, 4 core modules covered | [→](../ROADMAP.md) |
| T. Domain Acquisition | ✅ Complete | 3 KG books ingested, CFA retag, 7 papers for 3 empty domains | [→](../ROADMAP.md) |
| U. Concept Extraction | ✅ Complete | Concepts for sql/recommender_systems/adtech/rag_llm, 9 eval cases activated | [→](../ROADMAP.md) |
| V. Doc Trust Alignment | ✅ Complete | README 19→22 domains, domain table sync, TEST_COVERAGE refresh | [→](../ROADMAP.md) |
| W. CLI Citations + Synonym Fix | ✅ Complete | 16 CLI tests, synonym normalization, coverage gate raise | [→](../ROADMAP.md) |
| X. Data Accuracy | ✅ Complete | 209 domain_id fixes, generate_status.py canonical domain_id | [→](../ROADMAP.md) |
| Y. Test Fortification | ✅ Complete | Dashboard +65 tests, client +32, mypy 0, coverage 70% | [→](../ROADMAP.md) |
| Z. JSON MCP Output | ✅ Complete | output_format on 7 tools, JSON formatters, STRATEGIC_ASSESSMENT | [→](../ROADMAP.md) |
| AB. Scoped Assumption Audit | ✅ Complete | domain + scope params on audit_assumptions | [→](../ROADMAP.md) |
| AC. Explain Connection | ✅ Complete | Synthesis: graph path + evidence + LLM, MCP tool #21 | [→](../ROADMAP.md) |
| AD. Codex Audit Cleanup | ✅ Complete | CI cadence labels, stale refs, MRR threshold, README tools | [→](../ROADMAP.md) |
| AE. Interview Prep Fix | ✅ Complete | 7 synonym groups, 15 eval cases, 100% Hit@10, MRR 0.636 | [→](../ROADMAP.md) |
| AF. Concept Deduplication | ✅ Complete | 2,370 pairs merged, 310K concepts, zero eval regression | [→](../ROADMAP.md) |
| AG. Doc Trust Alignment | ✅ Complete | 11 stale claims fixed, --gate-domains for eval CI | [→](../ROADMAP.md) |
| AH. Semantic Chunking | ✅ Complete | Structure-driven chunker, heading-aware PDF splitting | [→](../ROADMAP.md) |
| AI. Literature Review | ✅ Complete | Graph+search+LLM review generation, MCP tool #22, operational scripts | [→](../ROADMAP.md) |
| AJ. Docling Migration | ✅ Complete | LaTeX-preserving PDF extraction (IBM Docling/Granite-258M) | [→](../ROADMAP.md) |
| Sprint 1 | ✅ Complete | MCP output_format on get_source + cross_domain_concepts | [→](STRATEGIC_ASSESSMENT.md) |
| RAG Optimization | ✅ Complete | 100% embeddings, 41K citation edges, 3-way search defaults | [→](STRATEGIC_ASSESSMENT.md) |
| Catalog Ingestion | ✅ Complete | 552 books ingested (Tier 1+2), 12 new domains, 107 eval cases | [→](STRATEGIC_ASSESSMENT.md) |

---

## Directory Structure

```
docs/
├── INDEX.md                    # 🗺️ YOU ARE HERE
├── SYSTEM_DESIGN.md            # Architecture summary
│
├── phases/                     # Phase documentation
│   ├── phase1/FOUNDATION.md
│   ├── phase1.5/PDF_INGESTION.md
│   ├── phase2/KNOWLEDGE_GRAPH.md
│   ├── phase3/ENHANCED_RETRIEVAL.md
│   └── phase4/PRODUCTION.md
│
├── status/                     # Current state
│   ├── CURRENT_STATUS.md
│   ├── VALIDATION_TRACKER.md
│   └── MIGRATION_GRAPH_DEFAULT.md
│
├── design/                     # Architecture research
│   ├── latency_analysis.md        # Graph signal latency (pre/post KuzuDB)
│   ├── rrf_validation.md          # Weighted sum vs. RRF empirical comparison
│   ├── search_quality_analysis.md # Search quality and scoring analysis
│   └── phase3_research_notes.md
│
├── scripts/                    # Script documentation
│   └── UTILITIES.md               # Categorized script index with usage examples
│
├── ACQUISITION_LIST.md         # Definitive buy/download list (cross-referenced against DB)
├── owned_inventory.json        # Ground truth owned inventory (generated)
├── DOMAIN_COVERAGE.md          # Domain gap analysis vs interview prep
│
├── guides/                     # How-to guides
│   └── LOCAL_DEVELOPMENT.md
│
└── archive/                    # Historical records
    ├── WEEK1_DELIVERABLES.md
    ├── WEEK_2_DELIVERABLES.md
    ├── 2025-12-16-codex-parallel-critique.md
    ├── 2025-12-16-research-kb-critique.md
    ├── gemini_audit_report_2026-01-08.md
    ├── phase1_5_completion_report.md
    ├── phase2_step9_completion_report.md
    └── quality-reports-2025-12-16/
```

---

## Key Metrics

See [CURRENT_STATUS.md](status/CURRENT_STATUS.md) for live metrics (auto-generated from database).

Run `python scripts/generate_status.py` to refresh metrics.

---

## External References

- **GitHub Repository**: https://github.com/brandon-behring/research-kb
