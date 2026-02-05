# Research-KB Documentation

## Current Status

- **Phase**: Core phases ✅ Complete | Phase 4.3 (ProactiveContext Integration) Planned
- **Status**: [→ Full Status (auto-generated)](status/CURRENT_STATUS.md)
- **KuzuDB**: ✅ Operational (284K concepts, 726K relationships, <300ms graph queries)
- **MCP Server**: 19 tools available
- **Domains**: 5 active (Causal Inference, Time Series, RAG & LLM, Interview Prep, Healthcare)
- **ProactiveContext**: Integrated via `fast_search` (~200ms latency)

---

## Quick Navigation

| I want to... | Go to... |
|--------------|----------|
| Understand the architecture | [System Design](SYSTEM_DESIGN.md) |
| See current status | [Current Status](status/CURRENT_STATUS.md) |
| Understand lever integration | [Integration Overview](INTEGRATION.md) |
| See technical integration details | [Lever Integration Technical](guides/LEVER_INTEGRATION_TECHNICAL.md) |
| Run the CLI | [CLAUDE.md](../CLAUDE.md#cli-usage) |
| Set up locally | [Local Development](guides/LOCAL_DEVELOPMENT.md) |
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
| 4.3 ProactiveContext | 📋 Planned | Context injection hook integration | [→](status/REMEDIATION_LOG.md#phase-43-proactivecontext-integration--planned) |

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
│   └── phase3_research_notes.md
│
├── guides/                     # How-to guides
│   ├── STEP_BY_STEP_VALIDATION_GUIDE.md
│   └── LOCAL_DEVELOPMENT.md
│
└── archive/                    # Historical records
    ├── WEEK1_DELIVERABLES.md
    └── WEEK_2_DELIVERABLES.md
```

---

## Key Metrics

See [CURRENT_STATUS.md](status/CURRENT_STATUS.md) for live metrics (auto-generated from database).

Run `python scripts/generate_status.py` to refresh metrics.

---

## External References

- **Full System Design**: `$HOME/Claude/lever_of_archimedes/research-kb-system-design.md`
- **GitHub Repository**: https://github.com/brandonmbehring-dev/research-kb
