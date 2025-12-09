# Research-KB Documentation

## Current Status

- **Phase**: Phase 3 complete ✅, Phase 4 (Visualization) ready to start
- **Status**: [→ Full Status (auto-generated)](status/CURRENT_STATUS.md)

---

## Quick Navigation

| I want to... | Go to... |
|--------------|----------|
| Understand the architecture | [System Design](SYSTEM_DESIGN.md) |
| See current status | [Current Status](status/CURRENT_STATUS.md) |
| Work on Phase 3 | [Phase 3 Plan](phases/phase3/ENHANCED_RETRIEVAL.md) |
| Run the CLI | [CLAUDE.md](../CLAUDE.md#cli-usage) |
| Set up locally | [Local Development](guides/LOCAL_DEVELOPMENT.md) |
| Update status docs | `python scripts/generate_status.py` |

---

## Phase Overview

| Phase | Status | Key Deliverables | Doc |
|-------|--------|------------------|-----|
| 1. Foundation | ✅ Complete | PostgreSQL, contracts, storage | [→](phases/phase1/FOUNDATION.md) |
| 1.5 PDF Ingestion | ✅ Complete | Dispatcher, citations, embeddings | [→](phases/phase1.5/PDF_INGESTION.md) |
| 2. Knowledge Graph | ✅ Complete | Concept extraction, graph queries | [→](phases/phase2/KNOWLEDGE_GRAPH.md) |
| 3. Enhanced Retrieval | 📋 Ready | Re-ranking, query expansion | [→](phases/phase3/ENHANCED_RETRIEVAL.md) |
| 4. Production | 📋 Planned | FastAPI, auth, deployment | [→](phases/phase4/PRODUCTION.md) |

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

| Metric | Value |
|--------|-------|
| Sources | 137 (65 textbooks + 72 papers) |
| Chunks | 33,973 (100% with embeddings) |
| Concepts | 19,458 |
| Relationships | 20,423 |
| Tests | 502 functions |
| 2-hop graph query | 2.11ms (target: <100ms) ✅ |

---

## External References

- **Full System Design**: `$HOME/Claude/lever_of_archimedes/research-kb-system-design.md`
- **GitHub Repository**: https://github.com/brandonmbehring-dev/research-kb
