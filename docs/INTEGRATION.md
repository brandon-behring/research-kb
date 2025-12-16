# Integration with Lever of Archimedes

This document describes research-kb's integration with the [lever_of_archimedes](~/Claude/lever_of_archimedes) system.

## Overview

**research-kb** is a semantic search system for causal inference literature. It serves as the domain knowledge backbone for lever_of_archimedes, providing:

1. **Hybrid retrieval** - Full-text + vector + graph-boosted search
2. **Concept extraction** - Automatic identification of methods, assumptions, problems
3. **Citation network** - Paper relationships and influence tracking
4. **RAG context** - Optimized context for AI assistants

## Architecture Relationship

```
lever_of_archimedes (orchestration layer)
├── hooks/lib/research_kb.sh      # Query integration
├── services/research_kb_daemon/  # Background service
└── knowledge/master_bibliography # Source of truth for papers

research-kb (domain knowledge layer)
├── Hybrid search engine
├── Concept knowledge graph
├── Citation network
└── Embedding index (BGE-large-en-v1.5)
```

## Key Integration Points

### 1. Daemon Service

A Unix socket daemon provides low-latency queries for AI assistant workflows:

- **Socket**: `/tmp/research_kb_daemon.sock`
- **Startup**: 2-3 seconds (loads embedding model)
- **Query latency**: <100ms after warmup
- **Protocol**: Newline-delimited JSON

See [LEVER_INTEGRATION_TECHNICAL.md](guides/LEVER_INTEGRATION_TECHNICAL.md) for protocol details.

### 2. Hook Integration

The `research_kb.sh` hook in lever_of_archimedes:
- Detects causal inference queries (~30 keywords)
- Routes to daemon (fast) or CLI (fallback)
- Applies context-appropriate weighting
- Returns formatted results for RAG

### 3. Bibliography Architecture

**Two-tier structure**:
- **Master**: `lever_of_archimedes/knowledge/master_bibliography/AVAILABLE.md` (~357 papers)
- **Ingested**: `research-kb/BIBLIOGRAPHY.md` (~150 papers, subset that's been processed)

The master bibliography is the authoritative source. research-kb ingests papers from it as needed.

### 4. Health Monitoring

`services/health/research_kb_status.jl` checks:
- Database connectivity
- Embedding server status
- Daemon responsiveness
- Query latency metrics

Results appear in lever_of_archimedes morning reports.

## Context Types

When querying, context type adjusts search weights:

| Context | FTS | Vector | Use Case |
|---------|-----|--------|----------|
| **building** | 20% | 80% | Writing code, broad understanding |
| **auditing** | 50% | 50% | Precise fact-checking |
| **balanced** | 30% | 70% | Default, general research |

Graph signals add +20% when enabled (normalized).

## Quick Reference

| Task | How |
|------|-----|
| Query via CLI | `research-kb query "instrumental variables"` |
| Query via daemon | Send JSON to `/tmp/research_kb_daemon.sock` |
| Check status | `research-kb stats` |
| See ingested papers | `research-kb sources` |
| Explore concepts | `research-kb graph "IV" --hops 2` |

## Further Reading

- [Technical Integration Guide](guides/LEVER_INTEGRATION_TECHNICAL.md) - Protocol details, code examples
- [System Design](SYSTEM_DESIGN.md) - Internal architecture
- [Full Design Document](~/Claude/lever_of_archimedes/docs/brain/ideas/research_kb_full_design.md) - 47KB comprehensive design
