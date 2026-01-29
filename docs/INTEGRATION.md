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
├── services/research_kb/         # Background service
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

- **Socket**: `/tmp/research_kb_daemon_${USER}.sock`
- **Startup**: 2-3 seconds (loads embedding model)
- **Query latency**: <100ms after warmup
- **Protocol**: JSON-RPC 2.0

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
| Query via daemon | Send JSON to `/tmp/research_kb_daemon_${USER}.sock` |
| Check status | `research-kb stats` |
| See ingested papers | `research-kb sources` |
| Explore concepts | `research-kb graph "IV" --hops 2` |

## Verification & Troubleshooting

### Verifying Integration (E2E Check)

Run these commands to confirm the full integration is working:

```bash
# 1. Check daemon is running and responsive (JSON-RPC 2.0)
echo '{"jsonrpc":"2.0","method":"health","id":1}' | nc -U /tmp/research_kb_daemon_$USER.sock
# Expected: {"jsonrpc":"2.0","result":{"status":"ok","database":"ok",...},"id":1}

# 2. Test search via daemon (JSON-RPC 2.0)
echo '{"jsonrpc":"2.0","method":"search","params":{"query":"instrumental variables","limit":3},"id":1}' | \
  nc -U /tmp/research_kb_daemon_$USER.sock
# Expected: {"jsonrpc":"2.0","result":[...search results...],"id":1}

# 3. Verify hook settings (Claude Code)
cat ~/.claude/settings.local.json | jq '.hooks'
# Expected: Paths pointing to ~/Claude/lever_of_archimedes/hooks/

# 4. Test research_kb.sh directly
source ~/Claude/lever_of_archimedes/hooks/lib/research_kb.sh
research_kb_query "double machine learning" 3
# Expected: Formatted search results
```

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `nc: unix connect failed` | Daemon not running | `systemctl --user start research-kb-daemon` or `research-kb-daemon &` |
| Empty search results | Database not seeded | Run `research-kb stats` to check, then `python scripts/ingest_corpus.py` |
| Hook not firing | Settings path wrong | Check `~/.claude/settings.local.json` points to `lever_of_archimedes` |
| Slow responses (>1s) | Embedding model loading | First query after daemon start is slow; subsequent queries <100ms |
| `Permission denied` on socket | Socket permissions | Check `/tmp/research_kb_daemon_$USER.sock` ownership |

### Current Integration Status (2026-01-15)

| Component | Status | Notes |
|-----------|--------|-------|
| Daemon socket | ✅ Working | `/tmp/research_kb_daemon_$USER.sock` |
| Claude Code hooks | ✅ Configured | `settings.local.json` → `lever_of_archimedes/hooks/` |
| research_kb.sh | ✅ Functional | 30+ causal keyword detection |
| ResearchKBBridge | ✅ Tested | 439-line test suite, daemon/CLI fallback |
| Health monitoring | ✅ Operational | Julia health check in morning reports |

### Domains Available

| Domain | Chunks | Description |
|--------|-------:|-------------|
| causal_inference | 137,285 | Papers + textbooks on CI methods |
| time_series | 7,414 | Forecasting, ARIMA, VAR, state-space |
| interview_prep | 9,398 | ML/DS interview cards (17 volumes) |
| rag_llm | 910 | RAG and LLM best practices |

## Further Reading

- [Technical Integration Guide](guides/LEVER_INTEGRATION_TECHNICAL.md) - Protocol details, code examples
- [System Design](SYSTEM_DESIGN.md) - Internal architecture
- [Full Design Document](~/Claude/lever_of_archimedes/docs/brain/ideas/research_kb_full_design.md) - 47KB comprehensive design
