# Lever of Archimedes Technical Integration Guide

Complete technical reference for integrating research-kb with lever_of_archimedes.

## Cross-Project Paths

| Path | Purpose |
|------|---------|
| `~/Claude/lever_of_archimedes/hooks/lib/research_kb.sh` | Hook integration (124 lines) |
| `~/Claude/lever_of_archimedes/services/research_kb_daemon/` | Daemon service directory |
| `~/Claude/lever_of_archimedes/services/health/research_kb_status.jl` | Health check script |
| `~/Claude/lever_of_archimedes/docs/brain/ideas/research_kb_full_design.md` | Full design document (47KB) |
| `~/Claude/lever_of_archimedes/knowledge/master_bibliography/AVAILABLE.md` | Master bibliography (~357 papers) |

## Daemon Service Architecture

### Socket Protocol

**Location**: `/tmp/research_kb_daemon.sock`
**Format**: Newline-delimited JSON

#### Request Actions

```json
// Search
{"action": "search", "query": "instrumental variables", "limit": 5, "context_type": "building"}

// Concept lookup
{"action": "concepts", "query": "IV"}

// Graph exploration
{"action": "graph", "concept": "instrumental variables", "hops": 2}

// Health check
{"action": "ping"}

// Shutdown
{"action": "shutdown"}
```

#### Response Format

```json
{
  "status": "ok",
  "results": [...],
  "latency_ms": 45,
  "context_type": "building"
}
```

Error response:
```json
{
  "status": "error",
  "error": "Description of the error"
}
```

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Cold start | 2-3 seconds (embedding model load) |
| Warm query p50 | <50ms |
| Warm query p95 | <100ms |
| Memory footprint | ~2GB (model + index) |

## Hook Integration

### Main Entry Point

From `hooks/lib/research_kb.sh`:

```bash
search_research_kb() {
    local query="$1"
    local context_type="${2:-balanced}"
    local limit="${3:-5}"

    # Try daemon first (fast path)
    if daemon_available; then
        daemon_search "$query" "$context_type" "$limit"
    else
        # Fallback to CLI with timeout
        timeout 5s research-kb query "$query" \
            --context "$context_type" \
            --limit "$limit" \
            --format agent
    fi
}
```

### Query Detection

The `is_causal_query()` function detects relevant queries using ~30 keywords:

**Method keywords**: instrumental variables, IV, 2SLS, difference-in-differences, DiD, regression discontinuity, RDD, synthetic control, propensity score, matching, LATE, CATE, ATE, doubly robust, double machine learning, DML

**Assumption keywords**: SUTVA, ignorability, unconfoundedness, overlap, positivity, parallel trends, exclusion restriction, relevance assumption, monotonicity

**Problem keywords**: endogeneity, confounding, selection bias, omitted variable, measurement error, simultaneity

### Context Type Selection

```bash
get_context_type() {
    local task="$1"

    case "$task" in
        "coding"|"building"|"implementing")
            echo "building"  # 20% FTS, 80% vector
            ;;
        "reviewing"|"auditing"|"checking")
            echo "auditing"  # 50% FTS, 50% vector
            ;;
        *)
            echo "balanced"  # 30% FTS, 70% vector
            ;;
    esac
}
```

## Weight Formulas

### Base Weights

```
score = fts_weight × fts_score + vector_weight × vector_score
```

| Context | fts_weight | vector_weight |
|---------|------------|---------------|
| building | 0.20 | 0.80 |
| auditing | 0.50 | 0.50 |
| balanced | 0.30 | 0.70 |

### Graph Enhancement

When graph signals are enabled:

```
graph_score = concept_overlap × 0.6 + path_proximity × 0.4
final_score = (1 - graph_weight) × base_score + graph_weight × graph_score
```

Where `graph_weight = 0.20` (20% contribution, normalized with base scores).

## Bibliography Architecture

### Master Bibliography (lever_of_archimedes)

Location: `knowledge/master_bibliography/AVAILABLE.md`

Structure:
```markdown
## Foundational Methods
- [ ] Angrist & Imbens (1994) - Instrumental Variables
- [x] Rosenbaum & Rubin (1983) - Propensity Score  # [x] = ingested

## Modern Methods
- [ ] Chernozhukov et al. (2018) - Double ML
...
```

### research-kb BIBLIOGRAPHY.md

Subset of ingested papers with local status:

```markdown
| Paper | Status | Chunks | Concepts |
|-------|--------|--------|----------|
| Rosenbaum & Rubin (1983) | Complete | 45 | 12 |
...
```

### Synchronization

The master bibliography is authoritative. To ingest new papers:

1. Add PDF to `fixtures/papers/` or `fixtures/textbooks/`
2. Run `python scripts/ingest_corpus.py`
3. Mark as `[x]` in master AVAILABLE.md
4. Update research-kb BIBLIOGRAPHY.md with status

## Health Check Integration

### research_kb_status.jl

```julia
function check_research_kb_health()
    checks = Dict{String, HealthStatus}()

    # Database connectivity
    checks["database"] = check_db_connection()

    # Embedding server
    checks["embeddings"] = check_embedding_server()

    # Daemon responsiveness
    checks["daemon"] = check_daemon_ping()

    # Query latency
    checks["latency"] = check_query_latency()

    return HealthReport(checks)
end
```

### Morning Report Integration

Health results appear in lever_of_archimedes morning reports:

```
## Research KB Status
- Database: healthy (276 sources, 142K chunks)
- Embeddings: healthy (BGE-large-en-v1.5)
- Daemon: healthy (p50: 42ms)
- Last ingestion: 2025-12-15
```

## RAG Context Flow

1. **Query arrives** via hook or daemon
2. **Context type** determined from task context
3. **Hybrid search** executes with appropriate weights
4. **Graph signals** enhance if concepts match query
5. **Results formatted** for RAG consumption (agent format)
6. **Context injected** into AI assistant prompt

### Agent Format

```json
{
  "provenance": "research-kb",
  "context_type": "building",
  "results": [
    {
      "content": "...",
      "source": "Angrist & Imbens (1994)",
      "relevance": 0.92,
      "concepts": ["instrumental variables", "LATE"]
    }
  ],
  "meta": {
    "query_time_ms": 45,
    "graph_enhanced": true
  }
}
```

## Troubleshooting

### Daemon Not Responding

```bash
# Check if socket exists
ls -la /tmp/research_kb_daemon.sock

# Check daemon process
pgrep -f research_kb_daemon

# Start daemon manually
python -m research_kb.daemon &
```

### Slow Queries

1. Check embedding server: `curl localhost:8080/health`
2. Check database: `research-kb stats`
3. Check index: PostgreSQL EXPLAIN ANALYZE on search queries

### Integration Test

```bash
# Via CLI
research-kb query "instrumental variables" --format agent

# Via daemon (if running)
echo '{"action": "ping"}' | nc -U /tmp/research_kb_daemon.sock
```

## Related Documentation

- [INTEGRATION.md](../INTEGRATION.md) - Architectural overview
- [SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) - Internal architecture
- [CLAUDE.md](../../CLAUDE.md) - CLI reference
