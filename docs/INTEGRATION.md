# External Integration Guide

This document describes how to integrate research-kb into your own tools, scripts, and AI workflows.

## Integration Surfaces

research-kb exposes three integration points:

1. **Daemon service** — Low-latency Unix socket (JSON-RPC 2.0)
2. **REST API** — FastAPI with Swagger docs
3. **MCP server** — Model Context Protocol for Claude Code / AI assistants

## 1. Daemon Service

A Unix socket daemon provides sub-100ms queries after warmup.

- **Socket**: `/tmp/research_kb_daemon_${USER}.sock`
- **Startup**: 2-3 seconds (loads embedding model)
- **Query latency**: <100ms after warmup
- **Protocol**: JSON-RPC 2.0

### Methods

| Method | Description |
|--------|-------------|
| `search` | Hybrid search (FTS + vector + graph + citation) |
| `fast_search` | Vector-only search (~200ms) |
| `graph_path` | Find path between concepts |
| `citation_info` | Get citation authority for sources |
| `health` | System health check |
| `stats` | Database statistics |

### Example Requests

```bash
# Health check
echo '{"jsonrpc":"2.0","method":"health","id":1}' | nc -U /tmp/research_kb_daemon_$USER.sock

# Search
echo '{"jsonrpc":"2.0","method":"search","params":{"query":"instrumental variables","limit":3},"id":1}' | \
  nc -U /tmp/research_kb_daemon_$USER.sock

# Graph path between concepts
echo '{"jsonrpc":"2.0","method":"graph_path","params":{"source":"IV","target":"unconfoundedness"},"id":1}' | \
  nc -U /tmp/research_kb_daemon_$USER.sock
```

### Python Client SDK

```python
from research_kb_client import ResearchKBClient

async with ResearchKBClient() as client:
    results = await client.search("double machine learning", limit=5)
    health = await client.health()
```

## 2. REST API

```bash
# Start the API server
uvicorn research_kb_api.main:app --host 0.0.0.0 --port 8000

# Search
curl http://localhost:8000/search?q=instrumental+variables&limit=5

# Swagger docs
open http://localhost:8000/docs
```

## 3. MCP Server (Claude Code)

Add to your Claude Code MCP configuration (`~/.config/claude-code/config.json`):

```json
{
  "mcpServers": {
    "research-kb": {
      "command": "research-kb-mcp",
      "args": []
    }
  }
}
```

This exposes 19 tools to Claude Code — see [CLAUDE.md](../CLAUDE.md#mcp-server-claude-code-integration) for the full tool list.

## Context Types

When querying, context type adjusts search weights:

| Context | FTS | Vector | Use Case |
|---------|-----|--------|----------|
| **building** | 20% | 80% | Writing code, broad understanding |
| **auditing** | 50% | 50% | Precise fact-checking |
| **balanced** | 30% | 70% | Default, general research |

Graph and citation signals add weight when enabled (normalized).

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `nc: unix connect failed` | Daemon not running | `systemctl --user start research-kb-daemon` or `research-kb-daemon &` |
| Empty search results | Database not seeded | Run `research-kb stats`, then `python scripts/setup_demo.py` |
| Slow responses (>1s) | Embedding model loading | First query after start is slow; subsequent <100ms |
| `Permission denied` on socket | Socket permissions | Check `/tmp/research_kb_daemon_$USER.sock` ownership |

## Further Reading

- [System Design](SYSTEM_DESIGN.md) — Internal architecture
- [SLO](SLO.md) — Service level objectives
