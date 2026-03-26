---
paths:
  - "packages/daemon/**"
  - "packages/mcp-server/**"
  - "packages/client/**"
  - "packages/api/**"
---

# Integration Services

research-kb exposes three integration surfaces:
- **Daemon service**: Unix socket at `/tmp/research_kb_daemon_${USER}.sock` (JSON-RPC 2.0)
- **REST API**: FastAPI at `http://localhost:8000` (Swagger at `/docs`)
- **MCP server**: Claude Code integration via Model Context Protocol

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for full details.

## Daemon Service

Low-latency query service via Unix domain socket (~200ms vector search, ~20ms health).

**Starting:**
```bash
research-kb-daemon                                    # Direct start
systemctl --user start research-kb-daemon             # Systemd
systemctl --user enable research-kb-daemon            # Auto-start on login
```

**Protocol:** JSON-RPC 2.0

**Methods:**
| Method | Description |
|--------|-------------|
| `search` | Hybrid search with optional graph/citation boosting |
| `fast_search` | Vector-only search (~200ms) |
| `graph_path` | Path between concepts (KuzuDB accelerated) |
| `citation_info` | Citation authority for sources |
| `health` | System health check (database, embed server, Kuzu, uptime) |
| `stats` | Database statistics |

**Example:**
```bash
echo '{"jsonrpc":"2.0","method":"health","id":1}' | nc -U /tmp/research_kb_daemon_$USER.sock
```

**Pre-warming:**
- KuzuDB pre-warms on startup to avoid 60s cold-start latency
- Warming runs in background; `health` and `fast_search` work immediately
- Typical warming: 5-15s (depends on page cache)
- Skip with `--no-warm` for testing
- Monitor via `health` endpoint (`kuzu_warmup` field) or Prometheus gauge

**Installation:**
```bash
./scripts/install_daemon.sh install
```

## MCP Server (Claude Code Integration)

The `mcp-server` package exposes research-kb to Claude Code via MCP protocol.

**Available Tools (22 total):**
| Tool | Description |
|------|-------------|
| `research_kb_search` | Hybrid search (FTS + vector + graph + citation), optional HyDE |
| `research_kb_fast_search` | Fast vector-only search (~200ms) |
| `research_kb_list_sources` | List sources (papers, textbooks) |
| `research_kb_get_source` | Get source details and chunks |
| `research_kb_get_source_citations` | Get citations for a source |
| `research_kb_get_citing_sources` | Find sources citing this one |
| `research_kb_get_cited_sources` | Find sources this one cites |
| `research_kb_citation_network` | Bidirectional citation network |
| `research_kb_biblio_coupling` | Similar sources by shared references |
| `research_kb_list_concepts` | List/search concepts |
| `research_kb_get_concept` | Concept with relationships |
| `research_kb_chunk_concepts` | Concepts linked to a chunk |
| `research_kb_find_similar_concepts` | Semantically similar concepts |
| `research_kb_cross_domain_concepts` | Cross-domain concept matching |
| `research_kb_graph_neighborhood` | Concept neighborhood exploration |
| `research_kb_graph_path` | Path between concepts (KuzuDB-accelerated) |
| `research_kb_list_domains` | Available knowledge domains |
| `research_kb_audit_assumptions` | Assumption audit with gap reporting (North Star) |
| `research_kb_explain_connection` | Graph path + evidence + LLM synthesis |
| `research_kb_literature_review` | Structured literature review with synthesis |
| `research_kb_stats` | Database statistics |
| `research_kb_health` | Health check (includes KuzuDB status) |

**Installation in Claude Code:**
```bash
# Recommended: user-global scope (available to all ~/Claude/* projects)
claude mcp add -s user research-kb \
  -- /path/to/research-kb/.venv/bin/research-kb-mcp
```

Writes to `~/.claude.json` top-level `mcpServers`.

**Config hierarchy** (highest priority first):
1. `.mcp.json` (project scope) — avoid unless project-specific overrides needed
2. `~/.claude.json` `mcpServers` (user scope) — **recommended for research-kb**

**Note:** Do NOT put MCP configs in `~/.claude/settings.local.json` — Claude Code does not read MCP servers from there.
