# Fix Research-KB Integration Plan

**Created**: 2025-12-29
**Status**: Ready for implementation
**Priority**: Fix broken integration (user-selected)

---

## Problem Summary

Skeptical audit revealed research-kb is **mostly functional but not integrated**:

| Component | Status | Issue |
|-----------|--------|-------|
| PostgreSQL + pgvector | ✅ Working | - |
| MCP Server | ✅ Working | Returns results |
| Daemon Socket | ✅ Working | JSON-RPC responds |
| CLI (no-rerank) | ✅ Working | 3s response |
| CLI (with rerank) | ❌ Broken | 30s timeout |
| Hook Integration | ❌ Broken | Wrong path |

---

## Root Causes

### 1. Hook Path Pointing to Stale Project

**File**: `~/.claude/settings.local.json`

```json
// CURRENT (broken):
"command": "$HOME/Claude/archimedes_lever/scripts/hooks/user_prompt_submit.sh"

// SHOULD BE:
"command": "$HOME/Claude/lever_of_archimedes/hooks/user_prompt_submit.sh"
```

- Old `archimedes_lever` hooks have **no research_kb.sh integration**
- New `lever_of_archimedes` hooks **have research_kb.sh integration** (Dec 11)

### 2. Reranker Service Not Running

- No `pgrep rerank` process found
- CLI defaults to `--rerank` enabled
- Times out after 30s waiting for socket that doesn't exist

---

## Implementation Plan

### Phase 1: Fix Hook Integration (5 min)

**Edit**: `~/.claude/settings.local.json`

```json
{
  "hooks": {
    "user-prompt-submit": {
      "command": "$HOME/Claude/lever_of_archimedes/hooks/user_prompt_submit.sh",
      "description": "RAG context injection via research-kb + ProactiveContext"
    },
    "session-start": {
      "command": "$HOME/Claude/lever_of_archimedes/hooks/session_start.sh",
      "description": "Start RAG query daemon on session start"
    }
  }
}
```

**Verification**:
```bash
# Test hook directly
echo '{"prompt": "instrumental variables assumption"}' | \
  $HOME/Claude/lever_of_archimedes/hooks/user_prompt_submit.sh

# Check log after Claude Code prompt
tail -20 /tmp/rag_hook.log
```

### Phase 2: Create Reranker Systemd Service (10 min)

**Create**: `~/.config/systemd/user/research_kb_rerank.service`

```ini
[Unit]
Description=Research-KB Reranker (BGE-reranker-v2-m3)
After=network.target

[Service]
Type=simple
WorkingDirectory=<PROJECT_ROOT>
Environment="PATH=<PROJECT_ROOT>/venv/bin:/usr/bin"
ExecStart=<PROJECT_ROOT>/venv/bin/python -m research_kb_pdf.rerank_server
Restart=on-failure
RestartSec=10s
TimeoutStartSec=120

[Install]
WantedBy=default.target
```

**Commands**:
```bash
systemctl --user daemon-reload
systemctl --user enable research_kb_rerank.service
systemctl --user start research_kb_rerank.service
systemctl --user status research_kb_rerank.service
```

**Verification**:
```bash
# Check socket exists
ls -la /tmp/research_kb_rerank.sock

# Test CLI with rerank (should complete in ~5s)
time research-kb query "IV" --limit 2
```

### Phase 3: End-to-End Verification (5 min)

1. **Hook test**: Send causal inference prompt in Claude Code, check `/tmp/rag_hook.log`
2. **CLI test**: `research-kb query "double machine learning" --limit 3`
3. **Daemon test**: `echo '{"jsonrpc":"2.0","method":"search","params":{"query":"IV"},"id":1}' | nc -U /tmp/research_kb_daemon_$USER.sock`
4. **MCP test**: Use `research_kb_search` tool in Claude Code

---

## Critical Files

| File | Action |
|------|--------|
| `~/.claude/settings.local.json` | Edit: Update hook paths |
| `~/.config/systemd/user/research_kb_rerank.service` | Create: New service |
| `~/Claude/lever_of_archimedes/hooks/user_prompt_submit.sh` | Verify: Executable, has research_kb.sh |
| `~/Claude/research-kb/packages/pdf-tools/src/research_kb_pdf/rerank_server.py` | Verify: Entry point works |

---

## Risk Mitigation

1. **Backup settings**: `cp ~/.claude/settings.local.json ~/.claude/settings.local.json.backup`
2. **GPU memory**: Reranker needs ~1GB VRAM. Monitor with `nvidia-smi` alongside embed server
3. **Fail-open**: Both hooks and reranker have fallback paths - won't block if they fail

---

## Future Work (Not This Phase)

- Expand corpus coverage (7% → higher)
- Fill method gaps (RDD, DiD, matching)
- Clean up stale `archimedes_lever` directory
