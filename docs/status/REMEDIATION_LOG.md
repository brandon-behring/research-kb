# Remediation Log: Research-KB Audit

Tracking progress on audit remediation (Jan 2026).

**Last Updated**: 2026-01-15

---

## Phase 0: Immediate Fixes (✅ Complete)

- [x] **Socket Path Unification**:
    - Updated `packages/common/src/research_kb_common/config.py` to default to `/tmp/research_kb_daemon_${USER}.sock`.
    - Resolved "split brain" between daemon configuration and `lever_of_archimedes` hooks.
- [x] **Documentation Sync**:
    - Updated `docs/INTEGRATION.md` to reflect the user-specific socket path.
    - Updated `docs/guides/LEVER_INTEGRATION_TECHNICAL.md`.

## Phase 1.5: MCP Reliability (✅ Complete)

- [x] **Logging Fix**:
    - Identified root cause of MCP failures: logs leaking to `stdout`.
    - Updated `packages/common/src/research_kb_common/logging_config.py` to use `sys.stderr`.
    - Patched `packages/mcp-server/src/research_kb_mcp/server.py` to initialize logging *before* `FastMCP` import.
- [x] **E2E Test Infrastructure**:
    - Created `packages/mcp-server/tests/e2e/test_server_lifecycle.py`.
    - Verified server starts, connects to DB, and responds to JSON-RPC `initialize` handshake.
- [x] **Extended E2E Coverage**:
    - Created `packages/mcp-server/tests/e2e/test_mcp_tools_e2e.py` (22 tests).
    - Full coverage: search, sources, concepts, graph, citations, cross-tool integration.
- [x] **Client SDK**:
    - Created `packages/client/` with `DaemonClient` (JSON-RPC 2.0).
    - Pydantic models for responses, CLI fallback mechanism.
    - 16 tests covering unit, mock, and integration.

## Phase 2: Eval Expansion (✅ Complete)

**Goal**: Expand retrieval test cases from 14 → 30+ with `expected_concepts` defined.

- [x] Generate golden dataset candidates using LLM-assisted script (24 candidates)
- [x] Curate candidates against actual corpus coverage (kept 9 causal inference relevant)
- [x] Add manual test cases for methodological gaps (RDD, IPW, 2SLS, etc.)
- [x] Add `expected_concepts` to all test cases (now 19/30)
- [x] Run evaluation: **100% hit rate** at K=5, concept recall 75.9%

**Final metrics**:
- Test cases: 30 (was 14)
- Hit Rate@K: 100% (target: ≥90%)
- Concept Recall: 75.9% (target: ≥70%)
- MRR: 0.838
- NDCG@5: 0.878

## Phase 3: Latency Investigation (✅ Complete)

**Goal**: Profile full-signal query latency and document findings.

**Pre-KuzuDB Findings** (see `docs/design/latency_analysis.md`):

| Configuration | Latency |
|--------------|---------|
| Baseline (FTS + vector) | 10.8s |
| + Citations only | 10.4s |
| + Graph only | **96.6s** |
| Full (all signals) | **94.5s** |

**Root cause**: O(N×M) recursive CTE queries in `compute_weighted_graph_score()`.

**Mitigations implemented**:
1. ✅ Timeout fallback: `GRAPH_SCORE_TIMEOUT = 2.0s` (`graph_queries.py:50`)
2. ✅ KuzuDB migration: Primary graph engine (~150ms batch scoring, `graph_queries.py:984`)
3. PostgreSQL CTEs retained as fallback only, capped at 2s

**Remaining**: Run Phase D production benchmarks to validate latency improvements.

## Phase 4.3: ProactiveContext Integration (📋 Planned)

**Exit Criteria** (spec only, implementation separate):

| Criterion | Target |
|-----------|--------|
| Latency budget | <500ms for context injection hook |
| Coverage | 80% of causal inference prompts receive enrichment |
| Fallback | Graceful degradation when daemon unavailable (return empty, don't block) |
| Quality | Injected context relevance score >0.6 |

**Test Plan**:
- Unit: Mock daemon responses, verify fallback behavior
- Integration: Real daemon, measure latency P50/P95/P99
- E2E: Full Claude Code session with hook enabled

**Integration Points**:
- `lever_of_archimedes/hooks/user_prompt_submit.sh`
- `lever_of_archimedes/hooks/lib/research_kb.sh`

---

## Archived Audits

Strategic recommendations preserved in `docs/archive/gemini_audit_report_2026-01-08.md`:
- RRF (Reciprocal Rank Fusion) for search
- Semantic chunking overhaul
- Neo4j/KuzuDB migration for graph performance

---

## Quick Reference

```bash
# Run E2E tests
pytest packages/mcp-server/tests/e2e/ -v -m e2e

# Run retrieval evaluation
python scripts/eval_retrieval.py --output-format json

# Benchmark latency
time research-kb query "instrumental variables" --limit 5
```
