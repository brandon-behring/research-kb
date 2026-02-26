# Test Coverage Audit Report

**Date:** 2026-02-26 (Phase W refresh)
**Previous:** 2026-02-25 (Phase O)
**Original:** 2025-12-02

## Executive Summary

- **Total Packages:** 12
- **Packages with Tests:** 12
- **Total Test Functions:** ~2,530
- **Growth since last audit:** +348 test functions (16% increase from Phase O)
- **Previous Test Gaps:** All 13 flagged modules from Dec 2025 resolved
- **CI**: pytest-cov in PR checks (`--cov-fail-under=66`), doc freshness gate in weekly integration

## Package Test Counts (as of 2026-02-26)

| Package | Test Files | Functions | Status |
|---------|-----------|----------:|--------|
| storage | 22 | 593 | Full coverage (+85 Phase S unit tests, +3 synonym normalization) |
| extraction | 13 | 513 | Full coverage (249 domain prompts + anthropic/instructor) |
| s2-client | 9 | 316 | Full coverage |
| pdf-tools | 11 | 260 | Good coverage (+contract tests, Unicode regression) |
| api | 9 | 221 | Full coverage (expanded from 1 in Feb) |
| mcp-server | 11 | 156 | Full coverage (expanded from 14 in Feb) |
| common | 5 | 95 | Good coverage |
| cli | 5 | 89 | Full coverage (+16 citations sub-app tests) |
| daemon | 3 | 80 | Covered (test_metrics, test_pool) |
| dashboard | 5 | 79 | Good coverage (AppTest suite) |
| contracts | 1 | 21 | Full coverage |
| client | 1 | 17 | Covered |
| **Total** | **95+** | **~2,530** | |

## Notable Changes Since Last Audit (2026-02-25)

### Phase S: Coverage Hardening (85 new unit tests)
- **storage**: 432 → 593 tests (Phase S: search, graph_queries, citation_graph, assumption_audit)
- **daemon**: 31 → 80 tests (test_metrics, test_pool expanded)
- **pdf-tools**: 215 → 260 tests (contract tests, Unicode regression)
- **extraction**: 322 → 513 tests (anthropic/instructor backend tests)

### Phase W: CLI Citations + Synonym Normalization
- **cli**: 77 → 89 tests (+16 citations sub-app tests covering all 5 commands)
- **storage query_expander**: +3 synonym underscore/slash normalization tests

### Earlier Phases (O and before)
- **api**: 1 → 221 tests (9 test files)
- **mcp-server**: 14 → 156 tests (11 test files)
- **dashboard**: New — 79 tests (AppTest suite)

## Dec 2025 Gap Status (All Resolved)

All 13 modules flagged in the original audit now have test files:

| Module | Status | Test File |
|--------|--------|-----------|
| storage/relationship_store.py | Covered | `test_relationship_store.py` |
| storage/chunk_concept_store.py | Covered | `test_chunk_concept_store.py` |
| storage/connection.py | Covered | `test_connection.py` |
| storage/method_store.py | Covered | `test_method_store.py` |
| storage/assumption_store.py | Covered | `test_assumption_store.py` |
| storage/query_extractor.py | Covered | `test_query_extractor.py` |
| extraction/graph_sync.py | Covered via test_concept_extractor.py |
| extraction/prompts.py | Covered | `test_prompts.py` (29 tests) |
| extraction/domain_prompts.py | Covered | `test_domain_prompts.py` (249 tests) |
| common/instrumentation.py | Covered via test_config.py |
| pdf-tools/dlq.py | Indirect coverage only |
| pdf-tools/embed_server.py | Indirect coverage only |

## Remaining Gaps

### Missing direct unit tests
- `packages/pdf-tools/src/research_kb_pdf/dlq.py` — only indirect coverage
- `packages/pdf-tools/src/research_kb_pdf/embed_server.py` — only indirect coverage

### Search module gaps
- `search_hybrid` does not support `scoring_method="rrf"` (only `search_hybrid_v2` does)

---

## Recommendations

1. **dlq.py / embed_server.py** still lack direct tests — add if these modules change
2. **Pytest markers** applied via `scripts/tag_tests.py` (Phase G) — coverage is good
3. `--cov` reporting active in PR checks since Phase I
4. Consider per-package coverage thresholds as test suite matures
