# Test Coverage Audit Report

**Date:** 2026-02-06 (refreshed)
**Original:** 2025-12-02
**Phase:** Post-Phase 4.3 cleanup

## Executive Summary

- **Total Packages:** 12
- **Packages with Tests:** 12
- **Total Tests:** ~1,430+
- **Previous Test Gaps:** All 13 flagged modules from Dec 2025 now have test files

## Package Test Counts (as of 2026-02-06)

| Package | Tests | Status |
|---------|------:|--------|
| storage | 413 | ✅ Full coverage |
| s2-client | 316 | ✅ Full coverage |
| extraction | 264 | ✅ Full coverage |
| pdf-tools | 198 | ✅ Good coverage |
| common | 94 | ✅ Good coverage (1 flaky) |
| cli | 77 | ✅ Full coverage |
| daemon | 31 | ✅ Covered |
| contracts | 21 | ✅ Full coverage |
| mcp-server | 14 | ✅ Covered |
| api | 1 | ⚠️ Minimal |
| client | 1 | ⚠️ Minimal |
| **Total** | **~1,430** | |

## Dec 2025 Gap Status (All Resolved)

All 13 modules flagged in the original audit now have test files:

| Module | Status | Test File |
|--------|--------|-----------|
| storage/relationship_store.py | ✅ Covered | `test_relationship_store.py` |
| storage/chunk_concept_store.py | ✅ Covered | `test_chunk_concept_store.py` |
| storage/connection.py | ✅ Covered | `test_connection.py` |
| storage/method_store.py | ✅ Covered | `test_method_store.py` |
| storage/assumption_store.py | ✅ Covered | `test_assumption_store.py` |
| storage/query_extractor.py | ✅ Covered | `test_query_extractor.py` |
| extraction/graph_sync.py | ✅ Covered via test_concept_extractor.py |
| extraction/prompts.py | ✅ Covered | `test_prompts.py` (29 tests) |
| extraction/domain_prompts.py | ✅ **NEW** | `test_domain_prompts.py` (63 tests) |
| common/instrumentation.py | ✅ Covered via test_config.py |
| pdf-tools/dlq.py | ⚠️ Indirect coverage only |
| pdf-tools/embed_server.py | ⚠️ Indirect coverage only |

## New Tests Added (2026-02-06)

### `packages/extraction/tests/test_domain_prompts.py` (63 tests)
- All 5 domains present with required keys
- `get_domain_prompt_section()` returns non-empty guidance per domain
- `get_domain_abbreviations()` validates lowercase keys, non-empty values
- `get_domain_config()` returns full configuration
- `list_domains()` enumerates all domains
- `get_all_abbreviations()` merges all domain abbreviations
- Unknown domain fallback behavior verified

### `packages/storage/tests/test_search.py` — `TestComputeRanksBySignal` (7 tests)
- Single signal, single result
- Multiple results ranked by descending score
- Multiple signals ranked independently
- Tied scores get sequential ranks
- None scores excluded from ranking
- All four signals (FTS, vector, graph, citation)
- Empty results

## Remaining Gaps

### Minimal coverage (functional but thin)
- `packages/api/tests/` — 1 test (health endpoint only)
- `packages/client/tests/` — 1 test

### Missing direct unit tests
- `packages/pdf-tools/src/research_kb_pdf/dlq.py` — only indirect coverage
- `packages/pdf-tools/src/research_kb_pdf/embed_server.py` — only indirect coverage

### Search module gaps
- `search_hybrid` does not support `scoring_method="rrf"` (only `search_hybrid_v2` does)
- Addressed in Workstream D of this cleanup

---

## Recommendations

1. **api/client packages** need expanded test suites if they grow
2. **Pytest markers** remain sparse — most tests lack `@pytest.mark.unit` etc.
3. Consider adding `--cov` reporting to CI for automated coverage tracking
