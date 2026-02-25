# Test Coverage Audit Report

**Date:** 2026-02-25 (Phase O refresh)
**Previous:** 2026-02-06
**Original:** 2025-12-02

## Executive Summary

- **Total Packages:** 12
- **Packages with Tests:** 12
- **Total Test Functions:** ~2,182
- **Growth since last audit:** +752 test functions (52% increase)
- **Previous Test Gaps:** All 13 flagged modules from Dec 2025 resolved
- **CI**: pytest-cov in PR checks (`--cov-fail-under=40`), doc freshness gate in weekly integration

## Package Test Counts (as of 2026-02-25)

| Package | Test Files | Functions | Status |
|---------|-----------|----------:|--------|
| storage | 18 | 432 | Full coverage |
| s2-client | 9 | 316 | Full coverage |
| extraction | 13 | 322 | Full coverage (249 domain prompts) |
| api | 9 | 221 | Full coverage (expanded from 1 in Feb) |
| pdf-tools | 11 | 215 | Good coverage (+3 Unicode regression) |
| common | 5 | 95 | Good coverage |
| mcp-server | 11 | 156 | Full coverage (expanded from 14 in Feb) |
| dashboard | 5 | 79 | Good coverage (AppTest suite) |
| cli | 5 | 77 | Full coverage |
| daemon | 3 | 31 | Covered |
| contracts | 1 | 21 | Full coverage |
| client | 1 | 17 | Covered |
| **Total** | **91** | **~2,182** | |

## Notable Changes Since Last Audit (2026-02-06)

### Dramatic Expansions
- **api**: 1 → 221 tests (9 test files: schemas, metrics, service, main, etc.)
- **mcp-server**: 14 → 156 tests (11 test files: formatters, extended formatters, etc.)
- **extraction**: 264 → 322 tests (domain prompts grew from 63 → 249 with Phase H/N/O)
- **dashboard**: New — 79 tests (5 test files, AppTest suite added in Phase M)
- **pdf-tools**: 198 → 215 tests (Unicode normalization regression tests added in Phase O)

### Phase O Additions
- `test_domain_prompts.py`: 249 tests (20 domains including portfolio_management)
- `test_chunker.py`: +3 Unicode normalization regression tests (NFKC for NBSP/SHY/ligatures)

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
