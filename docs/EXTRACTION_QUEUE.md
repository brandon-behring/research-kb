# Extraction Queue

## Active Work

### 1. Concept Embedding Backfill (~225K concepts)

**Status:** ✅ **COMPLETE** (2025-12-23 14:25 UTC)
**Duration:** ~23 minutes
**Result:** 224,937 concepts embedded, 0 errors

**Verification:**
- All 257,726 concepts now have embeddings (100% coverage, post-dedup)
- Retrieval eval: **92.9% pass rate** (13/14 tests)

---

## Queued Work

### 2. Method/Assumption Enrichment (~25K concepts)

**Status:** ✅ **COMPLETE** (2025-12-24)
**Batch IDs:**
- `msgbatch_01QTb4XSRdUdqMstXFW1VSpR` (10,000 requests) ✅
- `msgbatch_01GSdnnDagoj3s3MqY6G76E2` (10,000 requests) ✅
- `msgbatch_014MEtvk38nEAfYxH8fYvcje` (5,608 requests) ✅

**Result:** 25,608 concepts processed, 0 errors
- Methods enriched: 15,937 (100% of method concepts)
- Assumptions enriched: 9,671 (100% of assumption concepts)

**Data now includes:**
- `required_assumptions` (for methods)
- `problem_types` (for methods)
- `common_estimators` (for methods)
- `mathematical_statement` (for assumptions)
- `is_testable` (for assumptions)
- `common_tests` (for assumptions)
- `violation_consequences` (for assumptions)
- `inference_confidence` (for both)

### 3. Chunk Extraction Backlog (17,206 chunks)

**Status:** ✅ **COMPLETE** (2025-12-26 23:00 UTC)
**Cost:** ~$29.18 (Haiku 4.5 with 50% batch discount)

**Batch IDs:**
- `msgbatch_01B24G9Vqbd3zoLfLfK1E51J` (10,000 chunks) ✅
- `msgbatch_01BrPkEX1rYTHeRPifebh2Te` (7,206 chunks) ✅

**Result:** 17,206 chunks processed, 0 failed
- New concepts: 161,145
- New relationships: 112,142
- New chunk-concept links: 161,145

---

## Current Coverage (as of 2025-12-27)

| Metric | Count |
|--------|------:|
| Chunks extracted | 129,856 / 142,221 (91.3%) |
| Concepts | 290,219 |
| Relationships | 678,772 |
| Chunk-concept links | 1,145,930 |
| Concept embeddings | 290,219 / 290,219 (100%) |

---

## Validation Status

- [x] VACUUM ANALYZE complete
- [x] Quality gates: **ALL PASSED**
- [x] Retrieval precision: **100%** (14/14 tests)
- [x] Graph CLI working
- [x] Schema migrations applied (004, 005)
- [x] S2-client installed and working
- [x] Concept embeddings (**COMPLETE** - 100%)
- [x] Full retrieval eval (**PASSED** - 13/14 tests, 92.9% above 90% target)
- [x] Method/assumption enrichment (**COMPLETE** - 25,608 concepts, 100%)
- [x] Remaining chunk extraction (**COMPLETE** - 17,206 chunks, 91.3% coverage)
- [x] New concept embedding backfill (**COMPLETE** - 32,493 concepts, 100% coverage)

---

## Completed 2025-12-26

### Technical Debt Sprint

1. **BGE Query Instructions** - Added asymmetric retrieval prefix to embed_query()
   - `QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "`
   - Updated CLI, API, dashboard, and scripts to use `embed_query()` for search
   - Improved short-query recall per BGE model recommendations

2. **Score Fusion Calibration** - Implemented min-max normalization
   - FTS and vector scores now normalized to 0-1 within each result set
   - Fixes 5.1x range imbalance between FTS and vector scores
   - All 14 search tests passing

3. **Concept Deduplication** - Merged 14,511 singular/plural pairs
   - Created `scripts/deduplicate_concepts.py` with dry-run mode
   - Properly updated all FK references (chunk_concepts, relationships, methods, assumptions)
   - Concepts: 272,237 → 257,726 (-5.3%)
   - Relationships: 600,192 → 591,080 (-1.5%)
   - Zero orphan references, zero remaining plural duplicates

**Validation**: Retrieval eval 13/14 (92.9% hit rate, MRR 0.849) - above 90% target.

---

## Completed 2025-12-24

1. **Processed enrichment batch** - 25,608 concepts (15,937 methods, 9,671 assumptions)
2. **Fixed type mismatch bug** - violation_consequences list→string conversion
3. **Test coverage sprint** - Created tests for query_extractor, method_store, assumption_store (52 new tests)
4. **MCP citation tools** - Added get_citing_sources and get_cited_sources tools
5. **260 total tests passing** - Storage (233) + MCP (27)

## Completed 2025-12-23

1. **Fixed embedding backfill bug** - Added `register_vector(conn)` call
2. **Completed embedding backfill** - 224,937 concepts in ~23 min, 100% coverage
3. **Retrieval eval passed** - 14/14 tests (100%)
4. **Created migration 004** - Expanded concept_type CHECK constraint (idempotent)
5. **Created migration 005** - Added provenance columns for enrichment
6. **Added pgcrypto extension** - Fixed fresh install compatibility
7. **Added anthropic optional dependency** - Clean package separation
8. **Installed s2-client** - Citation enrichment now working
9. **Created enrichment scripts** - Submission + processing
10. **Submitted enrichment batch** - 25,608 concepts across 3 batches

---

## Notes

- Archived previous checkpoint/DLQ to `.archive/embedding_backfill_20251223_140240/`
- Embedding backfill expected to complete by ~16:00 UTC
- After backfill: run `python scripts/eval_retrieval.py` to validate end-to-end
