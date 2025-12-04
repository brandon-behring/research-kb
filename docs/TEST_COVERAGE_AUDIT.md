# Test Coverage Audit Report

**Date:** 2025-12-02
**Phase:** 5 - Package Test Gaps Analysis

## Executive Summary

- **Total Packages:** 8
- **Packages with Tests:** 6
- **Empty Packages:** 2 (interface, search)
- **Test Gap Modules:** 13

## Package-by-Package Analysis

### ✅ CLI (`packages/cli`)

**Status:** GOOD COVERAGE

**Source Modules:**
- `__init__.py`
- `formatters.py`
- `main.py`

**Test Files:**
- `test_cli_commands.py` (47 tests)
- `test_formatters.py`
- `test_graph_search.py` (7 tests)

**Coverage:** ~100% - All modules tested
**Priority:** None (complete)

---

### ⚠️  Common (`packages/common`)

**Status:** 1 MODULE MISSING TESTS

**Source Modules:**
- `errors.py` ✅
- `instrumentation.py` ❌
- `logging_config.py` ✅
- `retry.py` ✅

**Test Files:**
- `test_errors.py`
- `test_logging.py`
- `test_retry.py`

**Missing Tests:**
- `instrumentation.py` - Performance tracking and monitoring utilities

**Priority:** MEDIUM
**Reason:** Instrumentation is used across the codebase, but failures are non-critical

---

### ✅ Contracts (`packages/contracts`)

**Status:** GOOD COVERAGE

**Source Modules:**
- `models.py`

**Test Files:**
- `test_contract_models.py`

**Coverage:** 100%
**Priority:** None (complete)

---

### ⚠️  Extraction (`packages/extraction`)

**Status:** 2 MODULES MISSING TESTS

**Source Modules:**
- `concept_extractor.py` ✅
- `deduplicator.py` ✅
- `graph_sync.py` ❌
- `models.py` ✅
- `ollama_client.py` ✅
- `prompts.py` ❌

**Test Files:**
- `test_concept_extractor.py`
- `test_deduplicator.py`
- `test_extraction_models.py`
- `test_ollama_client.py`

**Missing Tests:**
- `graph_sync.py` - Synchronizes concepts to graph database
- `prompts.py` - LLM prompt templates

**Priority:** HIGH
**Reason:**
- `graph_sync.py` is critical for graph construction
- `prompts.py` contains static templates (low risk, but should validate format)

---

### ⚠️  PDF Tools (`packages/pdf-tools`)

**Status:** 2 MODULES MISSING TESTS

**Source Modules:**
- `bibtex_generator.py` ✅
- `chunker.py` ✅
- `dispatcher.py` ✅
- `dlq.py` ❌
- `embed_server.py` ❌
- `embedding_client.py` ✅
- `grobid_client.py` ✅
- `pymupdf_extractor.py` ✅

**Test Files:**
- `test_bibtex_generator.py`
- `test_chunker.py`
- `test_dispatcher.py`
- `test_embedding.py` (covers `embedding_client.py`)
- `test_grobid.py`
- `test_heading_detection.py`
- `test_pymupdf_extractor.py`
- `test_pymupdf_real.py`

**Missing Tests:**
- `dlq.py` - Dead letter queue for failed operations
- `embed_server.py` - Embedding server management

**Priority:** MEDIUM
**Reason:**
- `dlq.py` handles error cases (important but not in hot path)
- `embed_server.py` is utility code for server management

---

### ⚠️  Storage (`packages/storage`)

**Status:** 6 MODULES MISSING TESTS

**Source Modules:**
- `assumption_store.py` ❌
- `chunk_concept_store.py` ❌
- `chunk_store.py` ✅
- `citation_store.py` ✅
- `concept_store.py` ✅
- `connection.py` ❌
- `graph_queries.py` ✅
- `method_store.py` ❌
- `query_extractor.py` ❌
- `relationship_store.py` ❌
- `search.py` ✅
- `source_store.py` ✅

**Test Files:**
- `test_chunk_store.py`
- `test_citation_store.py`
- `test_concept_store.py`
- `test_graph_queries.py`
- `test_search.py`
- `test_source_store.py`

**Missing Tests:**
- `assumption_store.py` - Stores research assumptions
- `chunk_concept_store.py` - Links chunks to concepts
- `connection.py` - Database connection management
- `method_store.py` - Stores research methods
- `query_extractor.py` - Extracts structured queries
- `relationship_store.py` - Stores concept relationships

**Priority:** HIGH
**Reason:**
- Storage layer is critical infrastructure
- `relationship_store.py` is used for graph search (critical)
- `chunk_concept_store.py` links chunks to concepts (critical)
- Connection management needs validation

---

### ⊘ Interface (`packages/interface`)

**Status:** EMPTY PACKAGE (NO SOURCE CODE)

**Test Priority:** None (skip)

---

### ⊘ Search (`packages/search`)

**Status:** EMPTY PACKAGE (NO SOURCE CODE)

**Test Priority:** None (skip)

---

## Summary of Test Gaps

### Critical Priority (implement immediately)
1. **storage/relationship_store.py** - Used for graph search
2. **storage/chunk_concept_store.py** - Links chunks to concepts
3. **extraction/graph_sync.py** - Synchronizes concept graph

### High Priority (implement in Phase 5)
4. **storage/query_extractor.py** - Query processing
5. **storage/connection.py** - DB connection pooling
6. **storage/method_store.py** - Method storage
7. **storage/assumption_store.py** - Assumption storage

### Medium Priority (implement if time allows)
8. **common/instrumentation.py** - Performance monitoring
9. **pdf-tools/dlq.py** - Error handling
10. **pdf-tools/embed_server.py** - Server management

### Low Priority (can defer)
11. **extraction/prompts.py** - Static prompt templates

---

## Recommended Implementation Order

### Phase 5A: Critical Storage Tests (2-3h)
1. `test_relationship_store.py` - CRUD operations, graph queries
2. `test_chunk_concept_store.py` - Link creation, retrieval
3. `test_connection.py` - Pool management, error handling

### Phase 5B: Extraction Graph Tests (1-2h)
4. `test_graph_sync.py` - Concept synchronization
5. `test_prompts.py` - Validate prompt templates

### Phase 5C: Secondary Storage Tests (1-2h)
6. `test_query_extractor.py` - Query parsing
7. `test_method_store.py` - CRUD operations
8. `test_assumption_store.py` - CRUD operations

### Phase 5D: Utilities (optional, 1h)
9. `test_instrumentation.py` - Metrics collection
10. `test_dlq.py` - Error queue operations
11. `test_embed_server.py` - Server lifecycle

---

## Test Count Projection

**Current:**
- Unit tests: 71
- CLI tests: 47
- Script tests: 24
- Smoke tests: 8
- Quality tests: 10
- **Total: ~160 tests**

**After Phase 5 (Critical + High):**
- Add ~35 new unit tests
- **Projected Total: ~195 tests**

**After Phase 5 (All priorities):**
- Add ~50 new unit tests
- **Projected Total: ~210 tests**

---

## Coverage Targets

**Current Estimated Coverage:**
- CLI: ~100%
- Common: ~75% (missing instrumentation)
- Contracts: 100%
- Extraction: ~67% (missing 2/6 modules)
- PDF-tools: ~75% (missing 2/8 modules)
- Storage: ~50% (missing 6/12 modules)

**After Phase 5A-C (Critical + High):**
- Storage: ~85%
- Extraction: ~100%
- Overall: ~80%+

---

## Recommendations

1. **Implement Phase 5A immediately** - Critical storage infrastructure
2. **Run quality tests after 5A** - Verify relationship queries work
3. **Implement Phase 5B** - Complete extraction coverage
4. **Phase 5C optional** - Defer if validation passes
5. **Phase 5D skip** - Low risk, can defer to future work

---

## Next Steps

1. Mark audit complete ✅
2. Create `test_relationship_store.py`
3. Create `test_chunk_concept_store.py`
4. Create `test_connection.py`
5. Create `test_graph_sync.py`
6. Create `test_prompts.py`
7. Create `test_query_extractor.py`
8. Create `test_method_store.py`
9. Create `test_assumption_store.py`
10. Run all new tests locally
