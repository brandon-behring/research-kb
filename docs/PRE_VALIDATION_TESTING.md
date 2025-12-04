# Pre-Validation Testing Plan

**What phases can we test BEFORE running the 3 validation workflows?**

This document outlines the remaining test phases (4-6) that should be implemented **before** the 3 validation runs to increase their success rate.

---

## Summary

**Completed Phases**:
- ✅ Phase 3: CLI Command Tests (47 tests)
- ✅ Phase 3B: Script Tests with CI/CD (24 tests)
- ✅ Phase 3C: Graph Search Default (implementation complete, awaiting validation)

**Remaining Phases (Can Test Now)**:
- ⏳ Phase 4: Ingestion Smoke Tests (4-6h) - 8 tests
- ⏳ Phase 5: Package Test Gaps (4-6h) - Missing coverage
- ⏳ Phase 6: Quality Validation Tests (6-8h) - 10 tests + script

**Total Effort**: 14-20 hours before validation runs

**Benefits**:
- ✅ Higher validation run success rate
- ✅ Catch issues locally before CI
- ✅ Comprehensive test coverage
- ✅ Automated quality checks

---

## Phase 4: Ingestion Smoke Tests (4-6 hours)

**Purpose**: Validate real PDF ingestion quality with actual fixture files.

**Why do this before validation?**
- The validation workflow runs `ingest_corpus.py`
- Smoke tests ensure PDFs ingest correctly
- Catches fixture file issues early
- Validates chunk quality metrics

### What Gets Tested (8 tests)

**File**: `tests/smoke/test_ingestion_quality.py` (NEW)

1. **test_ingest_simple_paper** - Basic paper ingestion
   - Validates: Chunk count, average length, no empty chunks

2. **test_ingest_complex_paper** - Multi-column paper with math
   - Validates: LaTeX handling, equation detection

3. **test_ingest_textbook** - Large textbook ingestion
   - Validates: Chapter structure, section hierarchy

4. **test_embedding_quality** - Embedding generation
   - Validates: Vector dimensions, embedding not null

5. **test_citation_extraction** - Reference parsing
   - Validates: At least 1 citation extracted

6. **test_concept_extraction_quality** - Concept extraction
   - Validates: ≥N concepts, avg confidence >0.7

7. **test_deduplication** - Duplicate detection
   - Validates: No duplicate chunks by hash

8. **test_full_pipeline_integration** - End-to-end
   - Validates: Source → Chunks → Embeddings → Concepts

### Implementation Steps

```bash
# 1. Create test directory
mkdir -p tests/smoke

# 2. Create test files
touch tests/smoke/__init__.py
touch tests/smoke/conftest.py
touch tests/smoke/test_ingestion_quality.py

# 3. Implement tests (see plan for details)
vim tests/smoke/test_ingestion_quality.py

# 4. Run smoke tests locally
pytest tests/smoke/ -v -m smoke

# 5. Fix any issues found
```

### Expected Outcomes

**If all tests pass**:
- ✅ PDFs ingest correctly
- ✅ Chunk quality is good
- ✅ Embeddings generate successfully
- ✅ Validation runs will have higher success rate

**If tests fail**:
- ❌ Fix fixture files
- ❌ Fix ingestion issues
- ❌ Re-run tests until passing
- ❌ Then proceed to validation

**Effort**: 4-6 hours

---

## Phase 5: Package Test Gaps (4-6 hours)

**Purpose**: Fill missing test coverage in packages.

**Why do this before validation?**
- Ensures all packages have basic test coverage
- Catches bugs in untested code
- Improves overall system reliability

### What Gets Tested

**1. Check which packages need tests**:

```bash
# List all packages
ls -la packages/

# Check which have tests
find packages/ -name "tests" -type d

# Identify gaps
```

**Likely candidates**:
- `packages/search/` - If it has code
- `packages/interface/` - If it has code
- `packages/extraction/` - Additional coverage
- `packages/common/` - Utility function coverage

**2. Implement missing tests**:

For each package missing tests:
- Create `tests/` directory
- Create `__init__.py`
- Create `test_*.py` files
- Write unit tests for key functions
- Aim for 60%+ coverage

### Implementation Steps

```bash
# 1. Audit packages
for pkg in packages/*/; do
  if [ ! -d "$pkg/tests" ]; then
    echo "Missing tests: $pkg"
  fi
done

# 2. Create test files for packages without tests
# Example:
mkdir -p packages/extraction/tests
touch packages/extraction/tests/__init__.py
touch packages/extraction/tests/test_extractor.py

# 3. Implement tests
vim packages/extraction/tests/test_extractor.py

# 4. Run tests
pytest packages/ -v

# 5. Check coverage
pytest packages/ --cov=packages/ --cov-report=term-missing
```

### Expected Outcomes

**Target coverage**:
- Core packages (storage, pdf-tools): 80%+
- Supporting packages (extraction, common): 60%+
- Minimal packages (contracts): 40%+ (mostly models)

**Effort**: 4-6 hours (could be less if packages are empty)

---

## Phase 6: Quality Validation Tests (6-8 hours)

**Purpose**: Automated quality metrics validation.

**Why do this before validation?**
- The validation workflow runs quality checks
- Ensures extraction quality meets thresholds
- Automates quality monitoring
- Prevents quality regressions

### What Gets Tested (10 tests)

**File**: `tests/quality/test_extraction_metrics.py` (NEW)

1. **test_seed_concept_recall_threshold**
   - Validates: Overall recall >70%, Method recall >75%

2. **test_concept_confidence_distribution**
   - Validates: Avg confidence >70%

3. **test_retrieval_precision_threshold**
   - Validates: Precision@5 ≥90%, Precision@10 ≥85%

4. **test_no_duplicate_concepts**
   - Validates: No duplicate canonical names

5. **test_relationship_coverage**
   - Validates: Relationships/concept ratio >0.3

6. **test_citation_extraction_rate**
   - Validates: ≥50% of chunks have citations

7. **test_embedding_quality**
   - Validates: No null embeddings, dimensions correct

8. **test_chunk_length_distribution**
   - Validates: Avg 500-2000 tokens, <5% too short

9. **test_search_latency**
   - Validates: p95 <300ms, p99 <500ms

10. **test_graph_connectivity**
    - Validates: >80% concepts connected, no isolated clusters

### Quality Validation Script

**File**: `scripts/run_quality_checks.py` (NEW)

```python
#!/usr/bin/env python3
"""Run quality checks and exit with error if below threshold."""

async def main():
    # Run quality tests
    result = pytest.main([
        "tests/quality/",
        "-m", "quality",
        "--json-report",
    ])

    # Extract metrics
    metrics = extract_metrics(".quality_report.json")

    # Validate thresholds
    failures = []

    if metrics["recall"] < 0.70:
        failures.append("Recall below 70%")

    if metrics["precision"] < 0.90:
        failures.append("Precision@5 below 90%")

    if metrics["avg_confidence"] < 0.70:
        failures.append("Avg confidence below 70%")

    if failures:
        print("❌ Quality checks failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("✅ All quality checks passed")
```

### Implementation Steps

```bash
# 1. Create quality tests directory
mkdir -p tests/quality

# 2. Create test files
touch tests/quality/__init__.py
touch tests/quality/conftest.py
touch tests/quality/test_extraction_metrics.py

# 3. Implement quality tests
vim tests/quality/test_extraction_metrics.py

# 4. Create quality check script
vim scripts/run_quality_checks.py
chmod +x scripts/run_quality_checks.py

# 5. Run quality tests locally
pytest tests/quality/ -v -m quality

# 6. Run quality check script
python scripts/run_quality_checks.py
```

### Expected Outcomes

**Quality thresholds**:
- ✅ Seed concept recall ≥70%
- ✅ Retrieval Precision@5 ≥90%
- ✅ Avg confidence ≥70%
- ✅ No duplicate concepts
- ✅ Search latency p95 <300ms

**Effort**: 6-8 hours

---

## Testing Strategy

### Local Testing First (Recommended)

**Before implementing phases 4-6, test what we have**:

```bash
# 1. Run local validation script
./scripts/run_local_validation.sh

# This tests:
# - Database setup
# - Corpus ingestion
# - Retrieval quality
# - Concept extraction
# - All existing tests
```

**Fix any issues found** before proceeding to phases 4-6.

### Implement Phases in Order

**Recommended sequence**:

1. **Phase 4: Smoke Tests** (4-6h)
   - Tests actual PDFs
   - Validates ingestion pipeline
   - Catches fixture issues
   - Run first because validation depends on ingestion

2. **Phase 6: Quality Tests** (6-8h)
   - Tests extraction quality
   - Validates metrics
   - Creates quality script
   - Run second because it validates extraction

3. **Phase 5: Package Gaps** (4-6h)
   - Fill coverage holes
   - Less critical than 4 & 6
   - Can be done in parallel or last

### Testing Each Phase

**After implementing each phase**:

```bash
# Run the new tests
pytest tests/smoke/ -v          # Phase 4
pytest tests/quality/ -v        # Phase 6
pytest packages/<pkg>/tests/ -v # Phase 5

# Run ALL tests to ensure no regressions
pytest -v

# Check coverage
pytest --cov=packages/ --cov=tests/ --cov-report=term-missing

# Run local validation again
./scripts/run_local_validation.sh
```

---

## Timeline

### Option 1: Sequential (Complete Before Validation)

| Day | Activity | Hours |
|-----|----------|-------|
| 1 | Phase 4: Smoke Tests | 4-6 |
| 2 | Phase 6: Quality Tests | 6-8 |
| 3 | Phase 5: Package Gaps | 4-6 |
| 4 | Local validation + fixes | 2-4 |
| **Total** | **Before validation** | **16-24h** |
| 5+ | Run validation workflow #1 | 1h |

### Option 2: Interleaved (Test While Waiting)

| Day | Activity | Hours |
|-----|----------|-------|
| 1 | Local validation + Run validation #1 | 2h |
| 2-4 | Wait (implement Phase 4) | 4-6 |
| 5 | Run validation #2 | 1h |
| 6-8 | Wait (implement Phase 6) | 6-8 |
| 9 | Run validation #3 | 1h |
| 10-12 | Wait (implement Phase 5) | 4-6 |
| **Total** | **Interleaved** | **17-22h** |

### Option 3: Minimum Viable (Skip Phase 5)

| Day | Activity | Hours |
|-----|----------|-------|
| 1 | Phase 4: Smoke Tests | 4-6 |
| 2 | Phase 6: Quality Tests | 6-8 |
| 3 | Local validation + fixes | 2-4 |
| 4 | Run validation #1 | 1h |
| **Total** | **Minimum before validation** | **13-19h** |

---

## Recommendations

### ✅ DO Before Validation

**Priority 1 (Critical)**:
- ✅ Local validation (`./scripts/run_local_validation.sh`)
- ✅ Fix any errors from local validation
- ✅ Phase 4: Smoke Tests (validates ingestion)

**Priority 2 (Highly Recommended)**:
- ✅ Phase 6: Quality Tests (validates extraction)
- ✅ Run quality check script
- ✅ Verify all quality thresholds met

**Priority 3 (Nice to Have)**:
- ⚠️ Phase 5: Package Gaps (general coverage)

### ⏭️ SKIP for Now (Do After Validation)

**Can be done later**:
- Phase 5: Package Test Gaps (if packages are empty or low priority)
- Additional edge case tests
- Performance optimization

### 🎯 Recommended Approach

**Best balance of thoroughness and speed**:

```bash
# Day 1: Test what we have
./scripts/run_local_validation.sh
# Fix any issues found

# Day 1-2: Implement Phase 4 (4-6h)
# - Smoke tests for ingestion
# - Validates actual PDFs work
pytest tests/smoke/ -v

# Day 2-3: Implement Phase 6 (6-8h)
# - Quality validation tests
# - Quality check script
pytest tests/quality/ -v
python scripts/run_quality_checks.py

# Day 3: Final local validation
./scripts/run_local_validation.sh
# All should pass now

# Day 4: Trigger validation workflow #1
# Via GitHub UI or CLI
# High confidence of success

# Days 5-10: Wait + implement Phase 5 (optional)

# Day 11: Trigger validation workflow #2

# Days 12-17: Wait

# Day 18: Trigger validation workflow #3

# Day 19: Review + deploy
```

---

## Benefits of Pre-Testing

### Higher Validation Success Rate

**Without pre-testing**:
- ❌ Unknown issues in ingestion
- ❌ Unknown quality metrics
- ❌ Validation runs may fail
- ❌ Wasted CI minutes
- ❌ Longer feedback loop

**With pre-testing**:
- ✅ Ingestion validated locally
- ✅ Quality thresholds verified
- ✅ Higher validation success rate
- ✅ Saves CI minutes
- ✅ Faster iterations

### Comprehensive Coverage

**Current test count**: 71 tests (47 CLI + 24 script)

**After phases 4-6**: 99+ tests
- 47 CLI tests
- 24 script tests
- 8 smoke tests
- 10 quality tests
- 10+ package tests (Phase 5)

**Coverage improvement**: ~40% increase

### Automated Quality Monitoring

**After Phase 6**:
- ✅ Quality script runs automatically
- ✅ Fails if metrics below threshold
- ✅ Prevents quality regressions
- ✅ Can be added to CI workflow

---

## Quick Start

**Want to start testing now?**

### Minimal Path (4-6 hours):

```bash
# 1. Test current state
./scripts/run_local_validation.sh

# 2. Implement Phase 4 only
mkdir -p tests/smoke
vim tests/smoke/test_ingestion_quality.py
pytest tests/smoke/ -v

# 3. Run validation workflow #1
# Via GitHub UI

# Done! Phase 6 can be done during wait periods
```

### Recommended Path (10-14 hours):

```bash
# 1. Test current state
./scripts/run_local_validation.sh

# 2. Implement Phase 4
# (4-6 hours)

# 3. Implement Phase 6
# (6-8 hours)

# 4. Final validation
./scripts/run_local_validation.sh

# 5. Trigger validation workflow #1
# Via GitHub UI

# High confidence!
```

---

## Questions?

**Q: Can I skip phases 4-6 and go straight to validation?**
A: Yes, but validation runs have lower success chance. Recommended to at least run local validation script first.

**Q: Which phase is most important?**
A: Phase 4 (Smoke Tests) - validates the ingestion pipeline that runs in the validation workflow.

**Q: Can I implement phases during wait periods?**
A: Yes! Option 2 timeline shows this approach.

**Q: How long does local validation take?**
A: 15-30 minutes (without Ollama), 30-60 minutes (with Ollama).

**Q: What if local validation fails?**
A: Fix the issues before running CI validation. Local is faster for debugging.

---

## Summary

**Current State**:
- ✅ 71 tests implemented (CLI + scripts)
- ✅ Graph search default ready
- ⏸️ Awaiting 3 validation runs

**Next Steps** (Choose One):

**Option A: Thorough (Recommended)**
1. Implement Phase 4 (4-6h)
2. Implement Phase 6 (6-8h)
3. Run local validation
4. Trigger validation workflow #1

**Option B: Minimal**
1. Run local validation
2. Trigger validation workflow #1
3. Implement phases during wait periods

**Option C: Maximum**
1. Implement Phases 4, 5, 6 (14-20h)
2. Run local validation
3. Trigger validation workflow #1
4. Ultra-high confidence

**My Recommendation**: **Option A (Thorough)**
- Balanced approach
- High success rate
- Manageable time investment
- Good test coverage

---

**Ready to start?** See implementation details in the original plan:
`$HOME/.claude/plans/gentle-growing-sutton.md` (lines 2187-2330)
