# Graph Search Default Validation Tracker

**Purpose**: Track 3 weekly validation runs before enabling graph search by default in production.

**Status**: ⏳ **In Progress** (0/3 runs complete)

**Requirement**: Must have ≥3 successful runs with all quality gates passing.

---

## Validation Runs

### Run 1: [Date TBD]

**Status**: ⏸️ Pending

**Trigger**: Manual via GitHub Actions UI

**Quality Gates**:
- [ ] Corpus ingestion successful (~500 chunks)
- [ ] Retrieval validation passed (Precision@K ≥90%)
- [ ] Concept extraction completed (≥100 concepts)
- [ ] Seed concept validation passed (Recall ≥70%)
- [ ] Graph validation passed (relationships exist)
- [ ] All CLI tests passed (47 tests)
- [ ] All script tests passed (24 tests)
- [ ] Database cached successfully

**Performance Metrics**:
- Query latency p50: ___ ms
- Query latency p95: ___ ms
- Query latency p99: ___ ms
- Corpus ingestion time: ___ minutes
- Concept extraction time: ___ minutes

**Issues Found**: None

**Notes**:

---

### Run 2: [Date TBD]

**Status**: ⏸️ Pending

**Trigger**: Manual via GitHub Actions UI

**Quality Gates**:
- [ ] Corpus ingestion successful (~500 chunks)
- [ ] Retrieval validation passed (Precision@K ≥90%)
- [ ] Concept extraction completed (≥100 concepts)
- [ ] Seed concept validation passed (Recall ≥70%)
- [ ] Graph validation passed (relationships exist)
- [ ] All CLI tests passed (47 tests)
- [ ] All script tests passed (24 tests)
- [ ] Database cached successfully

**Performance Metrics**:
- Query latency p50: ___ ms
- Query latency p95: ___ ms
- Query latency p99: ___ ms
- Corpus ingestion time: ___ minutes
- Concept extraction time: ___ minutes

**Issues Found**: None

**Notes**:

---

### Run 3: [Date TBD]

**Status**: ⏸️ Pending

**Trigger**: Manual via GitHub Actions UI

**Quality Gates**:
- [ ] Corpus ingestion successful (~500 chunks)
- [ ] Retrieval validation passed (Precision@K ≥90%)
- [ ] Concept extraction completed (≥100 concepts)
- [ ] Seed concept validation passed (Recall ≥70%)
- [ ] Graph validation passed (relationships exist)
- [ ] All CLI tests passed (47 tests)
- [ ] All script tests passed (24 tests)
- [ ] Database cached successfully

**Performance Metrics**:
- Query latency p50: ___ ms
- Query latency p95: ___ ms
- Query latency p99: ___ ms
- Corpus ingestion time: ___ minutes
- Concept extraction time: ___ minutes

**Issues Found**: None

**Notes**:

---

## Aggregate Results

**Runs**: 0/3 complete

**Success Rate**: N/A

**Average Metrics**:
- Query latency p50: N/A
- Query latency p95: N/A
- Query latency p99: N/A
- Corpus ingestion time: N/A
- Concept extraction time: N/A

**Precision@K**: N/A

**Concept Recall**: N/A

---

## Quality Gate Thresholds

### Must Pass (Blockers)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Corpus ingestion** | ≥450 chunks | Phase 1 target was ~500 chunks |
| **Retrieval Precision@5** | ≥90% | Known-answer tests must pass |
| **Test pass rate** | 100% | All tests must pass |
| **Database caching** | Success | Daily validation depends on this |

### Should Pass (Warnings)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Concept extraction** | ≥100 concepts | Minimum for graph signals |
| **Seed concept recall** | ≥70% | Quality of extraction |
| **Query latency p95** | <300ms | User experience |
| **Concept extraction time** | <20min | CI timeout constraint |

### May Fail (Acceptable)

| Metric | Notes |
|--------|-------|
| **Ollama installation** | Optional, may not work in CI |
| **Full concept extraction** | May timeout (20min limit) |

---

## Decision Criteria

**✅ APPROVE for production** if:
- All 3 runs completed successfully
- No blockers in any run
- ≤1 warning per run
- Performance stable across runs (±20%)

**⚠️ CONDITIONAL APPROVAL** if:
- 2/3 runs successful with only warnings
- Warnings are understood and documented
- Mitigation plan exists

**❌ REJECT** if:
- Any run has blockers
- >1 run failed completely
- Performance regressions >50%
- Systematic issues across runs

---

## How to Trigger Validation Runs

### Option 1: Manual Workflow Dispatch (Recommended)

1. Go to GitHub repository
2. Navigate to **Actions** tab
3. Select **"Weekly Full Rebuild & Validation"** workflow
4. Click **"Run workflow"** button
5. Select branch (usually `main`)
6. Click **"Run workflow"**

### Option 2: Command Line (gh CLI)

```bash
# Install GitHub CLI if not already installed
# https://cli.github.com/

# Trigger the workflow
gh workflow run weekly-full-rebuild.yml

# Check status
gh run list --workflow=weekly-full-rebuild.yml

# View logs
gh run view <run-id> --log
```

### Option 3: Wait for Scheduled Run

The workflow runs automatically every Sunday at 2 AM UTC. However, manual runs are recommended for validation to control timing.

---

## Monitoring During Runs

### GitHub Actions UI

1. **Watch the workflow** in real-time via Actions tab
2. **Check job logs** for errors or warnings
3. **Download artifacts** (audit report, database dump)

### Key Steps to Monitor

| Step | Success Criteria | Failure Action |
|------|------------------|----------------|
| **Database creation** | Clean DB created | Check PostgreSQL logs |
| **Schema migration** | All migrations applied | Review migration files |
| **Embedding server** | Health check passes | Check port conflicts |
| **Corpus ingestion** | ~500 chunks ingested | Review PDF files |
| **Retrieval validation** | Precision@K ≥90% | Check test cases |
| **Concept extraction** | ≥100 concepts extracted | Check Ollama status |
| **Seed validation** | Recall ≥70% | Review seed concepts |
| **All tests** | 100% pass rate | Fix failing tests |

---

## Local Testing (Optional)

You can also test locally before triggering CI:

```bash
# 1. Start services
docker-compose up -d postgres
python -m research_kb_pdf.embed_server &

# 2. Run ingestion
python scripts/ingest_corpus.py

# 3. Run validation
python scripts/eval_retrieval.py

# 4. Extract concepts (if Ollama available)
python scripts/extract_concepts.py --limit 1000

# 5. Validate concepts
python scripts/validate_seed_concepts.py

# 6. Run tests
pytest packages/cli/tests/ -v
pytest tests/scripts/ -v
```

---

## After 3 Successful Runs

Once all 3 runs are successful:

1. ✅ **Update this document** with actual results
2. ✅ **Update MIGRATION_GRAPH_DEFAULT.md** status to "Validated"
3. ✅ **Create PR** to merge Phase 3C changes
4. ✅ **Document decision** in PR description
5. ✅ **Deploy to production** (if applicable)
6. ✅ **Monitor production** metrics for first week
7. ✅ **Update README** to remove "awaiting validation" notes

---

## Rollback Triggers

**Immediate rollback** if any of these occur:
- Query latency p95 > 500ms
- Error rate > 5%
- User complaints about relevance degradation
- Database connection pool exhaustion

**Investigate and fix** if:
- Query latency p95 > 300ms but <500ms
- Error rate > 1% but <5%
- Individual user complaints (check if valid)

---

## Contact

For questions or issues during validation:
- GitHub Issues: [repository-url]/issues
- Documentation: docs/MIGRATION_GRAPH_DEFAULT.md

---

**Last Updated**: 2025-12-02
**Next Review**: After Run 1 completion
