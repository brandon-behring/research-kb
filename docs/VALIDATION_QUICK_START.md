# Quick Start: Graph Search Validation

**Goal**: Run 3 successful validation runs to prove graph search quality before production deployment.

---

## TL;DR

```bash
# Option 1: Via GitHub UI (recommended)
1. Go to GitHub → Actions → "Weekly Full Rebuild & Validation"
2. Click "Run workflow" → Select "main" branch → Run
3. Wait 60 minutes
4. Review results
5. Update docs/VALIDATION_TRACKER.md
6. Repeat 2 more times (3 total runs)

# Option 2: Via GitHub CLI
gh workflow run weekly-full-rebuild.yml
gh run watch $(gh run list --workflow=weekly-full-rebuild.yml --limit 1 --json databaseId --jq '.[0].databaseId')

# Option 3: Local testing first (optional)
./scripts/run_local_validation.sh
```

---

## Step-by-Step

### 1. Create Tracking Issue

```bash
# Via GitHub UI:
# Go to Issues → New Issue → Use "Graph Search Default Validation Tracking" template

# Or via CLI:
gh issue create --title "Validation: Graph Search as Default (Phase 3C)" \
  --label validation,phase-3c,testing \
  --body-file .github/ISSUE_TEMPLATE/validation_tracking.md
```

### 2. Run First Validation

**Via GitHub UI**:
1. Navigate to repository → **Actions** tab
2. Select **"Weekly Full Rebuild & Validation"** workflow
3. Click **"Run workflow"** button
4. Select branch: `main`
5. Click **"Run workflow"**

**Via GitHub CLI**:
```bash
gh workflow run weekly-full-rebuild.yml
```

### 3. Monitor Progress

**Via GitHub UI**:
- Watch the workflow run in real-time
- Click on steps to see logs
- Check for green checkmarks (success) or red X (failure)

**Via GitHub CLI**:
```bash
# Get the latest run ID
RUN_ID=$(gh run list --workflow=weekly-full-rebuild.yml --limit 1 --json databaseId --jq '.[0].databaseId')

# Watch the run
gh run watch $RUN_ID

# View logs
gh run view $RUN_ID --log
```

### 4. Record Results

**Download artifacts**:
```bash
gh run download $RUN_ID
```

**Update tracking**:
```bash
# Edit docs/VALIDATION_TRACKER.md
# Fill in Run 1 section with:
# - Date
# - Status (✅ Complete)
# - Quality gates (checked/unchecked)
# - Performance metrics from logs
# - Any issues or notes

# Update GitHub issue
# Copy results to tracking issue
```

### 5. Wait 3-7 Days

**Why wait?**
- Different conditions (time of day, system load)
- Catch intermittent issues
- Prove consistency

### 6. Run Second Validation

Repeat steps 2-4 for Run 2.

### 7. Run Third Validation

Repeat steps 2-4 for Run 3.

### 8. Make Decision

**After 3 runs complete**:

```bash
# Review aggregate results in docs/VALIDATION_TRACKER.md

# If all 3 runs passed:
# ✅ Update docs/MIGRATION_GRAPH_DEFAULT.md status to "Validated"
# ✅ Create PR to merge Phase 3C
# ✅ Deploy to production

# If any run failed:
# ❌ Investigate issues
# ❌ Fix problems
# ❌ Re-run failed validation
```

---

## Success Criteria

**✅ PASS if**:
- All 3 runs completed successfully
- No blockers in any run
- ≤1 warning per run
- Performance stable (±20%)

**❌ FAIL if**:
- Any run has blockers
- >1 run failed completely
- Performance regressions >50%

---

## Blockers (Must Pass)

- [ ] Corpus ingestion ≥450 chunks
- [ ] Retrieval Precision@5 ≥90%
- [ ] All tests pass (71/71)
- [ ] Database cached successfully

---

## Warnings (Acceptable)

- Concept extraction <100 concepts (acceptable if ≥50)
- Seed concept recall <70% (acceptable if ≥50%)
- Ollama installation failed (acceptable, optional)
- Concept extraction timed out (acceptable, partial OK)

---

## Estimated Timeline

| Activity | Duration | When |
|----------|----------|------|
| **Run 1** | 60 min | Today |
| **Wait** | 3-7 days | - |
| **Run 2** | 60 min | Day 3-7 |
| **Wait** | 3-7 days | - |
| **Run 3** | 60 min | Day 6-14 |
| **Review** | 1 hour | Day 6-14 |
| **Total** | ~9-17 days | - |

---

## Cost

**GitHub Actions**: 2000 free minutes/month

**3 runs**: ~180 minutes (9% of free tier)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Workflow not found | Check `.github/workflows/weekly-full-rebuild.yml` exists on main branch |
| Run workflow button disabled | Ensure you have write access to repository |
| Corpus ingestion failed | Check embedding server health, review fixture files |
| Tests failed | Review test logs, fix issues, re-run |
| Concept extraction timed out | Acceptable (marked as warning), continue validation |

---

## Documents

- **Detailed tracker**: `docs/VALIDATION_TRACKER.md`
- **Migration guide**: `docs/MIGRATION_GRAPH_DEFAULT.md`
- **Trigger instructions**: `docs/TRIGGER_VALIDATION_WORKFLOW.md`
- **Local validation**: `scripts/run_local_validation.sh`

---

## Questions?

- GitHub issue: Create an issue with `validation` label
- Documentation: Review migration guide for detailed info
- Logs: Download workflow artifacts for detailed logs

---

**Last Updated**: 2025-12-02
