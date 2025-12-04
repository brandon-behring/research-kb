# Validation Quick Reference Card

**One-page reference for running the 3 validation workflows.**

Print this page or keep it open for quick access during validation runs.

---

## 🚀 Start Here

**Goal**: Run 3 successful validation runs spaced 3-7 days apart

**Method**: GitHub Web UI (easiest, no setup required)

**Time per run**: ~60 minutes (hands-off)

**Total timeline**: 9-17 days

---

## ✅ Prerequisites Checklist

- [ ] GitHub account (sign up at https://github.com/signup)
- [ ] Repository access with **write** permissions
- [ ] GitHub Actions enabled
- [ ] Workflow file exists: `.github/workflows/weekly-full-rebuild.yml`

**Verify**: Go to repo → Actions tab → See "Weekly Full Rebuild & Validation" in sidebar

---

## 📋 Run Validation (Repeat 3 Times)

### Step 1: Trigger (2 minutes)

```
1. Go to: https://github.com/YOUR_USERNAME/research-kb
2. Click: Actions tab
3. Select: "Weekly Full Rebuild & Validation" (left sidebar)
4. Click: "Run workflow" button (top right)
5. Select: "main" branch
6. Click: Green "Run workflow" button
```

### Step 2: Monitor (60 minutes)

```
1. Click on the new workflow run (appears after 10 seconds)
2. Click on "full-pipeline" job
3. Watch the steps complete (✓ = done, ⚫ = in progress, ✗ = failed)
```

**Critical steps to watch**:
- ✓ Ingest corpus → Must show ≥450 chunks
- ✓ Validate retrieval → Must show ≥90% Precision@5
- ✓ Run all CLI tests → Must show 47 passed
- ✓ Run all script tests → Must show 24 passed

### Step 3: Download Results (2 minutes)

```
1. Scroll to bottom of workflow run page
2. Find "Artifacts" section
3. Click: "weekly-audit-report"
4. Save the ZIP file
```

### Step 4: Record Results (10 minutes)

```
1. Go to: docs/VALIDATION_TRACKER.md
2. Find: "Run 1" (or Run 2, Run 3)
3. Update:
   - Date
   - Status (✅ Complete or ❌ Failed)
   - Check all quality gates
   - Fill in performance metrics from logs
4. Commit changes
```

### Step 5: Wait

```
⏰ Wait 3-7 days before next run
```

---

## 📊 Quality Gates Checklist

**Must Pass (Blockers)**:
- [ ] Corpus ingestion ≥450 chunks
- [ ] Retrieval Precision@5 ≥90%
- [ ] CLI tests 47/47 passed
- [ ] Script tests 24/24 passed
- [ ] Database cached successfully

**Should Pass (Warnings)**:
- [ ] Concept extraction ≥100 concepts
- [ ] Seed recall ≥70%

**May Fail (Acceptable)**:
- [ ] Ollama installation (optional)
- [ ] Concept extraction timeout (partial OK)

---

## 🎯 After 3 Runs

### If All 3 Passed:

```
1. Update docs/VALIDATION_TRACKER.md aggregate results
2. Update docs/MIGRATION_GRAPH_DEFAULT.md status to "Validated"
3. Create PR: "Phase 3C: Enable Graph Search by Default (Validated)"
4. Merge PR
5. Deploy to production
```

### If Any Run Failed:

```
1. Investigate failure in workflow logs
2. Fix the issue
3. Re-run that validation
4. Continue until 3 successful runs
```

---

## 🔍 Where to Find Metrics

**Chunk count**:
```
Click: "CRITICAL: Ingest corpus" step
Look for: "Total chunks: XXX"
```

**Precision@5**:
```
Click: "Validate retrieval quality" step
Look for: "Precision@5: XX%"
```

**Concept count**:
```
Click: "Extract concepts" step
Look for: "Extracted XXX concepts"
```

**Test results**:
```
Click: "Run all CLI tests" step
Look for: "XX passed in X.XXs"
```

**Duration**:
```
Each step shows duration in parentheses
"CRITICAL: Ingest corpus (12m 34s)" → 12 minutes
```

---

## 🆘 Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| Can't see Actions tab | Need write permissions - ask repo owner |
| "Run workflow" disabled | Need write permissions - ask repo owner |
| Workflow not found | Commit and push `.github/workflows/weekly-full-rebuild.yml` |
| Corpus ingestion failed | Check fixture files exist in `fixtures/` |
| Tests failed | Run tests locally first, fix issues, re-run workflow |
| No artifacts | Check "Upload audit report" step succeeded |

---

## 💻 GitHub CLI Alternative

**If you prefer command line**:

```bash
# Install (macOS)
brew install gh

# Authenticate
gh auth login

# Trigger workflow
cd /path/to/research-kb
gh workflow run weekly-full-rebuild.yml

# Monitor
gh run watch $(gh run list --workflow=weekly-full-rebuild.yml --limit 1 --json databaseId --jq '.[0].databaseId')

# Download artifacts
gh run download $(gh run list --workflow=weekly-full-rebuild.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```

---

## 📁 Key Files

**Tracking**:
- `docs/VALIDATION_TRACKER.md` - Record results here
- `docs/MIGRATION_GRAPH_DEFAULT.md` - Migration guide
- `.github/ISSUE_TEMPLATE/validation_tracking.md` - GitHub issue template

**Scripts**:
- `scripts/run_local_validation.sh` - Test locally first
- `.github/workflows/weekly-full-rebuild.yml` - The workflow

**Documentation**:
- `docs/STEP_BY_STEP_VALIDATION_GUIDE.md` - Full detailed guide
- `docs/TRIGGER_VALIDATION_WORKFLOW.md` - Detailed trigger instructions

---

## 📅 Sample Timeline

```
Day 0:  Setup accounts, verify access
Day 1:  Run validation #1 (60 min)
        Record results (10 min)
Day 5:  Run validation #2 (60 min)
        Record results (10 min)
Day 10: Run validation #3 (60 min)
        Record results (10 min)
        Calculate aggregates (30 min)
        Create deployment PR (30 min)
Day 11: Merge PR and deploy
```

---

## 🎓 Success Criteria

**✅ APPROVED** if:
- 3/3 runs passed
- No blockers
- ≤1 warning per run
- Performance stable (±20%)

**⚠️ CONDITIONAL** if:
- 2/3 runs passed
- Only warnings (no blockers)
- Mitigation plan exists

**❌ REJECTED** if:
- Any blockers
- >1 run completely failed
- Performance regression >50%

---

## 📞 Get Help

**Resources**:
- Full guide: `docs/STEP_BY_STEP_VALIDATION_GUIDE.md`
- Troubleshooting: See "Troubleshooting" section in step-by-step guide
- GitHub Actions docs: https://docs.github.com/en/actions

**Support**:
- Create GitHub issue with `help-wanted` label
- Check workflow logs for specific errors
- Try local validation first: `./scripts/run_local_validation.sh`

---

**Last Updated**: 2025-12-02

**Print this page and keep it handy during your validation runs!**
