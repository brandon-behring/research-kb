# How to Trigger the Weekly Validation Workflow

This guide explains how to manually trigger the weekly full rebuild and validation workflow for validating the graph search default change.

---

## Prerequisites

- [ ] Access to the GitHub repository
- [ ] Permissions to trigger workflows (write access or above)
- [ ] GitHub CLI installed (optional, for command-line triggering)

---

## Method 1: GitHub Web UI (Easiest)

### Step 1: Navigate to the Workflow

1. Go to your GitHub repository
2. Click the **Actions** tab at the top
3. In the left sidebar, find and click **"Weekly Full Rebuild & Validation"**

### Step 2: Trigger the Workflow

1. Click the **"Run workflow"** button (top right, above the workflow runs list)
2. A dropdown will appear:
   - **Branch**: Select `main` (or your current working branch)
3. Click the green **"Run workflow"** button

### Step 3: Monitor the Workflow

1. The workflow will appear in the list below (it may take a few seconds)
2. Click on the workflow run to see details
3. Monitor the progress:
   - Green checkmarks = passed
   - Yellow circles = in progress
   - Red X = failed

### Step 4: Review Results

Once complete:
1. Click on the workflow run
2. Review each job:
   - `full-pipeline` - Main validation job
3. Click on individual steps to see logs
4. Download artifacts:
   - **weekly-audit-report** - Summary report
   - **Database dump** - Cached in GitHub Actions cache

---

## Method 2: GitHub CLI (Command Line)

### Install GitHub CLI

If not already installed:

```bash
# macOS
brew install gh

# Linux (Debian/Ubuntu)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Windows
winget install --id GitHub.cli
```

### Authenticate

```bash
gh auth login
# Follow the prompts
```

### Trigger the Workflow

```bash
# Navigate to repository directory
cd <PROJECT_ROOT>

# Trigger the workflow
gh workflow run weekly-full-rebuild.yml

# You should see:
# ✓ Created workflow_dispatch event for weekly-full-rebuild.yml at main
```

### Monitor the Workflow

```bash
# List recent runs
gh run list --workflow=weekly-full-rebuild.yml

# Watch a specific run (replace <run-id> with the ID from list)
gh run watch <run-id>

# View logs
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>
```

---

## Method 3: REST API (Advanced)

### Using curl

```bash
# Set your GitHub token
export GITHUB_TOKEN="your_personal_access_token"

# Trigger the workflow
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/repos/YOUR_USERNAME/research-kb/actions/workflows/weekly-full-rebuild.yml/dispatches \
  -d '{"ref":"main"}'
```

---

## What Happens During the Workflow

### Job: `full-pipeline`

**Estimated Duration**: 60 minutes

**Steps**:

1. **Setup** (2 min)
   - Checkout code
   - Set up Python 3.11
   - Start PostgreSQL with pgvector

2. **Install Dependencies** (1 min)
   - Install all packages
   - Install pytest and dependencies

3. **Database Setup** (1 min)
   - Drop and recreate database (clean slate)
   - Apply schema
   - Apply migrations

4. **Start Embedding Server** (1 min)
   - Start BGE-large-en-v1.5 embedding server
   - Verify health check

5. **Setup Ollama** (5 min, optional)
   - Install Ollama
   - Start server
   - Pull llama3.1:8b model

6. **CRITICAL: Ingest Corpus** (10-15 min)
   - Ingest textbooks + papers
   - Target: ~500 chunks
   - **BLOCKER** if fails

7. **Validate Retrieval** (2 min)
   - Run known-answer tests
   - Target: Precision@5 ≥90%

8. **Extract Concepts** (15-20 min)
   - Extract concepts from chunks (limit 1000)
   - Target: ≥100 concepts
   - May timeout (acceptable)

9. **Validate Concept Quality** (2 min)
   - Compare to seed concepts
   - Target: Recall ≥70%

10. **Validate Knowledge Graph** (1 min)
    - Check graph structure
    - Verify relationships exist

11. **Run All Tests** (5 min)
    - Script tests (24 tests)
    - CLI tests (47 tests)
    - Must all pass

12. **Cache Database** (2 min)
    - Dump database to SQL
    - Upload to GitHub Actions cache
    - Used by daily validation

13. **Generate Report** (1 min)
    - Create audit report
    - Upload as artifact

---

## Monitoring Checklist

During the workflow run, monitor these key metrics:

### Critical (Must Pass)

- [ ] ✅ **Corpus ingestion**: ≥450 chunks
- [ ] ✅ **Retrieval Precision@5**: ≥90%
- [ ] ✅ **All tests pass**: 71/71 tests
- [ ] ✅ **Database cached**: Successfully uploaded

### Important (Should Pass)

- [ ] ⚠️ **Concept extraction**: ≥100 concepts
- [ ] ⚠️ **Seed recall**: ≥70%
- [ ] ⚠️ **Extraction time**: <20 minutes

### Optional (May Fail)

- [ ] ℹ️ **Ollama installation**: May fail in CI
- [ ] ℹ️ **Full concept extraction**: May timeout

---

## Reading the Workflow Logs

### Success Indicators

Look for these messages:

```
✓ Created workflow_dispatch event
✓ Corpus ingested in XXXs
✓ Chunk count: XXX (≥450 target)
✓ Retrieval validation passed
✓ Concept extraction completed
✓ All tests passed
✓ Database cached successfully
```

### Failure Indicators

Watch for these errors:

```
✗ Corpus ingestion failed
✗ Some tests failed
✗ Database cache upload failed
Error: ...
```

### Warnings (Acceptable)

These are OK:

```
⚠ Ollama installation failed
⚠ Concept extraction timed out
⚠ Seed concept validation completed with warnings
```

---

## After Workflow Completion

### 1. Download Artifacts

```bash
# Via GitHub CLI
gh run download <run-id>

# Or via web UI:
# Go to workflow run → Artifacts section → Download
```

### 2. Review Audit Report

Open `weekly-audit-report/audit_report.md`:

```markdown
# Weekly Validation Report
Date: 2025-XX-XX
Status: success

## Build Summary
- Run number: XXX
- Status: success
```

### 3. Update Validation Tracker

Edit `docs/VALIDATION_TRACKER.md`:

```markdown
### Run 1: 2025-XX-XX

**Status**: ✅ Complete

**Quality Gates**:
- [x] Corpus ingestion successful (~500 chunks)
- [x] Retrieval validation passed (Precision@K ≥90%)
...

**Performance Metrics**:
- Query latency p50: XXX ms
- Query latency p95: XXX ms
...
```

### 4. Record in GitHub Issue

Create or update a tracking issue:

```markdown
## Graph Search Default Validation

Tracking 3 validation runs before enabling graph search by default.

### Run 1 - 2025-XX-XX
- Status: ✅ Passed
- Workflow: https://github.com/.../actions/runs/XXX
- Notes: All gates passed, no issues

### Run 2 - TBD
...
```

---

## Troubleshooting

### "Workflow not found"

**Issue**: Cannot find the workflow in the list.

**Solution**:
1. Ensure the workflow file exists: `.github/workflows/weekly-full-rebuild.yml`
2. Check that the workflow file is in the `main` branch
3. Verify the workflow has been committed and pushed

### "Run workflow button disabled"

**Issue**: Button is grayed out.

**Solution**:
1. Check you have write access to the repository
2. Ensure you're on a branch where the workflow exists
3. Try refreshing the page

### "Workflow failed immediately"

**Issue**: Workflow fails in setup steps.

**Solution**:
1. Check PostgreSQL service health
2. Verify Python 3.11 is available
3. Review dependency installation logs

### "Corpus ingestion failed"

**Issue**: Step 6 fails to ingest corpus.

**Solution**:
1. Check fixture files exist in `fixtures/` directory
2. Verify embedding server starts successfully
3. Review ingestion logs for specific errors

### "Concept extraction timed out"

**Issue**: Step 8 times out after 18 minutes.

**Solution**:
- This is acceptable (marked as `continue-on-error: true`)
- The workflow will continue and mark this as a warning
- Partial concepts are still useful for validation

---

## Local Pre-Validation (Optional but Recommended)

Before triggering the CI workflow, run local validation:

```bash
# Make sure you're in the repository root
cd <PROJECT_ROOT>

# Run the local validation script
./scripts/run_local_validation.sh
```

This will:
- Test all steps locally
- Identify issues before CI
- Save CI minutes
- Provide faster feedback

**Estimated time**: 15-30 minutes (depending on Ollama)

---

## Cost Considerations

**GitHub Actions Free Tier**: 2000 minutes/month

**Weekly workflow**: ~60 minutes per run

**3 validation runs**: ~180 minutes (9% of free tier)

**Recommendation**: Space runs 3-7 days apart to ensure different conditions.

---

## Next Steps After 3 Successful Runs

1. ✅ Update `docs/VALIDATION_TRACKER.md` with final aggregate results
2. ✅ Update `docs/MIGRATION_GRAPH_DEFAULT.md` status to "Validated"
3. ✅ Create PR for Phase 3C
4. ✅ Merge to production
5. ✅ Monitor production metrics for first week

---

## Questions?

- **Workflow not running?** Check the troubleshooting section above
- **Results unclear?** Review the audit report artifact
- **Need help?** Create a GitHub issue

---

**Last Updated**: 2025-12-02
