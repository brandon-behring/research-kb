# Step-by-Step Validation Guide

**Complete beginner-friendly guide to running the 3 validation workflows.**

This guide assumes you're starting from scratch and walks you through every step, including account setup.

---

## Table of Contents

0. [Phase 0: Repository Setup (First-Time Only)](#phase-0-repository-setup-first-time-only)
1. [Prerequisites Setup](#prerequisites-setup)
2. [Method 1: GitHub Web UI (Easiest)](#method-1-github-web-ui-easiest)
3. [Method 2: GitHub CLI (Optional)](#method-2-github-cli-optional)
4. [Recording Results](#recording-results)
5. [Repeating for Runs 2 and 3](#repeating-for-runs-2-and-3)
6. [Making the Final Decision](#making-the-final-decision)
7. [Troubleshooting](#troubleshooting)

---

## Phase 0: Repository Setup (First-Time Only)

**Skip this section if your repository is already on GitHub.**

This section covers the one-time setup to get your local code onto GitHub so the validation workflows can run.

---

### Step 0.1: Verify GitHub CLI is Installed

**The easiest way to create the repository is using GitHub CLI (`gh`).**

```bash
# Check if gh is installed
gh --version

# If not installed:
# macOS: brew install gh
# Linux: See https://github.com/cli/cli/blob/trunk/docs/install_linux.md
# Windows: winget install --id GitHub.cli
```

**If you don't have gh**, you can create the repository via the GitHub web UI (see Step 0.4 Option A).

---

### Step 0.2: Authenticate with GitHub

```bash
# Login to GitHub CLI
gh auth login

# Follow prompts:
# 1. What account? → GitHub.com
# 2. Preferred protocol? → HTTPS
# 3. Authenticate Git? → Yes
# 4. How to authenticate? → Login with web browser

# Verify authentication
gh auth status
```

**You should see**: `✓ Logged in to github.com as YOUR_USERNAME`

---

### Step 0.3: Verify .gitignore Exists

**Before initializing git, ensure you have a `.gitignore` to exclude unnecessary files.**

```bash
cd <PROJECT_ROOT>

# Check if .gitignore exists
ls -la .gitignore
```

**If it doesn't exist**, create it with these contents:

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv/
*.egg-info/
dist/
build/

# Testing/Linting
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project-specific
fixtures/*              # Large PDFs - source separately
!fixtures/samples/      # But keep sample PDFs for CI
.extraction_checkpoint.json
*.log
.env
*.sql                   # Database dumps

# Keep these directories tracked
!.github/
!.claude/
EOF
```

**Why exclude `fixtures/`?**
- Full corpus PDFs can be 100MB+, too large for git
- `fixtures/samples/` contains small test PDFs for CI
- See `fixtures/samples/README.md` for details on obtaining full corpus

---

### Step 0.4: Initialize Git Repository

```bash
cd <PROJECT_ROOT>

# Initialize git (if not already)
git init

# Ensure main branch (not master)
git branch -M main

# Verify
git status
```

**You should see**: `On branch main` and a list of untracked files.

---

### Step 0.5: Create Remote Repository on GitHub

#### Option A: GitHub CLI (Recommended)

```bash
# Create private repository with metadata
gh repo create brandonmbehring-dev/research-kb \
  --private \
  --description "Semantic search system for causal inference literature with graph-boosted retrieval" \
  --add-readme=false

# Add topics (run separately)
gh repo edit brandonmbehring-dev/research-kb \
  --add-topic python \
  --add-topic semantic-search \
  --add-topic causal-inference \
  --add-topic knowledge-graph \
  --add-topic pgvector
```

#### Option B: GitHub Web UI

1. Go to [https://github.com/new](https://github.com/new)
2. **Repository name**: `research-kb`
3. **Description**: `Semantic search system for causal inference literature with graph-boosted retrieval`
4. **Visibility**: Private
5. **DO NOT** check "Add a README file" (we have one locally)
6. Click **"Create repository"**

---

### Step 0.6: Add Remote and Push

```bash
# Add the remote origin
git remote add origin https://github.com/brandonmbehring-dev/research-kb.git

# Stage all files
git add .

# Initial commit
git commit -m "Initial commit: research-kb semantic search system

- Hybrid search with FTS + vector + knowledge graph
- PostgreSQL + pgvector backend
- CLI for querying and management
- CI workflows for validation"

# Push to GitHub
git push -u origin main
```

**Expected output**: Files uploading, then `Branch 'main' set up to track remote branch 'main' from 'origin'.`

---

### Step 0.7: Verify Workflows Appear on GitHub

1. Go to `https://github.com/brandonmbehring-dev/research-kb`
2. Click the **"Actions"** tab
3. **Wait 30-60 seconds** for GitHub to detect workflows
4. You should see 3 workflows:
   - `PR Checks (Fast)`
   - `Daily Validation (Cached DB)`
   - `Weekly Full Rebuild & Validation`

**If workflows don't appear**:
- Verify `.github/workflows/` was pushed: Check Code tab → `.github/workflows/`
- Wait another minute and refresh
- Check that Actions are enabled (Settings → Actions → General)

**✓ Phase 0 Complete! Your repository is now on GitHub.**

---

## Prerequisites Setup

### Step 1: Verify GitHub Account

**You need a GitHub account to access the repository and trigger workflows.**

#### If you DON'T have a GitHub account:

1. Go to [https://github.com/signup](https://github.com/signup)
2. Enter your email address
3. Create a password
4. Choose a username
5. Verify your email
6. Complete the setup

#### If you HAVE a GitHub account:

1. Go to [https://github.com](https://github.com)
2. Click "Sign in" (top right)
3. Enter your credentials

**✓ You should now be logged into GitHub**

---

### Step 2: Verify Repository Access

**You need access to the research-kb repository.**

#### Check if you have access:

1. While logged into GitHub, navigate to your repository URL:
   ```
   https://github.com/YOUR_USERNAME/research-kb
   ```
   (Replace `YOUR_USERNAME` with your actual GitHub username)

2. You should see:
   - Repository name at the top
   - Code, Issues, Pull requests tabs
   - **Actions** tab (this is what we need)

#### If you see "404 - Page not found":

The repository might be:
- Private (and you don't have access)
- Not created yet
- Under a different username/organization

**Solution**:
- Check if it's under an organization: `https://github.com/ORGANIZATION_NAME/research-kb`
- Ask the repository owner to add you as a collaborator
- Or verify the repository name is correct

#### If you see the repository but no "Actions" tab:

**Solution**:
- You need **write access** to trigger workflows
- Ask the repository owner to give you "Write" or "Admin" permissions

**How to check your permissions**:
1. Click "Settings" tab (only visible if you have permissions)
2. If you can't see Settings, you only have read access
3. Contact the repository owner to upgrade your permissions

---

### Step 3: Verify Actions Are Enabled

**GitHub Actions must be enabled for the repository.**

1. Go to your repository
2. Click the **"Actions"** tab (top menu bar)
3. You should see one of:
   - **Option A**: A list of workflows (good! Actions are enabled)
   - **Option B**: A message saying "Actions are disabled" with a button to enable them

#### If Actions are disabled:

1. Click the **"I understand my workflows, go ahead and enable them"** button
2. You should now see the workflows list

#### If you don't have permission to enable Actions:

**Solution**: Ask the repository owner/admin to:
1. Go to repository **Settings**
2. Click **Actions** (left sidebar)
3. Under "Actions permissions", select **"Allow all actions and reusable workflows"**
4. Click **Save**

**✓ GitHub Actions should now be enabled**

---

### Step 4: Verify the Workflow Exists

**The weekly validation workflow must exist in the repository.**

1. Go to repository → **Actions** tab
2. Look in the left sidebar under "All workflows"
3. You should see: **"Weekly Full Rebuild & Validation"**

#### If you DON'T see the workflow:

**Check if the workflow file exists**:
1. Go to repository → **Code** tab
2. Click `.github/` folder
3. Click `workflows/` folder
4. Look for `weekly-full-rebuild.yml`

#### If the file doesn't exist:

The workflow file hasn't been committed to the repository yet.

**Solution**:
```bash
# On your local machine
cd <PROJECT_ROOT>

# Check if file exists locally
ls -la .github/workflows/weekly-full-rebuild.yml

# If it exists, commit and push it
git add .github/workflows/weekly-full-rebuild.yml
git commit -m "Add weekly validation workflow"
git push origin main

# Wait 30 seconds, then refresh GitHub Actions page
```

**✓ The workflow should now appear in the Actions tab**

---

## Method 1: GitHub Web UI (Easiest)

**This method requires no additional software. Just a web browser.**

---

### Run 1: Triggering the First Validation

#### Step 1: Navigate to Actions

1. Go to your repository on GitHub: `https://github.com/YOUR_USERNAME/research-kb`
2. Click the **"Actions"** tab in the top menu
   - It's between "Pull requests" and "Projects"
3. You should see a list of workflows on the left sidebar

**What you should see**:
```
All workflows
├─ Weekly Full Rebuild & Validation
├─ Daily Validation (Cached DB)
└─ PR Checks (Fast)
```

#### Step 2: Select the Workflow

1. In the left sidebar, click **"Weekly Full Rebuild & Validation"**
   - The name might be slightly different
   - Look for "weekly" and "validation" in the name

**What you should see**:
- The main area now shows previous runs (if any)
- A blue button labeled **"Run workflow"** in the top-right area
- If you don't see the button, you need write permissions (see Prerequisites)

#### Step 3: Trigger the Workflow

1. Click the blue **"Run workflow"** button (top right)
2. A dropdown appears with:
   - **"Use workflow from"**: Shows branch selector
   - A dropdown menu (probably showing "Branch: main")
   - A green **"Run workflow"** button

3. **Select the branch**:
   - Click the dropdown under "Use workflow from"
   - Select **"main"** (or your current working branch)
   - If you only see "main", that's perfect

4. Click the green **"Run workflow"** button at the bottom of the dropdown

**What happens next**:
- The dropdown closes
- After 3-10 seconds, a new workflow run appears in the list below
- It will show a yellow dot (⚫) meaning "in progress"

**If nothing happens after 30 seconds**:
- Refresh the page
- The run should appear

#### Step 4: Monitor the Workflow Run

**The workflow takes approximately 60 minutes to complete.**

1. Click on the newly created workflow run
   - It will have today's date and time
   - Shows your username
   - Has a yellow spinning circle (in progress)

**What you should see**:
```
Weekly Full Rebuild & Validation
#123 · main · Triggered via workflow_dispatch · <date> · <duration>

Jobs
└─ full-pipeline (running) ⚫
```

2. Click on **"full-pipeline"** to see detailed steps

**What you should see**:
```
full-pipeline
├─ Set up job ✓ (completed)
├─ Checkout ✓ (completed)
├─ Set up Python ✓ (completed)
├─ Install dependencies ⚫ (running)
├─ Drop and recreate database (pending)
├─ Run database migrations (pending)
├─ Start embedding server (pending)
├─ Set up Ollama (pending)
├─ CRITICAL: Ingest corpus (~500 chunks) (pending)
├─ Validate retrieval quality (pending)
├─ Extract concepts (pending)
├─ Validate concept extraction quality (pending)
├─ Validate knowledge graph (pending)
├─ Run all script tests (pending)
├─ Run all CLI tests (pending)
├─ Cache database dump (pending)
├─ Upload database cache (pending)
├─ Generate audit report (pending)
└─ Upload audit report (pending)
```

3. **Watch the steps complete**:
   - ✓ Green checkmark = completed successfully
   - ⚫ Yellow circle = in progress
   - ✗ Red X = failed
   - ⊘ Gray circle = skipped

#### Step 5: Key Steps to Monitor

**These are the critical steps to watch:**

**1. Ingest corpus** (~10-15 minutes):
   - Click on "CRITICAL: Ingest corpus (~500 chunks)"
   - Look for: "Total chunks: XXX"
   - **MUST BE ≥450** chunks (blocker if less)

**2. Validate retrieval quality** (~2 minutes):
   - Click on "Validate retrieval quality"
   - Look for: "Precision@5: XX%"
   - **MUST BE ≥90%** (blocker if less)

**3. Extract concepts** (~15-20 minutes):
   - Click on "Extract concepts (limit 1000 chunks)"
   - Look for: "Extracted XXX concepts"
   - **SHOULD BE ≥100** concepts (warning if less, not a blocker)
   - May timeout - this is OK (marked as warning)

**4. Run all tests** (~5 minutes):
   - Click on "Run all CLI tests"
   - Look for: "XX passed"
   - **MUST BE 47 passed, 0 failed** (blocker if any failures)
   - Click on "Run all script tests"
   - Look for: "XX passed"
   - **MUST BE 24 passed, 0 failed** (blocker if any failures)

**5. Cache database dump** (~2 minutes):
   - Click on "Cache database dump for daily validation"
   - Look for: "pg_dump ... complete"
   - Click on "Upload database cache"
   - Look for: "Cache saved successfully"
   - **MUST succeed** (blocker if fails)

#### Step 6: Wait for Completion

**Total time: ~60 minutes**

You can:
- **Option A**: Leave the tab open and check periodically
- **Option B**: Close the tab and come back in an hour
- **Option C**: Enable email notifications:
  1. Click your profile (top right)
  2. Settings → Notifications
  3. Check "Actions" under "Email notifications"

**How to know it's done**:
- Refresh the workflow run page
- The yellow spinning circle changes to:
  - ✓ Green checkmark = success
  - ✗ Red X = failure

#### Step 7: Review the Results

**Once the workflow completes:**

1. Go back to the workflow run page
2. Check the overall status:
   - ✓ **Green checkmark** = All steps passed (good!)
   - ✗ **Red X** = Some steps failed (investigate)

3. **If GREEN (success)**:
   - Scroll down to "Artifacts" section
   - You should see: "weekly-audit-report"
   - Click to download (it's a ZIP file)

4. **If RED (failure)**:
   - Click on "full-pipeline" job
   - Find the first step with a red ✗
   - Click on it to see error logs
   - See [Troubleshooting](#troubleshooting) section

#### Step 8: Download Artifacts

**Artifacts are generated files from the workflow (reports, logs).**

1. On the workflow run page, scroll down to the bottom
2. Look for the **"Artifacts"** section
3. You should see:
   - **weekly-audit-report** (small ZIP file)

4. Click on "weekly-audit-report" to download
5. Unzip the file on your computer
6. Open `audit_report.md` in a text editor

**What you should see in the report**:
```markdown
# Weekly Validation Report
Date: 2025-XX-XX
Status: success

## Build Summary
- Run number: XXX
- Status: success
```

**Save this file** - you'll need it for recording results.

---

### Recording Results (After Run 1)

**Now we need to record what happened in the tracking documents.**

#### Option A: Edit on GitHub Web UI

1. Go to repository → **Code** tab
2. Navigate to `docs/VALIDATION_TRACKER.md`
3. Click the **pencil icon** (✏️) to edit
4. Find the **"Run 1"** section
5. Fill in the details (see template below)
6. Scroll to bottom, click **"Commit changes"**

#### Option B: Edit Locally and Push

```bash
# On your local machine
cd <PROJECT_ROOT>

# Open the file in your editor
vim docs/VALIDATION_TRACKER.md
# or
code docs/VALIDATION_TRACKER.md
# or
nano docs/VALIDATION_TRACKER.md
```

#### What to Fill In

**Find the "Run 1" section** and update it:

```markdown
### Run 1: 2025-12-02  ← Change to today's date

**Status**: ✅ Complete  ← Change from "Pending" to "Complete"

**Trigger**: Manual via GitHub Actions UI

**Quality Gates**:
- [x] Corpus ingestion successful (~500 chunks)  ← Check if ≥450 chunks
- [x] Retrieval validation passed (Precision@K ≥90%)  ← Check if ≥90%
- [x] Concept extraction completed (≥100 concepts)  ← Check if ≥100
- [x] Seed concept validation passed (Recall ≥70%)  ← Check if ≥70%
- [x] Graph validation passed (relationships exist)  ← Check if passed
- [x] All CLI tests passed (47 tests)  ← Check if 47/47
- [x] All script tests passed (24 tests)  ← Check if 24/24
- [x] Database cached successfully  ← Check if cache uploaded

**Performance Metrics**:
- Query latency p50: ___ ms  ← Fill from logs if available
- Query latency p95: ___ ms  ← Fill from logs if available
- Query latency p99: ___ ms  ← Fill from logs if available
- Corpus ingestion time: 12 minutes  ← Fill from workflow duration
- Concept extraction time: 18 minutes  ← Fill from workflow duration

**Issues Found**: None  ← Or list any issues

**Notes**:
- All steps completed successfully
- No warnings or errors
- Database cached for daily validation
```

#### Where to Find Metrics

**To find the metrics**:

1. Go to workflow run page
2. Click on "full-pipeline" job
3. For each step, note the duration shown in parentheses:
   - "CRITICAL: Ingest corpus (12m 34s)" → 12 minutes
   - "Extract concepts (18m 12s)" → 18 minutes

4. For chunk count:
   - Click on "CRITICAL: Ingest corpus"
   - Look for line: "Total chunks: 523"

5. For concept count:
   - Click on "Extract concepts"
   - Look for line: "Extracted 142 concepts"

6. For test results:
   - Click on "Run all CLI tests"
   - Look for line: "47 passed in 0.31s"

#### Commit Your Changes

**If editing on GitHub Web UI**:
1. Scroll to bottom
2. Enter commit message: "docs: record validation run 1 results"
3. Click "Commit changes"

**If editing locally**:
```bash
# Save the file, then:
git add docs/VALIDATION_TRACKER.md
git commit -m "docs: record validation run 1 results"
git push origin main
```

**✓ Run 1 is now documented!**

---

### Creating a Tracking Issue (Optional but Recommended)

**A GitHub issue helps track progress publicly.**

#### Step 1: Create the Issue

1. Go to repository → **Issues** tab
2. Click green **"New issue"** button
3. Look for the template: **"Graph Search Default Validation Tracking"**
4. Click **"Get started"** next to it

**If you don't see the template**:
1. Click **"New issue"** anyway
2. Copy the template from `.github/ISSUE_TEMPLATE/validation_tracking.md`
3. Paste into the issue body

#### Step 2: Fill in the Issue

1. **Title**: "Validation: Graph Search as Default (Phase 3C)"
2. **Labels**: Add `validation`, `phase-3c`, `testing`
3. **Body**: Update the Run 1 section with your results
4. Click **"Submit new issue"**

#### Step 3: Update the Issue After Each Run

After Run 2 and Run 3:
1. Go to the issue
2. Click "Edit" (on the issue description)
3. Update the relevant run section
4. Click "Update comment"

**✓ You now have a public tracking issue!**

---

## Waiting Period (3-7 Days)

**Why wait?**
- Tests under different conditions
- Catches intermittent issues
- Proves consistency

**What to do during the wait**:
- Monitor the repository
- Check if daily validation runs succeed
- Review the results from Run 1
- Prepare for Run 2

**When to run Run 2**:
- Minimum: 3 days after Run 1
- Maximum: 7 days after Run 1
- Recommended: 5 days (middle of the range)

---

## Repeating for Runs 2 and 3

**The process is identical to Run 1.**

### For Run 2:

1. Wait 3-7 days after Run 1
2. Go to repository → **Actions** → **"Weekly Full Rebuild & Validation"**
3. Click **"Run workflow"**
4. Select **"main"** branch
5. Click **"Run workflow"**
6. Monitor the run (~60 minutes)
7. Download artifacts
8. Record results in `docs/VALIDATION_TRACKER.md` under "Run 2"
9. Update GitHub issue (if created)

### For Run 3:

1. Wait 3-7 days after Run 2
2. Repeat all steps above
3. Record results under "Run 3"
4. Update GitHub issue

**✓ After Run 3, you'll have 3 complete validation runs!**

---

## Making the Final Decision

**After all 3 runs are complete:**

### Step 1: Calculate Aggregate Results

Edit `docs/VALIDATION_TRACKER.md` and fill in the "Aggregate Results" section:

```markdown
## Aggregate Results

**Runs**: 3/3 complete

**Success Rate**: 100% (3/3 passed)  ← Or 67% (2/3 passed), etc.

**Average Metrics**:
- Query latency p50: N/A  ← Average if you have data
- Query latency p95: N/A
- Query latency p99: N/A
- Corpus ingestion time: 12 minutes  ← Average of 3 runs
- Concept extraction time: 18 minutes  ← Average of 3 runs

**Precision@K**: 95%  ← Average from retrieval validation

**Concept Recall**: 78%  ← Average from seed validation
```

### Step 2: Apply Decision Criteria

**Review the criteria in the tracker**:

✅ **APPROVE for production** if:
- All 3 runs completed successfully
- No blockers in any run
- ≤1 warning per run
- Performance stable across runs (±20%)

⚠️ **CONDITIONAL APPROVAL** if:
- 2/3 runs successful with only warnings
- Warnings are understood and documented
- Mitigation plan exists

❌ **REJECT** if:
- Any run has blockers
- >1 run failed completely
- Performance regressions >50%
- Systematic issues across runs

### Step 3: Document the Decision

In `docs/VALIDATION_TRACKER.md`, add a decision section:

```markdown
## Final Decision

**Date**: 2025-XX-XX

**Decision**: ✅ APPROVED for production

**Rationale**:
- All 3 runs completed successfully
- No blockers encountered
- All quality gates passed
- Performance stable across runs
- Corpus: 501, 498, 505 chunks (stable)
- Precision@K: 94%, 95%, 93% (>90% threshold)
- All tests passed in all runs

**Next Steps**:
1. Update MIGRATION_GRAPH_DEFAULT.md status to "Validated"
2. Create PR for Phase 3C deployment
3. Merge to production
4. Monitor production metrics for first week
```

### Step 4: Update Migration Guide

Edit `docs/MIGRATION_GRAPH_DEFAULT.md`:

Find the "Status" line at the top:
```markdown
**Status**: ⚠️ **Pending Validation** (requires ≥3 successful weekly runs)
```

Change to:
```markdown
**Status**: ✅ **Validated** (3/3 runs successful as of 2025-XX-XX)
```

Update the timeline table at the bottom:
```markdown
| Phase | Date | Status |
|-------|------|--------|
| **Implementation** | 2025-12-02 | ✅ Complete |
| **Test Validation** | 2025-12-02 | ✅ Complete |
| **CI Validation** | 2025-12-XX | ✅ Complete (3/3 runs) |
| **Manual Testing** | 2025-12-XX | ✅ Complete |
| **Production Deployment** | 2025-XX-XX | ✅ Ready |
```

### Step 5: Create Deployment PR

```bash
# Create a new branch
git checkout -b phase-3c-production-ready

# Commit the documentation updates
git add docs/
git commit -m "docs: validate graph search default (3 successful runs)

- All 3 validation runs completed successfully
- No blockers encountered
- All quality gates passed
- Ready for production deployment

See docs/VALIDATION_TRACKER.md for details."

# Push the branch
git push origin phase-3c-production-ready
```

### Step 6: Create PR on GitHub

1. Go to repository on GitHub
2. You should see a banner: "phase-3c-production-ready had recent pushes"
3. Click **"Compare & pull request"**
4. **Title**: "Phase 3C: Enable Graph Search by Default (Validated)"
5. **Description**:
   ```markdown
   ## Summary

   Enables graph-boosted search as the default behavior for the `query` command.

   ## Validation

   ✅ **3/3 validation runs successful**

   - Run 1: 2025-XX-XX - ✅ All gates passed
   - Run 2: 2025-XX-XX - ✅ All gates passed
   - Run 3: 2025-XX-XX - ✅ All gates passed

   ## Quality Gates

   - ✅ Corpus ingestion: 501, 498, 505 chunks (≥450 threshold)
   - ✅ Retrieval Precision@5: 94%, 95%, 93% (≥90% threshold)
   - ✅ All tests passed: 71/71 in all runs
   - ✅ Database cached successfully in all runs
   - ✅ Concept extraction: 142, 138, 145 concepts (≥100 threshold)
   - ✅ Seed recall: 78%, 76%, 79% (≥70% threshold)

   ## Performance

   - Average corpus ingestion: 12 minutes
   - Average concept extraction: 18 minutes
   - Performance stable across runs (±5%)

   ## Documentation

   - See `docs/VALIDATION_TRACKER.md` for detailed results
   - See `docs/MIGRATION_GRAPH_DEFAULT.md` for migration guide

   ## Breaking Changes

   - Default behavior changes from non-graph to graph search
   - Users can opt-out with `--no-graph` flag
   - Graceful fallback if concepts not extracted

   ## Rollback Plan

   If issues arise:
   - Revert this PR
   - Or use `--no-graph` in scripts/documentation
   - See MIGRATION_GRAPH_DEFAULT.md for details
   ```

6. **Reviewers**: Add team members (if applicable)
7. **Labels**: Add `phase-3c`, `validated`, `production-ready`
8. Click **"Create pull request"**

### Step 7: Merge and Deploy

**After PR approval**:

1. Click **"Merge pull request"**
2. Click **"Confirm merge"**
3. Delete the branch (GitHub will prompt)
4. Monitor production for the first week

**Production monitoring**:
- Check error rates
- Monitor query latency
- Watch for user complaints
- Review daily validation runs

**✓ Graph search is now the default!**

---

## Method 2: GitHub CLI (Optional)

**This method is faster but requires setup.**

### Installing GitHub CLI

#### macOS:
```bash
brew install gh
```

#### Linux (Debian/Ubuntu):
```bash
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

#### Windows:
```bash
winget install --id GitHub.cli
```

### Authenticating

```bash
# Run authentication
gh auth login

# Follow the prompts:
# 1. What account do you want to log into? → GitHub.com
# 2. What is your preferred protocol? → HTTPS
# 3. Authenticate Git with your GitHub credentials? → Yes
# 4. How would you like to authenticate? → Login with a web browser

# You'll get a one-time code
# Press Enter to open browser
# Paste the code in the browser
# Authorize GitHub CLI

# You should see: ✓ Authentication complete
```

### Triggering Workflow via CLI

```bash
# Navigate to repository
cd <PROJECT_ROOT>

# Trigger the workflow
gh workflow run weekly-full-rebuild.yml

# You should see:
# ✓ Created workflow_dispatch event for weekly-full-rebuild.yml at main
```

### Monitoring via CLI

```bash
# Get the latest run ID
RUN_ID=$(gh run list --workflow=weekly-full-rebuild.yml --limit 1 --json databaseId --jq '.[0].databaseId')

# Watch the run (live updates)
gh run watch $RUN_ID

# Or view current status
gh run view $RUN_ID

# View logs
gh run view $RUN_ID --log

# Download artifacts
gh run download $RUN_ID
```

**The artifacts will be downloaded to your current directory.**

---

## Troubleshooting

### Problem: "Actions are disabled for this repository"

**Solution**:
1. Go to repository → **Settings** tab
2. Click **Actions** in left sidebar
3. Under "Actions permissions", select "Allow all actions and reusable workflows"
4. Click **Save**

If you don't see Settings tab: You need admin permissions. Contact repository owner.

---

### Problem: "Workflow not found"

**Symptoms**:
- The workflow doesn't appear in the Actions tab
- "Run workflow" button doesn't appear

**Solution**:
1. Check if workflow file exists:
   - Go to Code → `.github/workflows/weekly-full-rebuild.yml`
   - If missing, the file hasn't been pushed to GitHub

2. Commit and push the workflow file:
   ```bash
   cd <PROJECT_ROOT>
   git add .github/workflows/weekly-full-rebuild.yml
   git commit -m "Add weekly validation workflow"
   git push origin main
   ```

3. Wait 30 seconds and refresh the Actions page

---

### Problem: "Run workflow button is disabled"

**Symptoms**:
- You can see the workflow
- But the "Run workflow" button is grayed out

**Solution**: You need **write** permissions to the repository.

**To check your permissions**:
1. Go to repository
2. Click **Settings** tab
3. If you can't see Settings → you only have read access
4. Ask repository owner to change your role to "Write" or "Admin"

**Repository owner instructions**:
1. Go to repository → **Settings**
2. Click **Collaborators** (left sidebar)
3. Find the user
4. Change role from "Read" to "Write"
5. Click **Save**

---

### Problem: "Corpus ingestion failed"

**Symptoms**:
- Step "CRITICAL: Ingest corpus" shows red X
- Error message about fixture files not found

**Solution**:
1. Check if fixture files exist:
   ```bash
   ls -la fixtures/textbooks/
   ls -la fixtures/papers/
   ```

2. If missing, you need to add the PDF fixtures
3. The workflow expects files defined in `scripts/ingest_corpus.py`
4. Either add the PDFs or update the script to use available files

---

### Problem: "Embedding server failed to start"

**Symptoms**:
- Step "Start embedding server" shows red X
- Error about connection refused

**Solution**:
1. Check the logs for port conflicts
2. Embedding server uses unix socket `/tmp/research_kb_embed.sock`
3. In CI, this should work automatically
4. If consistently failing, check if service dependencies are correct in workflow

---

### Problem: "All tests failed"

**Symptoms**:
- "Run all CLI tests" or "Run all script tests" shows red X
- Multiple test failures

**Solution**:
1. Click on the failing step
2. Look for the first failed test
3. Read the error message
4. Common issues:
   - Database connection issues
   - Missing dependencies
   - Environment variable not set

5. Fix the issue locally first:
   ```bash
   # Test locally
   pytest packages/cli/tests/ -v
   pytest tests/scripts/ -v
   ```

6. Once tests pass locally, re-run the workflow

---

### Problem: "Workflow runs but no artifacts"

**Symptoms**:
- Workflow completed
- But "Artifacts" section is empty

**Solution**:
1. Artifacts are only created if certain steps succeed
2. Check if "Generate audit report" step passed
3. Check if "Upload audit report" step passed
4. If these failed, the artifact wasn't created
5. Review those step logs for errors

---

### Problem: "Cannot download artifacts"

**Symptoms**:
- Artifact appears in list
- But clicking doesn't download

**Solution**:
1. Try a different browser
2. Check if browser is blocking downloads
3. Try GitHub CLI instead:
   ```bash
   gh run download <run-id>
   ```

---

### Problem: "Concept extraction timed out"

**Symptoms**:
- "Extract concepts" step shows yellow warning
- Says "timed out after 18 minutes"

**Solution**:
- **This is OK!** Marked as `continue-on-error: true`
- Partial concept extraction is acceptable
- Not a blocker for validation
- Mark as "warning" in tracking document

---

### Problem: "gh: command not found"

**Symptoms**: Trying to use GitHub CLI but command not found

**Solution**:
```bash
# macOS
brew install gh

# Linux
# See installation instructions in Method 2 section above

# Windows
winget install --id GitHub.cli

# After installation
gh auth login
```

---

### Problem: "gh auth login fails"

**Symptoms**: Cannot authenticate with GitHub CLI

**Solution**:
1. Try browser authentication:
   ```bash
   gh auth login
   # Select: Login with a web browser
   ```

2. If browser doesn't open automatically:
   - Copy the one-time code shown
   - Open browser manually
   - Go to: https://github.com/login/device
   - Paste code

3. If still failing, use personal access token:
   - Go to GitHub → Settings → Developer settings → Personal access tokens
   - Generate new token (classic)
   - Select scopes: `repo`, `workflow`
   - Copy token
   - ```bash
     gh auth login
     # Select: Paste an authentication token
     # Paste your token
     ```

---

## Summary Checklist

**Before starting**:
- [ ] GitHub account created and logged in
- [ ] Repository access verified (write permissions)
- [ ] Actions enabled in repository
- [ ] Workflow file exists in `.github/workflows/`

**For each validation run**:
- [ ] Trigger workflow via UI or CLI
- [ ] Monitor run for ~60 minutes
- [ ] Verify all critical steps passed
- [ ] Download artifacts
- [ ] Record results in VALIDATION_TRACKER.md
- [ ] Update GitHub issue (if created)
- [ ] Wait 3-7 days before next run

**After 3 runs**:
- [ ] Calculate aggregate results
- [ ] Make approval decision
- [ ] Update MIGRATION_GRAPH_DEFAULT.md
- [ ] Create deployment PR
- [ ] Get PR reviewed and approved
- [ ] Merge to production
- [ ] Monitor for first week

---

## Quick Commands Reference

```bash
# GitHub CLI - Trigger workflow
gh workflow run weekly-full-rebuild.yml

# GitHub CLI - List recent runs
gh run list --workflow=weekly-full-rebuild.yml --limit 5

# GitHub CLI - Watch a run
gh run watch <run-id>

# GitHub CLI - Download artifacts
gh run download <run-id>

# Local validation (optional)
./scripts/run_local_validation.sh

# Update docs
vim docs/VALIDATION_TRACKER.md

# Commit docs
git add docs/
git commit -m "docs: record validation run X results"
git push origin main
```

---

## Need Help?

**If you get stuck**:

1. **Check this guide's troubleshooting section** (above)
2. **Review workflow logs** on GitHub for specific errors
3. **Try local validation first** to debug issues faster
4. **Create a GitHub issue** with the `help-wanted` label
5. **Check GitHub Actions documentation**: https://docs.github.com/en/actions

**Common resources**:
- GitHub Actions docs: https://docs.github.com/en/actions
- GitHub CLI docs: https://cli.github.com/manual/
- Repository docs: `docs/` folder

---

**Last Updated**: 2025-12-02
