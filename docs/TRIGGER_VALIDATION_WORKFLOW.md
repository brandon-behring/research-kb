# How to Trigger CI Workflows

Instructions for manually running the two scheduled CI workflows.

---

## Prerequisites

- Write access to `github.com/brandonmbehring-dev/research-kb`
- GitHub CLI installed (optional, for command-line triggering)

---

## Method 1: GitHub Web UI

1. Navigate to https://github.com/brandonmbehring-dev/research-kb
2. Click the **Actions** tab
3. In the left sidebar, select the workflow:
   - **"Weekly Full Rebuild & Validation"** -- full data pipeline
   - **"Weekly Integration Tests"** -- DB-only tests
4. Click **"Run workflow"** (top right)
5. Select branch: `main`
6. Click the green **"Run workflow"** button

Monitor progress by clicking on the new run. Green checkmarks indicate success; red X indicates failure.

---

## Method 2: GitHub CLI

```bash
# Trigger full rebuild (~45 min)
gh workflow run weekly-full-rebuild.yml

# Trigger integration tests (~15 min)
gh workflow run integration-test.yml

# List recent runs
gh run list --workflow=weekly-full-rebuild.yml

# Watch a run in real time
gh run watch <run-id>

# View logs
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>
```

---

## Method 3: REST API

```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/repos/brandonmbehring-dev/research-kb/actions/workflows/weekly-full-rebuild.yml/dispatches \
  -d '{"ref":"main"}'
```

---

## What Each Workflow Does

### Integration Test (`integration-test.yml`)

**Duration**: ~15 min | **Schedule**: Manual trigger only

Runs the full test suite against an empty PostgreSQL + pgvector database. No demo data, no embeddings. Tests database operations, schema integrity, and mocked search paths.

### Full Rebuild (`weekly-full-rebuild.yml`)

**Duration**: ~45 min | **Schedule**: Manual trigger only

End-to-end data pipeline validation:

| Step | What Happens | Duration |
|------|--------------|----------|
| Setup | Checkout, Python 3.11, install packages | ~2 min |
| Schema | Apply `schema.sql` + migrations to pgvector | ~30s |
| Load demo corpus | Insert 9 papers, ~1300 chunks, concepts, citations | ~1 min |
| Start embed server | BGE-large-en-v1.5 (cached ~400MB model) | ~1 min |
| Generate embeddings | Embed all chunks (`--batch 100`) | ~10-20 min |
| Validate retrieval | Golden dataset eval, MRR >= 0.5 gate | ~2 min |
| Unit tests | Full unit suite (excluding service-dependent) | ~5 min |

---

## After Completion

1. Check the workflow run status in the Actions tab
2. Download the `retrieval-metrics` artifact for detailed results
3. If the run failed, click on the failed step to see logs

---

**Last Updated**: 2026-02-20
