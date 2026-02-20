# CI Pipeline Quick Reference

Overview of the two CI workflows and how to use them.

---

## Workflows

| Workflow | File | Schedule | Duration | Purpose |
|----------|------|----------|----------|---------|
| **PR Checks** | `pr-checks.yml` | Every PR | ~8 min | Unit + integration tests, pytest-cov |
| **Integration Test** | `integration-test.yml` | Sunday 2 AM UTC | ~15 min | DB-only tests against empty database |
| **Full Rebuild** | `weekly-full-rebuild.yml` | Sunday 3 AM UTC | ~45 min | Load demo data, embed, validate retrieval quality |

---

## Full Rebuild Pipeline Steps

The `weekly-full-rebuild.yml` workflow validates the entire data path:

1. **Checkout + setup** -- Python 3.11, pip install packages
2. **Apply schema** -- `schema.sql` + all migrations against pgvector
3. **Load demo corpus** -- `load_demo_data.py` (9 papers, ~1300 chunks, concepts, citations)
4. **Start embedding server** -- BGE-large-en-v1.5 (cached in GitHub Actions)
5. **Generate embeddings** -- `embed_missing.py --batch 100`
6. **Validate retrieval** -- `eval_retrieval.py --dataset golden_dataset.json --fail-below 0.5`
7. **Run unit tests** -- Full unit suite (excluding service-dependent tests)

---

## Triggering Manually

### GitHub UI

1. Go to https://github.com/brandonmbehring-dev/research-kb
2. Click **Actions** tab
3. Select the workflow in the left sidebar
4. Click **Run workflow** > select `main` > click **Run workflow**

### GitHub CLI

```bash
# Full rebuild
gh workflow run weekly-full-rebuild.yml

# Integration test
gh workflow run integration-test.yml

# Monitor
gh run watch $(gh run list --workflow=weekly-full-rebuild.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```

---

## Quality Gates

**Full Rebuild must pass:**
- MRR >= 0.5 on golden dataset (92 queries across 8+ domains)
- All unit tests pass
- Embedding generation completes without error

**PR Checks must pass:**
- Unit + integration tests
- Coverage report generated

---

## Reading Results

**Retrieval metrics** are uploaded as a `retrieval-metrics` artifact on every full rebuild run. Download from the workflow run page or via:

```bash
gh run download <run-id> -n retrieval-metrics
```

The JSON file contains hit_rate, MRR, NDCG, and per-domain breakdowns.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Workflow not in Actions tab | Ensure file is on `main` branch and pushed |
| "Run workflow" button disabled | Need write access to repository |
| Schema apply fails | Check `schema.sql` syntax, verify pgvector image |
| Embedding server timeout | Model cache may be cold; re-run workflow |
| MRR below threshold | Check golden_dataset.json for entries without matching chunks |

---

## Key Files

- `.github/workflows/weekly-full-rebuild.yml` -- Full pipeline workflow
- `.github/workflows/integration-test.yml` -- DB-only integration tests
- `.github/workflows/pr-checks.yml` -- PR gate checks
- `fixtures/eval/golden_dataset.json` -- 92 retrieval test queries
- `scripts/eval_retrieval.py` -- Retrieval evaluation script

---

**Last Updated**: 2026-02-20
