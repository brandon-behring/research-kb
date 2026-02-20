# CI Quick Start

How to read CI results and run validation locally.

---

## Reading CI Results

### PR Checks (every pull request)

After opening a PR, the `pr-checks` workflow runs automatically. Check the **Checks** tab on your PR for:

- Unit test results (pass/fail count)
- Integration test results
- Coverage report (XML artifact)

### Full Rebuild (weekly or on-demand)

The `weekly-full-rebuild` workflow validates the complete pipeline. After a run:

1. Go to **Actions** > click on the run
2. Check each step for green checkmarks
3. Download the `retrieval-metrics` artifact for MRR, hit rate, and per-domain scores

**Key metrics to check:**
- **MRR** (Mean Reciprocal Rank): >= 0.5 required
- **Hit Rate@K**: percentage of queries that find expected results
- **Per-domain scores**: identify weak domains

---

## What Failures Mean

| Failure | Likely Cause | Fix |
|---------|-------------|-----|
| Schema apply | Migration syntax error | Check `packages/storage/migrations/` |
| Load demo data | Missing fixture files | Regenerate with `export_demo_data.py` |
| Embedding timeout | Model not cached yet | Re-run (cache persists across runs) |
| MRR below threshold | Golden dataset entries lack matching chunks | Update `fixtures/eval/golden_dataset.json` |
| Unit tests fail | Code regression | Run `pytest -m unit` locally |

---

## Running Locally

### Prerequisites

- PostgreSQL with pgvector running (via `docker compose up -d`)
- Python 3.11+ with packages installed

### Full pipeline validation

```bash
# 1. Apply schema (if fresh database)
psql -h localhost -U postgres -d research_kb -f packages/storage/schema.sql
for f in packages/storage/migrations/*.sql; do
  psql -h localhost -U postgres -d research_kb -f "$f"
done

# 2. Load demo data
python scripts/load_demo_data.py

# 3. Start embedding server + generate embeddings
python -m research_kb_pdf.embed_server &
python scripts/embed_missing.py --batch 100

# 4. Evaluate retrieval
python scripts/eval_retrieval.py --dataset fixtures/eval/golden_dataset.json --per-domain --verbose

# 5. Run tests
pytest -m "unit and not requires_embedding and not requires_ollama and not requires_reranker" -q
```

### Quick smoke test (no embeddings needed)

```bash
pytest -m "unit and not requires_embedding and not requires_ollama and not requires_reranker and not requires_grobid" -q
```

---

## Triggering Workflows

See [TRIGGER_VALIDATION_WORKFLOW.md](TRIGGER_VALIDATION_WORKFLOW.md) for detailed instructions on manual triggering via GitHub UI, CLI, or API.

---

**Last Updated**: 2026-02-20
