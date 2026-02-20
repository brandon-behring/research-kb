# CI Validation Status

**Status**: Complete

Both CI workflows are implemented and operational.

---

## Workflows

| Workflow | File | Schedule | Status |
|----------|------|----------|--------|
| PR Checks | `pr-checks.yml` | Every PR | Active |
| Integration Test | `integration-test.yml` | Sunday 2 AM UTC | Active |
| Full Rebuild | `weekly-full-rebuild.yml` | Sunday 3 AM UTC | Active |

---

## Full Rebuild Pipeline

The `weekly-full-rebuild.yml` workflow validates the complete data path:

1. Load demo corpus (9 open-access papers, ~1300 chunks)
2. Generate embeddings (BGE-large-en-v1.5)
3. Evaluate retrieval quality against 92-query golden dataset
4. Run unit test suite

**Quality gate**: MRR >= 0.5 on golden dataset. Pipeline fails if threshold is not met.

**Artifacts**: `retrieval-metrics` JSON with hit_rate, MRR, NDCG, per-domain breakdown.

---

## How to Trigger

- **Automatic**: Runs every Sunday at 3 AM UTC
- **Manual**: Actions tab > "Weekly Full Rebuild & Validation" > Run workflow
- **CLI**: `gh workflow run weekly-full-rebuild.yml`

---

## References

- [CI Quick Reference](../VALIDATION_QUICK_REFERENCE.md)
- [Trigger Instructions](../TRIGGER_VALIDATION_WORKFLOW.md)
- [CI Quick Start](../VALIDATION_QUICK_START.md)

---

**Last Updated**: 2026-02-20
