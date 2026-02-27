# CI Validation Status

**Status**: Complete

Both CI workflows are implemented and operational.

---

## Workflows

| Workflow | File | Schedule | Status |
|----------|------|----------|--------|
| PR Checks | `pr-checks.yml` | Every PR | Active |
| Integration Test | `integration-test.yml` | Manual (workflow_dispatch) | Active |
| Full Rebuild | `weekly-full-rebuild.yml` | Manual (workflow_dispatch) | Active |

---

## Full Rebuild Pipeline

The `weekly-full-rebuild.yml` workflow validates the complete data path:

1. Load demo corpus (9 open-access papers, ~1300 chunks)
2. Generate embeddings (BGE-large-en-v1.5)
3. Evaluate retrieval quality against 98-case YAML benchmark (`retrieval_test_cases.yaml`)
4. Run unit test suite

**Quality gate**: MRR >= 0.85 on core domains (`--gate-domains` flag). Pipeline fails if threshold is not met.

**Artifacts**: `retrieval-metrics` JSON with hit_rate, MRR, NDCG, per-domain breakdown.

---

## How to Trigger

- **Manual trigger**: `gh workflow run weekly-full-rebuild.yml`
- **Manual**: Actions tab > "Weekly Full Rebuild & Validation" > Run workflow
- **CLI**: `gh workflow run weekly-full-rebuild.yml`

---

## References

- [CI Quick Reference](../VALIDATION_QUICK_REFERENCE.md)
- [Trigger Instructions](../TRIGGER_VALIDATION_WORKFLOW.md)
- [CI Quick Start](../VALIDATION_QUICK_START.md)

---

**Last Updated**: 2026-02-27
