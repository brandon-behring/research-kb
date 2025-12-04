---
name: Graph Search Default Validation Tracking
about: Track the 3 required validation runs before enabling graph search by default
title: 'Validation: Graph Search as Default (Phase 3C)'
labels: validation, phase-3c, testing
assignees: ''

---

## Overview

This issue tracks the 3 required validation runs before enabling graph search as the default in production.

**Goal**: Prove that graph-boosted search maintains quality and performance standards.

**Status**: 0/3 runs complete

---

## Validation Runs

### Run 1

**Date**: TBD
**Workflow**: [Link to workflow run]
**Status**: ⏸️ Pending

**Quality Gates**:
- [ ] Corpus ingestion successful (~500 chunks)
- [ ] Retrieval validation passed (Precision@K ≥90%)
- [ ] Concept extraction completed (≥100 concepts)
- [ ] Seed concept validation passed (Recall ≥70%)
- [ ] Graph validation passed
- [ ] All CLI tests passed (47 tests)
- [ ] All script tests passed (24 tests)
- [ ] Database cached successfully

**Performance**:
- Query latency p50: ___ ms
- Query latency p95: ___ ms
- Corpus ingestion time: ___ min

**Notes**:


---

### Run 2

**Date**: TBD
**Workflow**: [Link to workflow run]
**Status**: ⏸️ Pending

**Quality Gates**:
- [ ] Corpus ingestion successful (~500 chunks)
- [ ] Retrieval validation passed (Precision@K ≥90%)
- [ ] Concept extraction completed (≥100 concepts)
- [ ] Seed concept validation passed (Recall ≥70%)
- [ ] Graph validation passed
- [ ] All CLI tests passed (47 tests)
- [ ] All script tests passed (24 tests)
- [ ] Database cached successfully

**Performance**:
- Query latency p50: ___ ms
- Query latency p95: ___ ms
- Corpus ingestion time: ___ min

**Notes**:


---

### Run 3

**Date**: TBD
**Workflow**: [Link to workflow run]
**Status**: ⏸️ Pending

**Quality Gates**:
- [ ] Corpus ingestion successful (~500 chunks)
- [ ] Retrieval validation passed (Precision@K ≥90%)
- [ ] Concept extraction completed (≥100 concepts)
- [ ] Seed concept validation passed (Recall ≥70%)
- [ ] Graph validation passed
- [ ] All CLI tests passed (47 tests)
- [ ] All script tests passed (24 tests)
- [ ] Database cached successfully

**Performance**:
- Query latency p50: ___ ms
- Query latency p95: ___ ms
- Corpus ingestion time: ___ min

**Notes**:


---

## Aggregate Results

**Success Rate**: 0/3

**Average Metrics**:
- Query latency p50: N/A
- Query latency p95: N/A
- Corpus ingestion time: N/A

**Blockers**: None

**Warnings**: None

---

## Decision

- [ ] All 3 runs completed
- [ ] No blockers in any run
- [ ] ≤1 warning per run
- [ ] Performance stable across runs (±20%)

**Decision**: ⏸️ Pending

**Next Steps**: TBD

---

## References

- **Migration Guide**: docs/MIGRATION_GRAPH_DEFAULT.md
- **Validation Tracker**: docs/VALIDATION_TRACKER.md
- **How to Trigger**: docs/TRIGGER_VALIDATION_WORKFLOW.md
- **Weekly Workflow**: .github/workflows/weekly-full-rebuild.yml

---

## Notes

<!-- Add any additional notes, observations, or concerns here -->
