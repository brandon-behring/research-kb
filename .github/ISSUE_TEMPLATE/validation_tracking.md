---
name: CI Pipeline Issue
about: Report a CI workflow failure or request a validation run
title: 'CI: [brief description]'
labels: ci, testing
assignees: ''

---

## Workflow

Which workflow is affected?

- [ ] `pr-checks.yml` (PR gate)
- [ ] `integration-test.yml` (weekly DB tests)
- [ ] `weekly-full-rebuild.yml` (full pipeline)

## Description

<!-- Describe the issue or request -->

## Failure Details

**Run URL**: <!-- Link to the failed workflow run -->

**Failed Step**: <!-- e.g., "Generate embeddings", "Validate retrieval quality" -->

**Error Message**:
```
<!-- Paste relevant error output -->
```

## Checklist

- [ ] Checked workflow logs for the specific error
- [ ] Verified the issue is not a transient failure (re-ran once)
- [ ] Checked that `fixtures/eval/golden_dataset.json` is up to date
- [ ] Checked that `packages/storage/schema.sql` + migrations are valid

## Environment

- **Branch**: <!-- e.g., main -->
- **Trigger**: <!-- scheduled / manual / PR -->
- **Date**: <!-- When did this happen? -->
