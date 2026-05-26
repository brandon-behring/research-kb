# Code Review — 2026-05-25 metadata.roles + matcher hardening commits

Three commits landed on `main` this session:

- `3ef8ec8` — `feat(storage): metadata.roles helpers + role-aware graph export`
- `cdad60f` — `feat(cli): sources add-manual + set-role + graph export --role`
- `6fcf97f` — `fix(citation_graph): alias support + year-filtered partial-LIKE + rebuild flags` (closes #14)

Reviewed by three specialized agents in parallel: `feature-dev:code-reviewer` (general correctness), `pr-review-toolkit:silent-failure-hunter` (error handling / silent partial state), `pr-review-toolkit:pr-test-analyzer` (test coverage).

## Findings (confidence ≥ 80)

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | Partial-LIKE branch treats `%` and `_` in citation titles as wildcards — academic title `"100% NLP"` matches arbitrary unrelated sources | CRITICAL | **FIXED** (this commit) — see #15 |
| 2 | `--full-rebuild` / `--rebuild-unmatched` are non-transactional: DELETE commits before rebuild; crash mid-loop = graph zeroed | CRITICAL | Open — see #16 |
| 3 | Malformed `metadata.aliases` (non-array) crashes the EXISTS subquery globally — one bad row poisons every match | CRITICAL | **FIXED** (this commit) — see #17 |
| 4 | Alias branch year asymmetry: alias matches accept `year IS NULL` source side; partial-LIKE doesn't. Docstring asymmetry vs implementation | HIGH | Open — not filed; smaller follow-up |
| 5 | `sources add-manual` idempotency path is non-atomic: `update_metadata` + `add_roles` not in a transaction | HIGH | Open — not filed; low real-world probability for CLI use |
| 6 | `graph export --role <nonexistent>` silently emits empty graph (exit 0, file written, 0 nodes) | MEDIUM | Open — not filed; UX rather than correctness |
| 7 | `format_citation_graph_export` union/dedup logic is untested at integration level (CLI tests mock the function) | TEST GAP | Open — not filed |
| 8 | `--full-rebuild` / `--rebuild-unmatched` flags have no smoke test | TEST GAP | Folded into #16 |

Open findings (4-8) are documented here but deliberately not filed as separate issues this session — either they're low-probability for current usage (#4, #5) or they're UX/test gaps where the right scope-of-fix is still ambiguous (#6, #7).

## Fixes shipped in this commit

### Fix #1 — Partial-substring search switched from LIKE to position()

`packages/storage/src/research_kb_storage/citation_graph.py` priority-4 branch was using:

```sql
WHERE (LOWER(title) LIKE '%' || $1 || '%' OR $1 LIKE '%' || LOWER(title) || '%')
  AND ($2::int IS NULL OR year = $2)
```

`$1` is parameterized but Postgres still interpreted `%` and `_` within the bound value as wildcards. Replaced with metacharacter-safe `position()`:

```sql
WHERE (position($1 IN LOWER(title)) > 0
       OR (LENGTH(LOWER(title)) > 0 AND position(LOWER(title) IN $1) > 0))
  AND ($2::int IS NULL OR year = $2)
```

`position(a IN b) > 0` is the SQL-standard equivalent of `b ILIKE '%a%'` without the wildcard hazard. The `LENGTH > 0` guard prevents the spurious "empty source title is a substring of any citation" match.

Regression tests in `TestLikeMetacharacterSafety`:
- Citation `"100% NLP"` (yr 2020) vs source `"100 papers on building scalable nlp systems"` (yr 2020) → no match (was a false positive under LIKE)
- Citation `"abc_xyz_def"` vs source `"abc xyz def"` → no match (was a false positive under LIKE; `_` matches any single char)

### Fix #3 — JSONB type guard on alias EXISTS subquery

`packages/storage/src/research_kb_storage/citation_graph.py` priority-3 alias EXISTS now guards on `jsonb_typeof(metadata->'aliases') = 'array'` before calling `jsonb_array_elements_text`. A single source with `metadata.aliases = "foo"` (string-not-array) no longer poisons the entire matcher.

Regression tests in `TestMalformedAliasesGuard`:
- Source A with `metadata.aliases = "single-string"` + Source B with normal title → citation matching B succeeds (was raising under previous code)
- Same with `metadata.aliases = {"some_key": "..."}` (object shape)

## Ship verdict

**Ship-with-follow-ups.** Critical fixes #1 and #3 are landed in this commit. Critical #2 (transaction redesign) is filed as #16 — needs a focused follow-up before the next production `--full-rebuild`. Findings 4-8 are documented but not blocking.

## Reference

- Agents reports: not committed; see session conversation for verbatim output.
- Existing 26 matcher tests pass before and after the fixes; 4 new regression tests added (9 total in `test_citation_matcher_aliases.py`).
- Related: this review followed the 13-commit cross-repo work on the brandon-behring.dev portfolio site (A4 RL citation graph enrichment + cleanup). Today's focus shifted to research-kb only because that's where the highest-risk code surface lives.
