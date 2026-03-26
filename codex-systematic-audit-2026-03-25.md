# Codex Systematic Audit Report

**Date:** March 25, 2026
**Repo:** `/home/brandon_behring/Claude/research-kb`
**Scope:** Independent audit of the entire repository: contracts, documentation, tests, package boundaries, and operational scripts.

## Executive Verdict

The repository is technically stronger than its self-description, but its trust surface is weaker than it should be.

- **What is strong:** the monorepo is real, not cosmetic. Package boundaries exist, the type baseline is clean, the test surface is broad, and representative suites still pass.
- **What is weak:** interface defaults disagree across surfaces, documentation marked as authoritative is stale, the CI-style integration marker suite is not green, and the repo still claims cleanliness that current lint/format checks do not support.
- **Overall grade today:** **Conditional pass for internal engineering use.** I would not treat the repo as self-describing or “trust the docs and interfaces” ready until the issues below are corrected.

## Methodology

These are the checks I actually ran on March 25, 2026:

| Check | Command | Result |
|---|---|---|
| Lint | `./.venv/bin/ruff check packages scripts tests` | **Fail**: 15 errors |
| Formatting | `./.venv/bin/black --check packages scripts tests` | **Fail**: 28 files would reformat |
| Typing | `./.venv/bin/python scripts/mypy_baseline_check.py` | **Pass**: `0 known`, `0 new` |
| Docs audit | `./.venv/bin/python scripts/audit_docs.py --ci` | **Warn**: README says `1,198` sources, live DB has `1,247` |
| Live DB stats | `./.venv/bin/research-kb sources stats` | `1,247` sources, `1,018,196` chunks |
| Status freshness | `./.venv/bin/python scripts/generate_status.py --check` | **Fail**: `docs/status/CURRENT_STATUS.md` is out of date |
| Search pipeline file | `./.venv/bin/pytest tests/integration/test_search_pipeline.py -q` | **Pass**: `18 passed` |
| Seed validation file | `./.venv/bin/pytest tests/integration/test_seed_concept_validation.py -q` | **Pass**: `11 passed` |
| API app setup | `./.venv/bin/pytest packages/api/tests/test_main.py -q` | **Pass**: `30 passed` |
| MCP tool registration | `./.venv/bin/pytest packages/mcp-server/tests/test_tools.py -q` | **Pass**: `10 passed` |
| Broad integration marker set | `./.venv/bin/pytest packages/ tests/ -m "integration and not requires_embedding and not requires_ollama and not requires_reranker" -q` | **Fail**: `7 failed, 318 passed, 2452 deselected` |

## Findings

## 1. Critical: Graph defaults are not one contract

**What’s working**

- The CLI and MCP tool surface already converge on the intended policy: graph is off by default while KG re-extraction is pending.
- The service layer has at least some fallback logic when graph search is requested.

**What’s weak**

- The API request schema defaults `use_graph=True` in [schemas.py](/home/brandon_behring/Claude/research-kb/packages/api/src/research_kb_api/schemas.py#L53).
- The dashboard API client defaults `use_graph=True` in [api_client.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/src/research_kb_dashboard/api_client.py#L96).
- The dashboard search page hard-codes `use_graph=True` in [search.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/src/research_kb_dashboard/pages/search.py#L50).
- The client SDK defaults `use_graph=True` in [socket_client.py](/home/brandon_behring/Claude/research-kb/packages/client/src/research_kb_client/socket_client.py#L259).
- The CLI defaults graph off in [search.py](/home/brandon_behring/Claude/research-kb/packages/cli/src/research_kb_cli/commands/search.py#L55) and MCP does the same in [search.py](/home/brandon_behring/Claude/research-kb/packages/mcp-server/src/research_kb_mcp/tools/search.py#L21).
- The API service fallback only checks whether any concepts exist, not whether `chunk_concepts` is stale, in [service.py](/home/brandon_behring/Claude/research-kb/packages/api/src/research_kb_api/service.py#L269).
- Tests reinforce the wrong defaults: [test_schemas.py](/home/brandon_behring/Claude/research-kb/packages/api/tests/test_schemas.py#L131) and [test_api_client.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/tests/test_api_client.py#L156) both explicitly assert `use_graph=True`.
- The top-level README still says graph and citation are enabled by default in CLI and MCP in [README.md](/home/brandon_behring/Claude/research-kb/README.md#L108).

**Why it matters**

- In FastAPI, the request model is not just an internal detail; it becomes part of the generated OpenAPI surface. If the Pydantic defaults are wrong, the API contract is wrong.
- Right now the repo tells different stories depending on how a caller enters the system. That is worse than a simple bug because it makes behavior hard to reason about and hard to trust.
- The fallback is also checking the wrong health signal for the documented problem. The live docs say graph is degraded because `chunk_concepts` is stale, not because concepts are missing.

**How to improve**

- Define one shared search-capability policy and one shared set of search defaults.
- Drive API schema defaults, dashboard defaults, SDK defaults, CLI docs, MCP docs, and tests from that same policy.
- Expose live graph availability from the service layer instead of letting every client guess.
- Change tests so they assert cross-surface consistency, not per-surface local defaults.

**Evidence**

- [schemas.py](/home/brandon_behring/Claude/research-kb/packages/api/src/research_kb_api/schemas.py#L53)
- [api_client.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/src/research_kb_dashboard/api_client.py#L96)
- [search.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/src/research_kb_dashboard/pages/search.py#L50)
- [socket_client.py](/home/brandon_behring/Claude/research-kb/packages/client/src/research_kb_client/socket_client.py#L259)
- [search.py](/home/brandon_behring/Claude/research-kb/packages/cli/src/research_kb_cli/commands/search.py#L55)
- [search.py](/home/brandon_behring/Claude/research-kb/packages/mcp-server/src/research_kb_mcp/tools/search.py#L21)
- [service.py](/home/brandon_behring/Claude/research-kb/packages/api/src/research_kb_api/service.py#L269)
- [README.md](/home/brandon_behring/Claude/research-kb/README.md#L108)
- [test_schemas.py](/home/brandon_behring/Claude/research-kb/packages/api/tests/test_schemas.py#L131)
- [test_api_client.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/tests/test_api_client.py#L156)

**External research**

- FastAPI request bodies are defined from Pydantic models and surfaced in generated API docs: https://fastapi.tiangolo.com/tutorial/body/
- Pydantic field defaults are part of the model definition and downstream schema behavior: https://docs.pydantic.dev/latest/concepts/fields/

## 2. Critical: The repo’s “source of truth” documentation is not trustworthy

**What’s working**

- The repo already has the right instincts: an auto-generated status file, a documentation audit script, and auto-generated README sections.
- The docs audit logic in [audit_docs.py](/home/brandon_behring/Claude/research-kb/scripts/audit_docs.py#L602) is useful and specific.

**What’s weak**

- [CURRENT_STATUS.md](/home/brandon_behring/Claude/research-kb/docs/status/CURRENT_STATUS.md#L1) claims to be auto-generated on `2026-03-25 10:26:34`, but `scripts/generate_status.py --check` reported it was already out of date later the same day.
- The live DB reported `1,247` sources and `1,018,196` chunks, while [CURRENT_STATUS.md](/home/brandon_behring/Claude/research-kb/docs/status/CURRENT_STATUS.md#L12) still says `1,198` sources and `1,012,187` chunks.
- [INDEX.md](/home/brandon_behring/Claude/research-kb/docs/INDEX.md#L5) still advertises `997 sources`, `857K chunks`, and `100% embedded`.
- [STRATEGIC_ASSESSMENT.md](/home/brandon_behring/Claude/research-kb/docs/STRATEGIC_ASSESSMENT.md#L81) still says “No more ingestion until KG is restored” and claims all black/ruff work is done, while its current-state metrics also remain stale.
- [README.md](/home/brandon_behring/Claude/research-kb/README.md#L200) still shows `41,852` citations, while [CURRENT_STATUS.md](/home/brandon_behring/Claude/research-kb/docs/status/CURRENT_STATUS.md#L17) shows `40,933`.

**Why it matters**

- The repo is trying to present itself as audit-friendly and self-checking. Stale “authoritative” docs are more damaging in that environment than in a casual side project, because readers are encouraged to trust them.
- This is also a governance failure, not just a copy-editing failure. The auto-generated parts are not broad enough, and the freshness gate is not strong enough.

**How to improve**

- Stop duplicating live metrics in hand-maintained docs wherever possible.
- Expand generation/checking beyond marker sections so the landing pages cannot silently drift.
- Fail CI when the status file is stale, not just warn.
- Timestamp all live-metric claims and distinguish “historical snapshot” from “current state.”

**Evidence**

- [CURRENT_STATUS.md](/home/brandon_behring/Claude/research-kb/docs/status/CURRENT_STATUS.md#L1)
- [INDEX.md](/home/brandon_behring/Claude/research-kb/docs/INDEX.md#L5)
- [STRATEGIC_ASSESSMENT.md](/home/brandon_behring/Claude/research-kb/docs/STRATEGIC_ASSESSMENT.md#L81)
- [README.md](/home/brandon_behring/Claude/research-kb/README.md#L200)
- [audit_docs.py](/home/brandon_behring/Claude/research-kb/scripts/audit_docs.py#L602)
- Command results:
  - `./.venv/bin/python scripts/audit_docs.py --ci` -> README says `1,198` sources, live DB has `1,247`
  - `./.venv/bin/research-kb sources stats` -> `1,247` sources, `1,018,196` chunks
  - `./.venv/bin/python scripts/generate_status.py --check` -> status file out of date

**External research**

- Tan, Wagner, and Treude studied outdated code-element references in repository documentation and showed that repo documentation drift is a real, recurring engineering problem, not a cosmetic one: https://link.springer.com/article/10.1007/s10664-023-10397-6

## 3. High: The broad integration suite is not green, and the failure mode appears to be state leakage

**What’s working**

- Several representative suites passed in isolation on March 25, 2026:
  - `tests/integration/test_search_pipeline.py -q` -> `18 passed`
  - `tests/integration/test_seed_concept_validation.py -q` -> `11 passed`
  - `packages/api/tests/test_main.py -q` -> `30 passed`
  - `packages/mcp-server/tests/test_tools.py -q` -> `10 passed`
- The failures in the broad integration marker run are localized, not repo-wide chaos.

**What’s weak**

- The CI-style integration marker run failed with `7 failed, 318 passed, 2452 deselected`.
- The failure cluster is centered in assumption-audit behavior:
  - `packages/storage/tests/test_assumption_audit.py`
  - `tests/integration/test_search_pipeline.py`
- The implementation caches KG staleness per process in [assumption_audit.py](/home/brandon_behring/Claude/research-kb/packages/storage/src/research_kb_storage/assumption_audit.py#L191).
- The integration fixtures reset the DB and the pool, but not that process-global cache in [packages/storage/conftest.py](/home/brandon_behring/Claude/research-kb/packages/storage/conftest.py#L39).
- The unit test suite already works around this exact issue by forcing `_kg_stale_cache = False` and then clearing it in [test_assumption_audit_unit.py](/home/brandon_behring/Claude/research-kb/packages/storage/tests/test_assumption_audit_unit.py#L33).

**Why it matters**

- Order-dependent integration failures are worse than ordinary failures because they hide behind partial green runs.
- The repo currently gives different answers depending on whether assumption-audit tests ran earlier in the same process. That is a real isolation bug, not just a stale assertion.
- The staleness guard is reasonable for production, but it is currently contaminating isolated integration fixtures.

**How to improve**

- Reset `_kg_stale_cache` in integration fixtures, not just unit tests.
- Scope the cache to the database identity or transaction snapshot instead of the Python process.
- Add a regression test that runs an assumption-audit test and a search-pipeline assumption test in the same process.
- Decide whether the staleness guard is a runtime concern only. If so, make it injectable or overridable for tests that seed their own valid `chunk_concepts`.

**Evidence**

- [assumption_audit.py](/home/brandon_behring/Claude/research-kb/packages/storage/src/research_kb_storage/assumption_audit.py#L191)
- [assumption_audit.py](/home/brandon_behring/Claude/research-kb/packages/storage/src/research_kb_storage/assumption_audit.py#L962)
- [assumption_audit.py](/home/brandon_behring/Claude/research-kb/packages/storage/src/research_kb_storage/assumption_audit.py#L1019)
- [packages/storage/conftest.py](/home/brandon_behring/Claude/research-kb/packages/storage/conftest.py#L39)
- [test_assumption_audit_unit.py](/home/brandon_behring/Claude/research-kb/packages/storage/tests/test_assumption_audit_unit.py#L33)
- [test_assumption_audit.py](/home/brandon_behring/Claude/research-kb/packages/storage/tests/test_assumption_audit.py#L831)
- [test_search_pipeline.py](/home/brandon_behring/Claude/research-kb/tests/integration/test_search_pipeline.py#L214)
- Broad integration result: `7 failed, 318 passed, 2452 deselected`
- Targeted reproduction result:
  - `./.venv/bin/pytest packages/storage/tests/test_assumption_audit.py::TestAuditAssumptions::test_method_with_sufficient_graph_assumptions tests/integration/test_search_pipeline.py::TestAssumptionAudit::test_audit_finds_iv_assumptions -q`
  - Both tests failed in the same process, even though `tests/integration/test_search_pipeline.py -q` passed on its own earlier.

**External research**

- No external citation is needed to justify this finding. The local evidence is specific and sufficient.

## 4. High: The repo’s quality posture is overstated

**What’s working**

- Type checking is genuinely clean.
- The repo has a serious quality culture: many tests, explicit markers, docs audit, and package-level READMEs.

**What’s weak**

- `ruff` is not green.
- `black --check` is not green.
- [STRATEGIC_ASSESSMENT.md](/home/brandon_behring/Claude/research-kb/docs/STRATEGIC_ASSESSMENT.md#L81) still says “No more mypy/black/ruff phases (all at zero baseline).”
- This is a claims problem as much as a tooling problem. The repo is currently saying “done” while the actual gates disagree.

**Why it matters**

- Reviewers notice inconsistency faster than they notice technical merit. A repo that claims clean hygiene but fails basic style gates looks less disciplined than a repo that openly says “these are pending cleanup items.”
- This is especially important here because the project explicitly positions itself as auditable and professional.

**How to improve**

- Either restore strict enforcement and keep the claim, or remove the claim until the repo is actually clean again.
- Prefer small, continuous black/ruff cleanup over periodic “hygiene phases.”
- Add a lightweight status badge or machine-generated summary so cleanliness claims come from checks, not prose.

**Evidence**

- [STRATEGIC_ASSESSMENT.md](/home/brandon_behring/Claude/research-kb/docs/STRATEGIC_ASSESSMENT.md#L81)
- Command results:
  - `./.venv/bin/ruff check packages scripts tests` -> `15` errors
  - `./.venv/bin/black --check packages scripts tests` -> `28` files would reformat
  - `./.venv/bin/python scripts/mypy_baseline_check.py` -> `0 known`, `0 new`

**External research**

- No external citation is needed here. The claim is contradicted by the repo’s own current checks.

## 5. Medium: The dashboard and client surfaces turn missing API contracts into guessed numbers

**What’s working**

- The dashboard mostly goes through an API client instead of reaching into the database directly.
- The UI exposes real operational needs, which is a good sign: the project is not pretending to be purely academic.

**What’s weak**

- The queue page estimates processed chunks with `chunk_concept_links // 3` in [queue.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/src/research_kb_dashboard/pages/queue.py#L42).
- It then derives ETA from a hardcoded `1.5 chunks/min` assumption in [queue.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/src/research_kb_dashboard/pages/queue.py#L136).
- The page explicitly admits the API lacks the needed endpoint, but still renders approximate progress as if it were operationally meaningful.
- The operator guidance is wrong: the page tells users to run `research-kb extraction-status` in [queue.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/src/research_kb_dashboard/pages/queue.py#L146), while the actual CLI command is `research-kb sources extraction-status` in [sources.py](/home/brandon_behring/Claude/research-kb/packages/cli/src/research_kb_cli/commands/sources.py#L98).

**Why it matters**

- Dashboards should make missing contracts visible, not hide them behind invented math.
- Approximate operational metrics are acceptable when clearly labeled as exploratory. They are not acceptable when they can be mistaken for real queue state.

**How to improve**

- Add a real extraction-status endpoint if the UI is meant to support operations.
- If that endpoint is not worth building, remove the invented progress and ETA numbers and present the limitation directly.
- Fix the CLI guidance immediately; it is simply wrong.

**Evidence**

- [queue.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/src/research_kb_dashboard/pages/queue.py#L42)
- [queue.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/src/research_kb_dashboard/pages/queue.py#L136)
- [queue.py](/home/brandon_behring/Claude/research-kb/packages/dashboard/src/research_kb_dashboard/pages/queue.py#L146)
- [sources.py](/home/brandon_behring/Claude/research-kb/packages/cli/src/research_kb_cli/commands/sources.py#L98)

**External research**

- No external citation is needed. This is a direct contract/design gap visible in the repo.

## 6. Medium: Operational scripts are large enough to become a separate maintenance problem

**What’s working**

- The repo has deep operational capability: ingestion, enrichment, validation, catalog work, benchmarking, and auditing scripts are all present.
- That is real leverage for a single maintainer or a small team.

**What’s weak**

- Several operational entrypoints are very large:
  - `scripts/ingest_corpus.py` -> `1888` lines
  - `scripts/catalog_library/enrich_catalog.py` -> `1456` lines
  - `scripts/catalog_library/catalog_books.py` -> `1191` lines
  - `scripts/audit_docs.py` -> `869` lines
- Broad `except Exception` handling appears in the most important paths:
  - [ingest_corpus.py](/home/brandon_behring/Claude/research-kb/scripts/ingest_corpus.py#L1809)
  - [enrich_catalog.py](/home/brandon_behring/Claude/research-kb/scripts/catalog_library/enrich_catalog.py#L508)
  - [catalog_books.py](/home/brandon_behring/Claude/research-kb/scripts/catalog_library/catalog_books.py#L489)

**Why it matters**

- Big, exception-heavy scripts accumulate invisible state and implicit branching. They become the part of the repo most likely to resist testing, refactoring, and debugging.
- In this repo, the scripts are not peripheral. They are part of the product’s operational backbone, so their maintainability matters.

**How to improve**

- Split the large scripts into importable library modules with thin CLI wrappers.
- Replace catch-all exceptions with typed errors or explicit result objects where possible.
- Add focused unit tests for the extracted pure functions instead of relying on long end-to-end script tests for everything.

**Evidence**

- Command result:
  - `find packages scripts -name '*.py' -print0 | xargs -0 wc -l | sort -nr | sed -n '1,15p'`
- [ingest_corpus.py](/home/brandon_behring/Claude/research-kb/scripts/ingest_corpus.py#L1809)
- [enrich_catalog.py](/home/brandon_behring/Claude/research-kb/scripts/catalog_library/enrich_catalog.py#L508)
- [catalog_books.py](/home/brandon_behring/Claude/research-kb/scripts/catalog_library/catalog_books.py#L489)

**External research**

- McCabe’s classic complexity work remains the canonical argument that growing control-flow complexity becomes a maintainability and defect-risk problem if left unchecked: https://ics.uci.edu/~jajones/INF102-S18/readings/03_mccabe.pdf

## 7. Medium: The repo still tells at least two different product stories

**What’s working**

- The top-level README clearly positions the system as multi-surface and multi-domain in [README.md](/home/brandon_behring/Claude/research-kb/README.md#L7).
- That broader positioning matches the current package layout much better than the older causal-inference-only framing.

**What’s weak**

- The API app description still says “causal inference literature” in [main.py](/home/brandon_behring/Claude/research-kb/packages/api/src/research_kb_api/main.py#L44).
- The system design doc still opens with “causal inference literature” in [SYSTEM_DESIGN.md](/home/brandon_behring/Claude/research-kb/docs/SYSTEM_DESIGN.md#L7).
- The MCP package README still says it exposes a “causal inference knowledge base” in [README.md](/home/brandon_behring/Claude/research-kb/packages/mcp-server/README.md#L1).

**Why it matters**

- Narrative drift makes package docs feel older than the code.
- It also makes it harder for collaborators to understand whether “causal inference” is the domain, the historical origin, or just one of many supported slices.

**How to improve**

- Pick one canonical description of the product and propagate it to package entrypoints, app metadata, and doc landing pages.
- If causal inference remains the flagship domain, say that explicitly instead of letting old phrasing imply single-domain scope.

**Evidence**

- [README.md](/home/brandon_behring/Claude/research-kb/README.md#L7)
- [main.py](/home/brandon_behring/Claude/research-kb/packages/api/src/research_kb_api/main.py#L44)
- [SYSTEM_DESIGN.md](/home/brandon_behring/Claude/research-kb/docs/SYSTEM_DESIGN.md#L7)
- [README.md](/home/brandon_behring/Claude/research-kb/packages/mcp-server/README.md#L1)

**External research**

- No external citation is needed. This is a repo-local product-positioning inconsistency.

## What Should Be Preserved

- The package decomposition is useful and broadly coherent.
- The type baseline is strong enough to be worth protecting.
- The repo already contains the beginnings of good governance: doc audit checks, generated sections, explicit test markers, and a serious integration surface.
- The assumption-audit staleness guard is directionally sensible. The problem is how it is cached and surfaced, not the idea that stale KG data should be treated carefully.

## Priority Fixes

1. Unify graph-default policy across API schema, dashboard, SDK, CLI, MCP, tests, and docs.
2. Make status freshness a hard guarantee, not a best-effort warning.
3. Fix the assumption-audit cache leakage in integration contexts.
4. Decide whether the repo really wants a zero-warning quality posture. If yes, enforce it again. If not, stop claiming it.
5. Replace the dashboard’s invented extraction metrics with either a real endpoint or a simpler honest UI.
6. Start breaking the largest operational scripts into smaller library modules.

## Final Judgment

This is a good engineering repo that is currently undermining itself. The strongest parts are the architecture, the breadth of the interfaces, and the seriousness of the test and tooling story. The weakest parts are not the fundamentals of the system; they are the inconsistencies that make the repo harder to trust than it should be. That is fixable, but it requires treating defaults, freshness, and test isolation as first-class engineering work rather than cleanup.
