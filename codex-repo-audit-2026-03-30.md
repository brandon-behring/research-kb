# Codex Repo Audit and Roadmap Refinement

**Date:** 2026-03-30  
**Repo:** `research-kb`  
**Artifact:** dual-track audit of committed repo state plus current working-tree/local-machine delta

## Executive Verdict

- **Engineering truth:** `Partial, but materially real`
  - [fact] Core platform breadth is strong: 12 packages, CLI/API/dashboard/daemon/MCP surfaces, `1,756` live sources, `1,395,596` chunks, `49,174` citations, and green DB-only integration locally.
  - [fact] The largest operational blockers are not “the system does not exist”; they are **graph disconnection** (`chunk_concepts = 0`), **domain-label noise**, and **split chunk metadata contracts**.
  - [inference] This repo is beyond prototype stage, but parts of the live corpus are less semantically trustworthy than the platform around them.

- **External trust:** `Partial`
  - [fact] Generated status surfaces and the fast docs audit are green.
  - [fact] Several narrative docs are still materially misleading or stale on retrieval-eval counts, graph defaults, CLI navigation, and chunking history.
  - [inference] The repo is more credible than some older audit files imply, but it is still not fully self-describing.

- **Bottom line**
  - [fact] The methodology broadly makes sense as a staged ingestion and retrieval system.
  - [inference] The weak point is not “no methodology”; it is **too many parallel methodologies** for cataloging and chunk metadata, with incomplete normalization between them.

## Truth Sources and Methodology

### Labels

- `[fact]`: directly observed in code, docs, DB, or command output
- `[inference]`: reasoned conclusion from multiple facts
- `[historical context]`: prior audit/docs used only as background
- `[local WIP]`: current working-tree or untracked local state, not committed baseline

### Truth Hierarchy

1. [fact] Live DB state and runnable verification commands
2. [fact] Current code, workflows, schemas, and tests
3. [fact] Generated docs (`CURRENT_STATUS.md`, README auto-generated sections)
4. [fact] Hand-maintained docs and narrative roadmaps
5. [historical context] Archived docs and prior audits

### Audit Scope

- [fact] Progress audit
- [fact] Documentation accuracy audit
- [fact] PDF organization/classification audit
- [fact] Chunking/ingestion audit
- [fact] Validation posture audit

### Sampling Performed

- [fact] Catalog/classification sample: 40 entries
  - 10 high-skew domain entries
  - 10 `skip` entries
  - 10 ambiguous/low-confidence entries
  - 10 already-ingested entries
- [fact] Chunking sample: 6 representative sources
  - 4 Docling textbook paths
  - 1 rechunked source with stale source metadata
  - 1 large draft textbook

## Current Progress Matrix

### Live Snapshot

- [fact] `docs/status/CURRENT_STATUS.md` matches the live database.
- [fact] Live DB on this machine reports:
  - `1,756` sources
  - `1,395,596` chunks
  - `310,063` concepts
  - `743,984` concept relationships
  - `49,174` citations
  - `36` populated domains
  - `100%` chunk embedding coverage
  - `chunk_concepts = 0`

| Area | Status | Evidence | Audit Read |
|---|---|---|---|
| Platform surfaces | Complete | README, packages, entrypoints | Real monorepo with working user/operator interfaces |
| Bulk corpus ingestion | Partial | Live DB + catalog checkpoint | Big corpus exists, but catalog ingestion is still incomplete |
| Retrieval stack | Complete | CLI/API/MCP code, integration tests | 3-way search is operational |
| Citation layer | Complete | Live citations + workflows + tests | Real signal, not aspirational |
| PDF extraction/chunking | Partial | Docling code + live chunk data | Extraction works, metadata normalization does not |
| Knowledge graph grounding | Blocked | `chunk_concepts = 0` | Concepts exist, grounding layer does not |
| Validation posture | Partial | current command runs | Fast/core validation is green; E2E and methodology-specific validation have gaps |
| Roadmap coherence | Partial | ROADMAP + STRATEGIC_ASSESSMENT + live state | Strategy exists, but execution priorities need tightening |

### What Has Clearly Been Done

- [fact] The repo has moved well beyond the old “495 sources / 22 domains” era.
- [fact] The broad DB-only integration marker set is green locally: `326 passed, 2490 deselected`.
- [fact] Storage assumption-audit tests are green locally: `55 passed`.
- [fact] PDF dispatcher tests are green locally: `23 passed`.
- [fact] The live corpus is fully embedded at the chunk level.

### What Is Only Partially Done

- [fact] Catalog ingestion is incomplete: `785 / 3,697` catalog books completed, `2,317` non-skip books still not completed, `33` checkpoint failures.
- [fact] Knowledge graph extraction exists, but chunk grounding is absent.
- [fact] Domain tagging exists, but live domain labels are noisy enough to distort corpus-level status summaries.
- [fact] Docling migration happened, but the corpus does not have one unified Docling-era chunk metadata contract.

### What Is Blocked

- [fact] Graph-backed retrieval/synthesis is blocked by `chunk_concepts = 0`.
- [fact] Classifier validation is currently broken in local WIP because `scripts/validate_classifier.py` imports a deleted `scripts/classify_library_books.py`.

## Documentation Accuracy Matrix

| Surface | Claim | Reality | Assessment |
|---|---|---|---|
| `README.md` core architecture | 22 tools, 36 domains, graph disabled by default | [fact] consistent with current code and DB | Accurate enough |
| `README.md` retrieval eval counts | “108 YAML test cases across 36 domains” | [fact] `fixtures/eval/retrieval_test_cases.yaml` currently has `107` cases across `31` domains | Stale |
| `docs/CLI.md` search defaults | “Graph-boosted search (Default)” | [fact] CLI code defaults `use_graph=False` | Stale |
| `docs/INDEX.md` CLI navigation | links `CLAUDE.md#cli-usage` | [fact] that anchor does not exist | Broken link / stale |
| `ROADMAP.md` future work | lists “Adaptive chunking” as future work | [fact] Docling/HybridChunker structure-aware chunking is already live | Stale |
| `docs/phases/phase1.5/PDF_INGESTION.md` | documents PyMuPDF heading pipeline as current Phase 1.5 story | [fact] live code is Docling-first; GROBID is metadata-only | Historical snapshot, not current methodology |
| `docs/STRATEGIC_ASSESSMENT.md` | generally current strategic picture | [fact] more current than ROADMAP, but too optimistic about tagging/test cleanup being “done enough” | Useful but incomplete |
| `scripts/audit_docs.py` status | passes cleanly | [fact] its coverage is mostly README- and generated-doc-centered; it does not police `docs/CLI.md`, `ROADMAP.md`, or broken `docs/INDEX.md` anchors | Accurate but narrow |

### Documentation Assessment

- [fact] Generated truth surfaces are reliable.
- [fact] Narrative docs are mixed: some are current, some are stale, and the current fast audit does not cover all of them.
- [inference] Documentation trust is no longer in crisis, but it is **still uneven**. The repo now has a stronger automated truth layer than narrative governance layer.

## PDF Organization and Classification Assessment

### What Makes Sense

- [fact] There is a real staged catalog pipeline:
  - base cataloging in `scripts/catalog_library/catalog_books.py`
  - enrichment in `scripts/catalog_library/enrich_catalog.py`
  - later/manual refinement via `classify_phase_b.py`, review CSVs, and import scripts
  - ingestion prioritization via `scripts/mass_ingest_catalog.py`
- [fact] The pipeline includes:
  - duplicate detection
  - book/non-book classification
  - domain assignment
  - priority scoring
  - manual review hooks
  - checkpointed ingestion
- [fact] Completed ingestion is more balanced than the raw catalog skew.
  - Raw book catalog top domains: `dynamical_systems 694`, `skip 595`, `mathematics 293`, `physics 246`
  - Completed-book top domains: `software_engineering 123`, `machine_learning 74`, `numerical_methods 56`, `linear_algebra 52`
- [inference] As an **acquisition triage system**, the methodology is directionally sound.

### What Does Not Make Sense Yet

- [fact] The “book” gate is too permissive.
  - `71` entries in `catalog_books_r2.json` are still book-classified while their titles contain `solution`, `solutions manual`, or `instructor`.
  - Examples:
    - `Instructor's Solutions Manual to Thomas' Calculus`
    - `Solutions Manual and Supplementary Materials for Econometric Analysis...`
    - `Instructor’s Solution Manuals to Introduction to Electrodynamics`
- [fact] Suspicious non-book or low-signal titles still appear in book inventory and high-skew domains.
  - Examples from the 40-entry sample:
    - `eightsep.eps` as `dynamical_systems`
    - `IRLR_Solution_1.png` as `actuarial_insurance`
    - `Unknown`, `book_2`, and numeric-title entries with nonzero priority
- [fact] The largest live domain bucket is noisy.
  - `machine_learning` contains high-volume textbooks such as:
    - `Quantum Mechanics`
    - `Feynman Lectures`
    - multiple `Thomas Calculus` variants
    - `Neuroscience`
    - `Introduction to Electrodynamics`
- [fact] Raw catalog uncertainty is high.
  - `820` book entries are low confidence
  - `1,845` book entries are `manual_review`
- [fact] The multi-domain story is internally inconsistent.
  - `catalog_summary_r2.json` reports `311` `multi_domain_entries`
  - actual records do not carry `candidate_domains` or `multi_domain`
  - they do carry `secondary_domains`, but those are noisy and sometimes absurd, e.g. `linear_algebra` with secondary `rag_llm`
- [inference] The current classification/organization system is fit for **rough sorting and queueing**, not for treating catalog domain labels as trustworthy corpus truth.

### Classification Conclusion

- [inference] Keep the catalog pipeline, but demote its output from “domain truth” to “triage hypothesis” unless an entry passes stronger exclusion and relabeling checks.
- [inference] The repo should not take “36 domains populated” to mean “36 domains cleanly curated.”

## Chunking and Ingestion Assessment

### What Makes Sense

- [fact] The live extraction path is Docling-first:
  - `packages/pdf-tools/src/research_kb_pdf/docling_extractor.py`
  - `HybridChunker`
  - BGE tokenizer alignment
  - GROBID for paper metadata/citations
- [fact] Sampled chunk sizes are mostly sane.

| Sample Source | Source Metadata | Sampled Chunk Shape | Token p50 / p90 / max | Read |
|---|---|---|---|---|
| Machine Learning: A Probabilistic Perspective | `docling` | legacy `section_header` | `235 / 296 / 315` | good envelope |
| Wooldridge Solutions Manual | `docling` | legacy `section_header` | `255 / 299 / 321` | good envelope, questionable source inclusion |
| Econometric Analysis of Cross Section and Panel Data | `pymupdf` | new `section` + `chunking_method=docling` | `253 / 299 / 314` | source metadata drift |
| Introduction to Algorithms | `docling` | legacy `section_header` | `179 / 288 / 310` | good envelope |
| Options, Futures, and Other Derivatives | `docling` | legacy `section_header` | `237 / 299 / 315` | good envelope |
| Speech and Language Processing [draft] | `docling` | legacy `section_header` | `290 / 300 / 349` | higher-density but still under 512 |

- [fact] In the 6-source sample:
  - no sampled chunk exceeded `512` tokens
  - page provenance was complete in sampled chunks
  - duplicate content did not show up in the sampled first `1,200` chunks/source
- [inference] The chunk-size methodology itself is reasonable.

### What Does Not Make Sense Yet

- [fact] The live corpus has **two chunk metadata contracts**:
  - `1,102,056` chunks: `section_header`, no `chunking_method`
  - `292,807` chunks: `section` plus `chunking_method='docling'`
  - `526` chunks: `section` without `chunking_method`
- [fact] Source-level and chunk-level metadata are often inconsistent.
  - `1,144` sources report `extraction_method='docling'` but all chunks use the legacy `section_header` shape
  - `412` sources report `extraction_method='pymupdf'` while chunks carry Docling-style `section` and `chunking_method='docling'`
- [fact] Source heading summaries are unreliable.
  - `1,151` Docling sources have `total_headings = 0`
  - sampled Docling sources still had meaningful section headers in chunk metadata
- [fact] This inconsistency is explained by multiple ingestion paths:
  - dispatcher/docling path writes `section`, `heading_level`, `chunking_method`
  - bulk textbook/catalog ingestion path writes `section_header` and omits `chunking_method`
  - rechunking can convert chunk shape without updating source metadata
- [inference] The repo currently has a **good chunker** but not a **single trustworthy chunk schema**.

### Ingestion Validation Read

- [fact] Dispatcher unit tests pass: `23 passed`
- [fact] Large-PDF E2E ingestion is not clean:
  - `tests/e2e/test_ingestion_pipeline.py` failed `1` test locally
  - failure cause: test DB lacked `econometrics` in `domains`, causing FK failure on `sources.domain_id`
  - Docling and GROBID themselves completed before the FK failure
- [inference] The ingestion pipeline is more solid than the E2E fixture, but the test setup is still under-spec'd for full-domain ingest cases.

### Chunking Conclusion

- [inference] Chunking methodology is conceptually sound.
- [inference] The real problem is **normalization after chunking**, not chunk splitting itself.

## Validation Posture

### Current Local Results

| Check | Result | Read |
|---|---|---|
| `scripts/generate_status.py --check` | Pass | generated status is trustworthy |
| `scripts/audit_docs.py --ci` | Pass | fast doc checks are green |
| `ruff check packages scripts tests` | Pass | lint is clean |
| `scripts/mypy_baseline_check.py` | Pass | type baseline clean |
| `black --check packages scripts tests` | Fail | only `scripts/validate_classifier.py` would reformat; local WIP issue |
| `pytest packages/storage/tests/test_assumption_audit.py -q` | `55 passed` | prior assumption-audit failure cluster is no longer current |
| `pytest packages/ tests/ -m "integration ..."` | `326 passed` | broad DB-only integration is green |
| `pytest packages/pdf-tools/tests/test_dispatcher.py -q` | `23 passed` | dispatcher path covered |
| `pytest tests/e2e/test_ingestion_pipeline.py -q ...` | `1 failed, 10 passed, 1 deselected` | E2E gap remains |
| `python scripts/validate_classifier.py --limit 20` | Fail | broken local validator import path |

### Validation Assessment

- [fact] The repo’s current validation posture is **stronger than older March audit artifacts imply**.
- [fact] It is not uniformly green across all layers.
- [inference] The most important validation gap now is **methodology-specific**, not general CI hygiene:
  - classifier validation is broken in local WIP
  - E2E domain seeding is incomplete
  - doc audit coverage is narrower than the repo’s narrative surface

## Committed Baseline vs Local WIP Delta

### Committed Baseline

- [fact] Branch state: `main...origin/main [ahead 2]`
- [fact] Recent committed work includes:
  - `0f49739` docs/audit remediation
  - `0f0cb39` assumption-audit test contract fix
  - `bfa8be9` bulk library ingestion complete

### Local WIP

- [local WIP] modified: `docs/STRATEGIC_ASSESSMENT.md`
- [local WIP] modified: `scripts/mass_ingest_catalog.py`
- [local WIP] staged deletion: `scripts/classify_library_books.py`
- [local WIP] untracked: `scripts/validate_classifier.py`
- [local WIP] untracked: `codex-systematic-audit-2026-03-30.md`

### WIP Impact

- [local WIP] `scripts/mass_ingest_catalog.py` adds useful MEAP dedup and case-insensitive path remapping.
- [local WIP] `docs/STRATEGIC_ASSESSMENT.md` improves ingestion realism.
- [local WIP] The staged deletion of `scripts/classify_library_books.py` currently breaks the untracked `scripts/validate_classifier.py`.
- [inference] Local WIP slightly improves the ingestion story but currently makes the classifier-validation story worse. Do not treat it as baseline truth.

## Documentation Governance Assessment

*Merged from systematic audit (2026-03-30). Test results from that audit were pre-remediation and are not carried forward.*

### The Generated vs Narrative Split

- [fact] The repo has **two documentation tiers** with different trust levels:
  - **Generated surfaces** (`CURRENT_STATUS.md`, README auto-generated sections, `generate_status.py --check`): Reliable, DB-backed, always current.
  - **Narrative surfaces** (`ROADMAP.md`, `docs/INDEX.md`, `docs/CLI.md`, `docs/STRATEGIC_ASSESSMENT.md`, `CLAUDE.md`): Hand-maintained, prone to drift, not fully covered by `audit_docs.py`.
- [inference] The generated surfaces already show the right direction. The gap is extending that mechanism to the highest-traffic narrative pages.

### CLI and Graph-Default Narrative Inconsistency

- [fact] `packages/cli/src/research_kb_cli/commands/search.py` sets `use_graph=False` by default, but a docstring inside the same file still says graph-boosted search is enabled by default.
- [fact] `docs/CLI.md`, `docs/status/MIGRATION_GRAPH_DEFAULT.md`, and `docs/phases/phase3/ENHANCED_RETRIEVAL.md` still carry old `research-kb query` and graph-default narratives.
- [inference] This is a layered drift problem: legacy docs are stale, and one current code docstring is stale too.

### Stop Doing

- Stop hardcoding live corpus metrics in hand-maintained docs when a generated status source already exists.
- Stop marking phases "complete" if the corresponding verification gates or narratives are red.
- Stop expanding coverage gates, eval counts, or doc-alignment phases while the current trust surface is still inconsistent.
- Stop treating `CURRENT_STATUS.md` freshness alone as sufficient if the main landing pages contradict it.

### Roadmap-as-Ledger Problem

- [fact] `ROADMAP.md` behaves as a historical phase ledger while still presenting itself as a live roadmap.
- [fact] It claims `495` sources / `22` domains vs live `1,756` / `36` — a 3.5x gap.
- [inference] A top-level roadmap that lags the actual repo by an order of magnitude on corpus size is worse than no roadmap, because it teaches readers the wrong model of project maturity.
- [inference] Replace the phase-ledger with a live-priorities document that points to generated status for counts.

## Refined Roadmap

### Now

| Priority | Status | Item | Rationale | Dependency |
|---|---|---|---|---|
| 1 | Partial | Normalize chunk metadata contract | unify `section`/`section_header` and `chunking_method` across ingestion paths | no external dependency |
| 2 | Partial | Reconcile source metadata after rechunking | source-level `extraction_method` and `total_headings` are currently unreliable | depends on item 1 |
| 3 | Partial | Tighten catalog exclusion rules | solution manuals, placeholders, and image-like entries should not be ingestable books | no external dependency |
| 4 | Blocked by local WIP | Replace or repair classifier validator | current validator imports a deleted module | no external dependency beyond code cleanup |
| 5 | Partial | Fix narrative docs not covered by `audit_docs.py` | `docs/CLI.md`, `docs/INDEX.md`, README eval counts, ROADMAP adaptive chunking | no external dependency |
| 6 | Partial | Seed full domain set in E2E test DB | large-PDF E2E ingestion currently fails on missing FK target domains | test-fixture change only |

### Next

| Priority | Status | Item | Rationale | Dependency |
|---|---|---|---|---|
| 1 | Not started | Re-audit live domain labels, starting with `machine_learning` and `dynamical_systems` | current largest buckets are visibly noisy | items 2-4 in “Now” |
| 2 | Not started | Add post-ingest quality checks for title plausibility and manual/solutions leakage | prevents polluted corpus growth | item 3 in “Now” |
| 3 | Not started | Expand `audit_docs.py` scope | current pass status overstates narrative-doc health | item 5 in “Now” |
| 4 | Not started | Separate “acquisition queue” labels from “corpus truth” labels | current catalog labels are being asked to do both jobs | item 1 in “Next” |
| 5 | Not started | Add chunk-shape/status metrics to generated status or dashboard | makes chunk-contract drift visible instead of hidden | item 1 in “Now” |

### Later

| Priority | Status | Item | Rationale | Dependency |
|---|---|---|---|---|
| 1 | Blocked | Restore `chunk_concepts` via KG re-extraction | required for graph-backed retrieval to mean what docs say it means | Anthropic credits + stable corpus |
| 2 | Blocked | Re-enable graph as a trustworthy default candidate | only after chunk grounding is real again | KG restoration |
| 3 | Not started | Multi-hop reasoning chains | currently premature without grounded chunk-concept links | KG restoration |
| 4 | Not started | Temporal reasoning / contradiction detection | valuable, but downstream of graph/data trust cleanup | KG restoration |

### Roadmap Read

- [inference] The next best work is not “more features.”
- [inference] The next best work is **methodology normalization**:
  - corpus truth
  - chunk truth
  - doc truth
  - validator truth

## Assumptions and Defaults Used

- [fact] This report treats live DB state and runnable commands as more trustworthy than hand-maintained docs.
- [fact] This report treats local WIP separately from committed baseline.
- [fact] No code, schema, or doc edits were made beyond creating this audit artifact.
- [inference] Older audits were used only as historical context, not as evidence.

## Command and Evidence Appendix

| Command | Result | Why It Matters |
|---|---|---|
| `git status --short --branch` | `main...origin/main [ahead 2]`, 5 local WIP paths | establishes dual-track scope |
| `git log -n 15` | confirms active March 27-30 ingestion/audit work | proves recent progress is real |
| `./.venv/bin/python scripts/generate_status.py --check` | pass | generated status is trustworthy |
| `./.venv/bin/python scripts/audit_docs.py --ci` | pass | fast doc audit is green, but narrow |
| `./.venv/bin/ruff check packages scripts tests` | pass | lint posture is clean |
| `./.venv/bin/black --check packages scripts tests` | fail on `scripts/validate_classifier.py` only | local-WIP formatting issue |
| `./.venv/bin/python scripts/mypy_baseline_check.py` | pass | type baseline clean |
| `./.venv/bin/pytest packages/storage/tests/test_assumption_audit.py -q` | `55 passed` | older assumption-audit failure conclusion is stale |
| `./.venv/bin/pytest packages/ tests/ -m "integration and not requires_embedding and not requires_ollama and not requires_reranker" -q --maxfail=20` | `326 passed` | broad DB-only integration is green |
| `./.venv/bin/pytest packages/pdf-tools/tests/test_dispatcher.py -q` | `23 passed` | dispatcher path healthy |
| `./.venv/bin/pytest tests/e2e/test_ingestion_pipeline.py -q -m "e2e and not requires_embedding and not requires_ollama and not requires_reranker"` | `1 failed, 10 passed, 1 deselected` | E2E fixture/domain seeding gap remains |
| `./.venv/bin/python scripts/validate_classifier.py --limit 20` | `ModuleNotFoundError: classify_library_books` | local validator broken |
| DB query: `SELECT COUNT(*) ... FROM chunks GROUP BY metadata shape` | `1,102,056` legacy vs `292,807` new-shape docling chunks | proves split chunk contract |
| DB query: source/chunk contract mismatch counts | `1,144` docling sources with legacy chunk shape; `412` pymupdf sources with docling chunks | proves source-level metadata drift |
| YAML inspection of `fixtures/eval/retrieval_test_cases.yaml` | `107` cases / `31` domains | disproves README’s `108 / 36` retrieval-eval claim |

## Final Audit Read

- [inference] The repo’s methodology **mostly makes sense** at the systems level.
- [inference] It **does not yet make enough sense** at the normalization layer:
  - catalog labels are not clean enough to be treated as domain truth
  - chunk metadata is not normalized enough to be treated as one schema
  - narrative docs are not normalized enough to be treated as one story
- [inference] If the next wave of work fixes those three normalization problems, the project will become much easier to trust, maintain, and extend than it is today.
