# 0001 — research-kb scope: retrieval + citation backbone + assumption auditing (KG ambition retired)

- **Status**: Accepted
- **Date**: 2026-06-11
- **Deciders**: Brandon (research-side design review 2026-06-10/11)
- **Canonical register**: `~/Claude/lever_of_archimedes/docs/plans/active/2026-06-research-side-design-review/decisions.yaml` (rows R2, R3, R7)

## Context

research-kb's chunk-level knowledge graph degraded as the corpus grew 3.5×: `chunk_concepts` references
pre-growth chunk IDs, concepts exist for only 3/36 domains, and re-extraction costs $150–300 (blocked on
API credits). The graph-backed MCP tools silently fall back to weaker results — a fail-loud violation.
Meanwhile the telemetry census (review `evidence.md#1`, as-of 2026-06-10, this-Linux-machine scope) shows
**zero lifetime calls** to every concept-graph tool, against 129 `research_kb_search` calls and light
citation-tool use. synthesis-kb's claim-level concept layer — eval-gated (F1 0.72–0.91) — now serves the
cross-domain-bridges goal that motivated the chunk-level KG.

Separately, synthesis-kb drops concepts whose evidence cites papers research-kb does not hold
(zero-anchor drops, #22/#23): measured demand is single-digit named arXiv papers per domain.

## Decision

1. **Scope**: research-kb is the **primary-literature retrieval + citation backbone + assumption-auditing**
   service. The chunk-level concept-graph ambition is retired; **synthesis-kb is the concept layer**.
2. **Retirement tracks data integrity, not usage alone**:
   - The stale-concept-layer tools (`graph.py` family: `graph_neighborhood`, `graph_path`,
     `explain_connection`, `cross_domain_concepts`, concept lookups) are **disabled or
     degradation-labeled** (fail-loud) — slice RS4.
   - **`audit_assumptions` stays** (intact methods/assumptions tables: 15,505/9,468 rows, 88.8% enriched;
     its zero usage may be discoverability — fixed with #24).
   - Citation tools stay (intact, lightly used).
3. **#24 docstring fix runs in both directions**: stop understating domain coverage (36 domains) AND stop
   overstating graph capability.
4. **KG re-extraction backlog CLOSED.** KuzuDB + sync-service removal evaluated in RS4 (it accelerates a
   layer that is retiring; Postgres fallback exists).
5. **Acquisition = wanted-list contract** (R2): synthesis-kb emits a first-class wanted-primaries report
   (formalized `unresolved_*.tsv`); after #25 migrates the S2 pipeline here, `ingest_missing_papers`
   consumes it. The four #22 papers ingest now via the existing pipeline. #23 non-arXiv islands remain
   deliberate per-domain ingestion decisions — no bulk cache ingestion (M0 of the 2026-05-21 design is
   formally dropped; `inbox/` is deleted in slice RS1).
6. **Mac access** (R7): none built. Recorded enabling fact: both machines share a LAN always — on first
   real Mac research-query need, bind the MCP/daemon/API on the LAN and register user-scope MCP on the Mac.

## Falsifier

A concrete chunk-level-graph use case — a query class that claim-level concepts (synthesis-kb) cannot
serve — reopens the KG decision. Recheck by 2026-12-31.

## Consequences

- Honest tool surface: agents never get quietly weak graph results.
- Fewer moving parts: KuzuDB/sync possibly removed; 4-way hybrid search officially 3-way (FTS+vector+citation).
- The $150–300 re-extraction line item disappears.
- This directory (`docs/decisions/`) is new — per hub ADR-0003, decisions live per-repo with a hub index
  (hub ADR-0008).

## Execution (RS4, executed 2026-06-13)

R3 was executed across research-kb (record: hub roadmap RS4 row; research-kb #27/#24 close-outs):

- **Stale MCP tools → fail-loud retirement stubs** (registered, return a RETIRED redirect to
  synthesis-kb — never silently-empty graph results): `graph_neighborhood`, `graph_path`,
  `cross_domain_concepts`, `explain_connection`, `list_concepts`, `get_concept`, `chunk_concepts`,
  `find_similar_concepts`, plus `literature_review` (it seeded from the stale graph).
- **`audit_assumptions` + citation tools kept** (intact, unchanged).
- **KuzuDB removed** (confirmed dead — no DB file, no sync cron/timer): deleted `kuzu_store.py`,
  `sync_kuzu.py`, `graph_queries.py`, `cross_domain.py`, `synthesis.py`, `literature_review.py`,
  `query_extractor.py`, daemon `warmup.py`; dropped the `kuzu` dependency; stripped the
  `use_graph`/graph-score path so `research_kb_search` is **FTS + vector + citation (3-way)**.
  Graph surfaces removed on the API (`/graph`, `/concepts` routes), CLI (`graph`, `review` commands),
  dashboard (concept page) and daemon (`graph_path` method).
- **Concept DATA preserved (dormant), not dropped**: the Postgres `concepts` /
  `concept_relationships` / `chunk_concepts` tables + their data (4 domains) + the
  `ConceptStore`/`RelationshipStore`/`ChunkConceptStore` CRUD + the concept-ingest scripts remain,
  so reopening (the falsifier) stays cheap. No table-drop migration was issued.
- **#24 fixed both directions** (closed): search/sources/API docstrings made domain-agnostic, naming
  `research_kb_list_domains` as the source of truth (stop understating the 38-domain corpus); the
  "knowledge graph signals" claim + `use_graph` parameter removed from `research_kb_search` (stop
  overstating graph capability).
- **KG re-extraction backlog CLOSED**: `docs/STRATEGIC_ASSESSMENT.md §7` and
  `docs/status/MIGRATION_GRAPH_DEFAULT.md` bannered superseded; the $150–300 line item dropped.

Recorded residual (not a silent fallback): `SearchResult.graph_score` (contract field) and storage
`SearchQuery.use_graph`/`graph_weight`/`max_hops` remain inert/dormant (always 0 / ignored) to avoid a
cross-package contract ripple; a few dead formatters and the unused `KUZU_WARMUP_*` metric objects were
left in place. None are reachable as a weak-result path.
