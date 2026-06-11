> **SUPERSEDED (2026-06-11)** by the research-side design review:
> `~/Claude/lever_of_archimedes/docs/plans/active/2026-06-research-side-design-review/decisions.yaml`
> (locally: `docs/decisions/0001-scope-retrieval-citation-backbone.md`). M0 (cache-inbox consumption)
> was never built and is now **formally dropped** — acquisition is the wanted-list contract (review R2);
> `inbox/` is deleted in slice RS1. Historical rationale only below.
>
> ~~**Synchronized copy.** Canonical version: `~/Claude/research-agent/docs/plans/active/synthesis_kb_migration_2026-05-21.md`.~~
> (research-agent is archived; both copies now historical.)

---

# Design Analysis: A Personal Synthesis Layer Above research-kb

## Context

You asked whether research-agent's architecture made sense compared to a collection of skills. The clarifying turn opened a bigger question: how should literature review of one topic, synthesis across domains, and the existing repos all tie together — with the goal of building **a knowledge graph and citation network among your synthesized dossiers** that links back to research-kb's primary sources.

The reframe: this is not a refactor of research-agent. It is the design of a **synthesis-kb** — a sibling RAG to research-kb that indexes your dossiers (synthesized claims) while keeping primary-literature content strictly separated. research-agent's question becomes downstream and largely answered.

---

## 1. Premises Established in This Session

1.1  **Epistemic separation is non-negotiable.** Primary literature (research-kb) and synthesized claims (synthesis-kb) must not be conflated in retrieval. They live in separate repos and separate databases.

1.2  **research_toolkit already produces the inputs.** The v3 strict-live regime gives high-fidelity dossiers with `claim_graph.jsonl`, `evidence_ledger.yml` (SHA-256 byte-anchor verification), `bib_ledger.yml`, and `/research-kb-export` writing to `~/Claude/research-kb/inbox/research_toolkit/`. Nothing currently consumes the inbox; that is the gap.

1.3  **research-agent never got used.** Its LangGraph orchestration was built for an unattended-batch use case that did not materialize. Decisions about it follow from the synthesis-kb design, not the other way around.

1.4  **Five workflows drive the design (all wanted, plus visualization):**
   - (1.1) Search prior claims when starting a new dossier
   - (1.2) Find related dossiers via shared primary citations
   - (1.3) Cross-claim consistency audit
   - (1.4) Cross-domain concept bridges
   - (1.5) Visualize the dossier knowledge graph

1.5  Conclusion: synthesis-kb must support embeddings, graph traversal, citation network analysis, claim-pair comparison, and graph export for visualization. This is real KB infrastructure, not a script.

---

## 2. The Four-Repo Architecture (target state)

| Repo | Layer | Content | Role |
|------|-------|---------|------|
| `research-kb` | Primary literature | Papers, textbooks, code repos | Source of truth for what the literature says (untouched) |
| `synthesis-kb` (new) | Personal synthesis | Dossiers, claims, evidence, concepts extracted from claims | Source of truth for what *you* have synthesized |
| `research_toolkit` | Dossier production | Skills + validators + cache | Builds dossiers; exports to synthesis-kb's inbox |
| `research-agent` | Legacy | (LangGraph pipeline) | To be retired or repurposed (§7) |

2.1  **Flow direction is asymmetric.** research_toolkit → synthesis-kb (writes via `/synthesis-kb-ingest`). synthesis-kb → research-kb (reads-only, for chunk resolution and citation network linkage). No edges go back from primaries to synthesis.

2.2  **Storage isolation** (decided): synthesis-kb is its own repo with its own Postgres + pgvector + Kuzu + MCP server. Mirrors research-kb's structure to maximize code reuse and minimize learning curve.

---

## 3. synthesis-kb Schema (sketch)

Mirrors research-kb's design with synthesis-specific entities. JSONB extensibility kept throughout (matches research-kb pattern at `schema.sql:130-141`).

```sql
-- Core entities
dossiers            (id, topic_slug, version, title, generated_at,
                     evidence_ledger_path, claim_graph_path,
                     embedding vector(1024), metadata JSONB)
claims              (id, dossier_id, claim_type, text, status,
                     confidence_score, embedding vector(1024),
                     fts_vector tsvector, metadata JSONB)
evidence            (id, claim_id, ...identity fields below... )
entities            (id, entity_type, canonical_name, aliases[],
                     metadata JSONB)
claim_entities      (claim_id, entity_id)  -- junction

-- Concept graph (mirrors research-kb's Phase 2 KG)
dossier_concepts    (id, name, canonical_name, concept_type,
                     embedding, source_dossier_ids[], confidence,
                     research_kb_concept_uuid UUID NULL)  -- FK to research-kb concept when names align (resolved at ingestion)
concept_relationships (source_id, target_id, type, evidence)

-- Cross-domain bridges (first-class, novel to synthesis-kb)
bridges             (id, claim_a_id, claim_b_id, bridge_type,
                     evidence, link_confidence, discovered_at)
```

3.1  **Evidence identity model** (Q2 answered by workflow requirements). Every evidence record stores three identifiers so faithfulness can be verified locally and retrieval can survive research-kb changes:

```yaml
evidence record:
  display:
    bibkey: chernozhukov2018dml          # human-readable
    page: 12
  truth:
    cache_id: cache_8a3f4e               # research_toolkit cache pointer (immutable)
    text_path_offset: [4821, 4862]
    sha256_of_span: 8a3f4e...            # cryptographic anchor
    extraction_method: verbatim_match
    link_confidence: 0.98
  retrieval:
    research_kb_source_uuid: 7f3a-...    # stable (source-level)
    research_kb_chunk_uuid: 8c2d-...     # nullable; re-resolvable on drift
```

The principle: **synthesis-kb owns the cryptographic truth-anchor; research-kb chunk UUID is a derived index that can be re-bound** via `bibkey + sha256_of_span` lookup if research-kb re-ingests a paper. No data loss across primary-KB churn.

3.2  **Ingestion scope** (Q3 answered): full graph expansion + concept extraction over claims. Each dossier → tables. Claim text embedded + FTS-indexed + concept-extracted. Files stay canonical; ingestion is re-runnable.

---

## 4. synthesis-kb MCP Tool Surface

Mirrors research-kb's 22 tools where the operation makes sense over synthesized content, adds three novel tools for the synthesis use case.

**Mirrored from research-kb:**

| Tool | Operation |
|------|-----------|
| `synthesis_kb_search` | BM25 + vector over claims (workflow 1.1) |
| `synthesis_kb_fast_search` | Vector-only quick lookup |
| `synthesis_kb_list_dossiers` | Enumerate dossiers (replaces list_sources) |
| `synthesis_kb_get_dossier` | Dossier detail + claim list |
| `synthesis_kb_get_claim` | Claim detail + evidence chain |
| `synthesis_kb_concept_neighborhood` | Concept graph traversal (workflow 1.4) |
| `synthesis_kb_find_similar_concepts` | Embedding-based concept discovery |
| `synthesis_kb_biblio_coupling` | Dossier-dossier shared-primaries (workflow 1.2) |
| `synthesis_kb_citation_network` | Dossier → primaries directed graph |
| `synthesis_kb_stats` | Corpus metrics |
| `synthesis_kb_health` | Operational |

**Novel to synthesis-kb:**

| Tool | Operation |
|------|-----------|
| `synthesis_kb_claim_provenance` | Resolve a claim to its full evidence chain → research-kb chunks (workflow 1.1, 1.3) |
| `synthesis_kb_consistency_check` | Pair-wise claim comparison (LLM-based) for contradictions (workflow 1.3) |
| `synthesis_kb_explain_bridge` | Generate path between dossiers via shared concepts/citations (workflow 1.4) |
| `synthesis_kb_export_graph` | Export subgraph as JSON for visualization (workflow 1.5) |

4.1  Cross-KB calls (synthesis → research-kb) happen only inside `claim_provenance` and during ingestion-time URI resolution. synthesis-kb is otherwise self-contained.

---

## 5. Visualization Layer (workflow 1.5)

5.1  **M4 deliverable: Streamlit dashboard in `synthesis-kb/packages/dashboard/`**, mirroring research-kb's existing dashboard pattern (`research-kb/packages/dashboard/`). Reads from synthesis-kb's Postgres + Kuzu directly. Local-first; no hosting.

5.2  **Views:**
   - Dossier graph (force layout) — nodes are dossiers, edges are bibliographic coupling weight
   - Concept-bridge map — concepts across dossiers, edges from `bridges` table
   - Provenance trace — for a selected claim, show evidence chain → primary chunks
   - Citation network — your dossiers (top layer) → primary sources (bottom layer)

5.3  **Static export via `synthesis_kb_export_graph` MCP tool** — emits Cytoscape JSON, Mermaid, GraphML. This tool is the shared boundary: Streamlit consumes it; any future frontend (see §5.4) consumes the same output. No data lock-in.

5.4  **Future: public web visualization on `brandon-behring.dev`** (tracked as a goal, not part of M1–M6):
   - Site is Astro v6 + Cloudflare Workers (`brandon-behring/brandon-behring.dev`)
   - Integration approach: new route (e.g., `/knowledge`) that renders the synthesis graph client-side via Cytoscape.js
   - Data flow: build-time sync script pulls `synthesis_kb_export_graph` output → commits Cytoscape JSON to the Astro repo → static page renders it. Keeps the site static; no runtime DB exposure or Worker proxy needed.
   - Privacy filter: a `--public` flag on the export tool emits only dossiers/claims marked `visibility: public`. Default is private; opt-in for the website.
   - **Followup**: file a tracking issue on `brandon-behring/brandon-behring.dev` after plan mode exits. Issue title: "Personal synthesis map integration (consumer of synthesis-kb export)". Body: link to synthesis-kb repo when scaffolded, MCP tool name, recommended Cytoscape.js render approach, public-only filter requirement.

---

## 6. Ingestion Pipeline (two-output flow)

6.1  **Each research_toolkit project produces two outputs**, destined for two different KBs:

| Output | Source | Destination | Skill |
|--------|--------|-------------|-------|
| Raw cached blobs (HTML/PDF/txt) | `<project>/cache/` | `~/Claude/research-kb/inbox/research_toolkit_caches/` | `/research-kb-cache-export` (new) |
| Claim graph + evidence ledger | `<project>/claim_graph.jsonl`, `evidence_ledger.yml`, `bib_ledger.yml` | `~/Claude/synthesis-kb/inbox/research_toolkit_dossiers/` | `/synthesis-kb-export` (renamed from `/research-kb-export`) |

6.2  **research-kb ingests primary sources** (consumer of the cache inbox):
   - Read `cache_manifest.yml`; for each cached blob, extract bibkey + URL + mime_type
   - Run existing PDF/HTML → Docling → chunks → embeddings pipeline (`research-kb`'s `pdf-ingestion` skill)
   - Populate `sources`, `chunks`, `citations`, optionally extract concepts via `concept-extraction` skill
   - Idempotent on `file_hash` (existing `sources.file_hash UNIQUE NOT NULL` invariant at `schema.sql:25`)
   - **Net effect**: research-kb is now grown by `/research-gather` runs, not just manual PDF drops

6.3  **synthesis-kb ingests dossiers** (consumer of the dossier inbox):
   - **Validate** — re-run `validators/claim_graph.py` and `validators/evidence_ledger.py`. Reject if v3 strict-live fails.
   - **Resolve primaries** — for each evidence record, look up `bibkey` → research-kb `sources.id` (require research-kb to be populated first via 6.2). Best-effort match `sha256_of_span` → `chunks.id`. Store the three-identifier evidence (display/truth/retrieval per §3.1).
   - **Expand into tables** — upsert dossier, claims, evidence, entities. Embed claim text.
   - **Extract concepts** — concept extraction over claim text using the same taxonomy as research-kb (METHOD, ASSUMPTION, DEFINITION, THEOREM, ...). Reuse research-kb's `concept-extraction` skill body with a prompt fork tuned for synthesized text (claims are more abstract than primary chunks).
   - **Resolve concept links** — for each extracted concept, attempt string/alias match against `research-kb.concepts`. Populate `dossier_concepts.research_kb_concept_uuid` where match found; leave null otherwise. Re-runnable as research-kb grows.
   - **Build relationships** — concept-concept edges (within dossier) + cross-dossier shared-concept edges.
   - **Compute bridges** — pair-wise concept embedding similarity across dossiers above a threshold; populate `bridges`.

6.4  **Ordering constraint** — cache export and synthesis export from a project should run in this order: `/research-kb-cache-export` → research-kb ingestion → `/synthesis-kb-export` → synthesis-kb ingestion. The second-half ingestion needs research-kb populated to resolve evidence URIs. If a cache hasn't been ingested into research-kb yet, the synthesis-kb evidence record stores the bibkey + sha256 anchor and marks `research_kb_source_uuid` as null with a `pending_resolution` flag, resolvable later.

6.5  **Re-ingestion** — both ingestion skills are idempotent. research-kb dedupes on `file_hash`. synthesis-kb upserts on `(topic_slug, version)` for dossiers, and on stable IDs (`claim_id`, `evidence_id`, `entity_id`) for sub-records. Dossier files and cache blobs remain canonical; both KBs are re-buildable from the filesystem.

6.6  **Naming migration** — `/research-kb-export` skill in research_toolkit is renamed `/synthesis-kb-export`. Keep the old name as a deprecated alias for one release cycle. New skill `/research-kb-cache-export` added. Inbox path under `~/Claude/research-kb/inbox/research_toolkit/` (existing) holds claim graphs until migration; a one-time `mv` relocates them to `~/Claude/synthesis-kb/inbox/research_toolkit_dossiers/`.

---

## 7. research-agent: Resolution

7.1  Under this architecture, research-agent's analytical capabilities map to synthesis-kb's MCP tools and/or research_toolkit skills:

| research-agent node | New home |
|---------------------|----------|
| `query_planner` | research_toolkit's `/research-plan` (exists) |
| `literature_search` | research-kb's `research_kb_search` (exists) |
| `concept_explorer` | `synthesis_kb_concept_neighborhood` + `synthesis_kb_find_similar_concepts` |
| `citation_analyzer` | research-kb's `research_kb_citation_network` (exists) + `synthesis_kb_biblio_coupling` |
| `assumption_auditor` | research-kb's `research_kb_audit_assumptions` (exists, primary-side) + a new synthesis-side audit |
| `connection_explorer` | `synthesis_kb_explain_bridge` |
| `synthesis` | research_toolkit's `/research-synthesize` (to be added) or invoked manually with claim_graph as input |

7.2  Recommendation: **archive research-agent**. Keep the repo as a reference for the parallel-fan-out pattern and the 451-test corpus. Do not maintain. The orchestration is not needed because:
   - synthesis-kb is queryable via MCP tools individually (composable, no graph needed)
   - research_toolkit handles dossier production (no need to "decompose query → search → analyze → synthesize" as one pipeline)
   - Interactive use through Claude Code skills replaces the unattended-batch use case

7.3  If a programmatic CLI is ever desired, write a thin shell script that chains MCP tool calls — no LangGraph, no LangChain.

---

## 8. Build Sequence

A staged build ships value at each milestone. Do not try to build all five workflows at once.

| Milestone | Scope | Workflows enabled | Effort estimate |
|-----------|-------|-------------------|-----------------|
| **M0: Cache ingestion side** | research_toolkit gains `/research-kb-cache-export` skill (raw blobs + manifest to `~/Claude/research-kb/inbox/research_toolkit_caches/`). research-kb's existing `pdf-ingestion` skill adapted to consume the cache inbox (idempotent on `file_hash`). Run on one project to populate research-kb with its primaries. | None yet — primes research-kb so synthesis-kb evidence can resolve | ~1 week |
| **M1: synthesis-kb foundations** | synthesis-kb repo scaffold (Postgres + pgvector + Kuzu + FastMCP, mirrored from research-kb). Schema (§3). Rename `/research-kb-export` → `/synthesis-kb-export` (deprecate alias). New `/synthesis-kb-ingest` skill. Ingest one existing dossier end-to-end with evidence resolved to research-kb chunks from M0. MCP tools: list_dossiers, get_dossier, get_claim, stats. | None yet — validates end-to-end resolution | 1–2 weeks |
| **M2: Search + coupling** | Embed claim text, FTS index, MCP search + biblio_coupling. Resolve evidence to research-kb URIs. | (1.1) prior-claims search, (1.2) related dossiers | 1 week |
| **M3: Concept graph** | Concept extraction over claims, dossier_concepts + relationships, bridges precomputed. MCP concept_neighborhood + explain_bridge. | (1.4) cross-domain bridges | 1–2 weeks |
| **M4: Visualization** | Streamlit dashboard, export_graph tool, Mermaid export. | (1.5) visualize | 1 week |
| **M5: Consistency audit** | LLM-based claim-pair comparison, contradiction detection, scheduled audit reports. | (1.3) consistency audit | 1–2 weeks |
| **M6: Retire research-agent** | Archive repo with README pointer to synthesis-kb. | — | 1 day |

8.1  M1 is the discipline check: if ingesting one dossier and querying it does not feel useful, the whole plan is suspect. Cheap to abandon at M0/M1.

---

## 9. Decisions Resolved + Still Open

**Resolved in this planning session:**

| ID | Decision | Choice |
|----|----------|--------|
| Q1 | Storage isolation | Separate repo + DB (`~/Claude/synthesis-kb/`) |
| Q2 | Ingestion location + inbox path | Two-output split — `/research-kb-cache-export` → research-kb inbox; `/synthesis-kb-export` → synthesis-kb inbox. Consumer in each KB. (§6) |
| Q3 | Cross-domain pattern | Both — `synthesis_kb_explain_bridge` MCP tool + `/cross-domain-synthesize` capture skill that produces bridge dossiers (§4 + research_toolkit) |
| Q4 | Visualization | Streamlit dashboard (M4) + `synthesis_kb_export_graph` MCP tool. Future: `brandon-behring.dev` Astro integration via build-time sync (§5.4) |
| Q5 | Concept extraction approach | Linked graphs — synthesis-kb extracts independently using research-kb's taxonomy; FK (`research_kb_concept_uuid`) populated at ingestion when names align |
| Q6 | research-agent fate | Archive — README pointer to synthesis-kb, no maintenance (§7.2) |
| Q7 | Source of truth | Split — files (claim_graph, evidence_ledger, bib_ledger, dossier markdown) are canonical inputs; concepts, bridges, embeddings, similarity scores are DB-only derived data. One-way sync files → DB. |
| Q8 | Re-ingestion behavior | Upsert by stable IDs (claim_id, evidence_id, entity_id) + status lifecycle. Removed claims → `status='stale'` (soft delete). Claim history lives in git on canonical files. |
| Q9 | Ingestion trigger pattern | Wrapper `/dossier-publish <project>` chains all four steps (cache-export → research-kb-ingest → synthesis-export → synthesis-ingest). Individual skills remain available for debugging. Idempotent — safe to re-run after partial failure. |

**Explicitly deferred with decision criteria:**

9.1  **Q9.1 — consistency audit mechanism (workflow 1.3).** Open question pending cost estimates. Candidates retained:
   - Embedding filter + LLM confirm (preferred default)
   - LLM-only (Claude structured output)
   - NLI model (local DeBERTa entailment head)
   - Manual flagging only (lowest M5 scope)
   - **Decision criterion**: cost estimate of LLM-call volume at ~50–100 dossiers × ~50 claims each, given embedding-filter recall. Decide at M5 design start, after M2 search workload is observed.

9.2  **Q9.2 — bridge computation timing.** Open, kept under active consideration. Interim recommendation for M3:
   - **Cheap bridges at ingestion** — string/alias-match shared concept names + shared primary citations + concept-embedding-similarity above threshold. O(N²) on cheap ops; fast for current scale (<10 dossiers).
   - **LLM-confirmed bridges (analogous-method, contradiction) deferred to on-demand** — computed live by `synthesis_kb_explain_bridge` for explicitly requested dossier pairs only. Not precomputed.
   - **Re-evaluation trigger**: dossier count > 30, OR ingestion latency > 30s for a new dossier. At that point, consider moving expensive bridges to a `runpod-deploy`-orchestrated batch job triggered by a scheduled cron or post-ingestion hook.

9.3  **Q9.3 — concept extractor model.** Tiered options, decision pending cost estimates:
   - **Tier 1 (default)**: Reuse research-kb's Ollama llama3.1:8b. Local; free; matches existing pipeline.
   - **Tier 2 (upgrade path)**: `runpod-deploy`-orchestrated GPU pod running a larger extractor model (e.g., llama3.1:70b or fine-tuned model). User owns the `runpod-deploy` package — clean integration path. Useful if Tier 1 quality is insufficient on synthesized text.
   - **Tier 3 (selective)**: Claude (Haiku/Sonnet) escalation for low-confidence concepts identified by Tier 1.
   - **Decision criterion**: extraction quality on a labeled sample of claims from existing dossiers. Compare Tier 1 vs. Tier 2 vs. Tier 3 on precision/recall against gold concepts. Decide at M3 design start.

9.4  **Q9.4 — visibility model for public/website export.** Open, pending brandon-behring.dev integration concretization. Candidates retained:
   - Per-dossier `visibility: public|private`
   - Per-claim visibility
   - Per-dossier default + per-claim override
   - **Decision criterion**: at the time of bdev integration design (post-M6, after the tracking issue in §12.1 surfaces real requirements), pick the granularity matching actual sharing intent. Schema impact is small (one nullable column on `dossiers` or `claims`); non-breaking addition.

9.5  **Pre-M3 cost-estimate task.** Before M3 begins, run a one-off measurement:
   - Sample 5 representative claims from an existing dossier
   - Time + cost Ollama extraction (Tier 1) vs. one runpod-deploy run with llama3.1:70b (Tier 2) vs. Claude Haiku extraction (Tier 3)
   - Time + cost embedding-filter + LLM-confirm on 100 sampled claim pairs (Q8.1 mechanism)
   - These data points settle Q9.1, Q9.3, and inform Q9.2's re-evaluation threshold.

---

## 10. Verification

10.1  **At M1 end** — ingest one of your existing dossier projects. Confirm:
   - `claim_graph.jsonl` round-trips into tables and back
   - Every evidence record resolves to a research-kb source UUID (or is flagged as unresolved)
   - `synthesis_kb_get_dossier` returns the dossier with all claims
   - SHA-256 anchors verify against research_toolkit cache (`/citation-audit` passes against synthesis-kb's view)

10.2  **At M2 end** — search for a claim term you remember writing. Confirm the right claim ranks at top. Run biblio coupling; confirm dossiers known to share a paper come up.

10.3  **At M3 end** — run `explain_bridge` between two dossiers from genuinely different domains (e.g., causal inference and RL). Inspect the bridge: is it a real conceptual connection, or noise?

10.4  **At M4 end** — open the dashboard. Can you find your way around your own synthesis?

10.5  **At M5 end** — flag a known contradiction (insert one deliberately). Does the audit catch it?

---

## 11. Synthesis

11.1  The reframed task is the design of `synthesis-kb` as a personal-synthesis-layer sibling to research-kb. Architecture mirrors research-kb to maximize reuse; content type and MCP tools are synthesis-specific.

11.2  Most of the inputs already exist (research_toolkit's v3 evidence regime is exactly the provenance layer you need). The gap is on the ingestion + query side, which is what synthesis-kb provides.

11.3  research-agent is downstream of this decision: archive it. Its capabilities re-home as synthesis-kb MCP tools and research_toolkit skills.

11.4  The staged build (M1 → M6) lets you abandon at M1 if the workflow doesn't pay off. The architecture is comprehensive but the first deliverable is small.

11.5  **Confidence: medium-high.** The mirror-architecture decision is well-supported because (a) research-kb's design is proven, (b) the same primitives (BM25 + vector + graph + PageRank) serve all five workflows, (c) research_toolkit's existing exports drop straight in once the export skill is split into cache + dossier variants. Risk is in workflow 1.3 (consistency audit) — pushed to M5 so it does not gate the rest.

---

## 12. Post-Plan Followups

These are external-write actions that cannot happen inside plan mode. Execute after `ExitPlanMode`:

12.1  **File tracking issue on `brandon-behring/brandon-behring.dev`** — see §5.4 for content. Title: "Personal synthesis map integration (consumer of synthesis-kb export)". Body should reference the M4 deliverable (`synthesis_kb_export_graph` MCP tool emits Cytoscape JSON), the proposed Astro route (`/knowledge` or `/synthesis-map`), the build-time sync pattern (keeps the site static), and the privacy filter requirement (only `visibility: public` dossiers export). Mark as long-term goal, no deadline.

12.2  **No other repos require issues at this stage.** synthesis-kb does not exist yet; research-kb and research_toolkit changes are tracked in this plan and will be implemented during M0–M6 directly.

---

## 13. Decisions to Revisit as the System Matures

These questions surfaced during planning but were not resolved — either because they depend on real usage data or because they only matter once the system is partially built. Revisit each at the milestone where the decision becomes actionable.

| ID | Question | Trigger to revisit |
|----|----------|---------------------|
| Q10 | **Bibkey ↔ source UUID reconciliation** — how is the mapping from `bib_ledger.yml` bibkeys (e.g., `chernozhukov2018dml`) to research-kb `sources.id` UUIDs maintained? Auto-discovered (string + DOI match)? Manual map file? Hybrid with manual override? | Late M0 / early M1 — when implementing `/synthesis-kb-ingest`'s "Resolve primaries" phase |
| Q11 | **Dossier-audit findings flow** — research_toolkit's `/dossier-audit` produces DROP/CORRECT/FLAG records. Do these become annotations on synthesis-kb claims (new `audit_findings` table), separate audit records, or stay file-only? | During M5 design — when consistency audit is implemented; the two audit flows may share a table |
| Q12 | **Cross-pipeline source deduplication** — if research-kb has a manually-ingested textbook PDF and research_toolkit later caches the same content via web (HTML), how is dedup handled? `file_hash` won't match (different formats). Same DOI? Same title+authors? Manual link? | When the first duplicate is observed — likely M2 or later |
| Q13 | **Concept-name disambiguation across domains** — when synthesis-kb extracts "convergence" from a causal-inference dossier and "convergence" from a measure-theory dossier, are they the same concept (one node) or different (homonym distinction)? | M3 — when concept extraction is implemented and the first homonym collision surfaces |
| Q14 | **Multi-machine sync model** — if the user works on multiple machines, is the DB synced via pg_dump? Or do files sync via git and the DB rebuilds locally on each machine? | When the user first sets up synthesis-kb on a second machine |
| Q15 | **Book / publication integration** — `book-scaffold-astro` and `book-template-astro` exist. Should synthesis-kb support a "book chapter" record type, or a `published_as` link from claims/dossiers to book sections? Or is publication strictly out of scope? | When the first dossier is being assembled into a book chapter |
| Q16 | **MCP HTTP transport for synthesis-kb** — current scaffold uses stdio (local). HTTP becomes necessary if brandon-behring.dev needs runtime data (vs. build-time sync). | At brandon-behring.dev integration design time (after §12.1 issue is engaged) |
| Q17 | **Concept extraction prompt evolution** — if the extraction prompt changes (e.g., to better handle BRIDGE-type concepts), are all dossiers re-extracted? How is concept_extraction_version tracked per concept row? | When the first prompt change is proposed (M3 or later) |
| Q18 | **Tier-2/3 escalation for consistency audit and extractor** — when cost estimates (per §9.5) recommend Tier 2 (`runpod-deploy`) or Tier 3 (Claude), how is the escalation wired? Direct in synthesis-kb code, or via a hook? | Once §9.5 cost-estimate task completes |

These are not blocking. M0–M6 build can proceed against the resolved decisions in §9. Each item above gets surfaced as a focused decision when its trigger fires.
