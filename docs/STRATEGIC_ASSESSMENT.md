# Strategic Assessment: Research-KB Value Delivery

**Date**: 2026-02-27 (original) | **Last updated**: 2026-04-03
**Context**: Post-bulk-ingestion (1,756 sources across 36 domains)

---

## 1. The Diagnosis

### 1.1 Original purpose

Per lever_of_archimedes design docs, research-kb is **Domain 3 of an 11-domain knowledge codex** whose purpose is:

> "Transform agent workflows from 'implement from memory' to 'retrieve canonical algorithms, verify assumptions, and audit against established methods.'"

The system was designed to be the **hippocampus** (long-term memory) of an AI-augmented research workflow.

### 1.2 Where the drift happened

26 phases (A through Y) completed. Categorized:

| Category | Phases | Count |
|----------|--------|-------|
| Infrastructure/quality | G, I, M, P, Q, R, S, V, W, X, Y | 11 |
| Data acquisition/tagging | H, N, T, U | 4 |
| Core capability | 1-4, D, E | 6 |
| Integration | F | 1 |
| Eval/docs | J, K, O | 3 |

After Phase F (January 2026), **zero phases advanced core capability**. The last 20 phases are infrastructure hardening, coverage gates, doc alignment, domain tagging, and test fortification.

### 1.3 What was never built (updated 2026-03-30)

Features from original plans, Gemini audit, or ROADMAP "Future Work":

1. ~~**Path-Augmented Synthesis**~~ -> Done (Phase AC: `explain_connection`)
2. **Learned Weight Optimization**: Script exists (`optimize_weights.py`), refactored Sprint 5 (disk cache + parallel precompute, ~7 min first run, ~2s cached). Not yet run at scale on full eval set
3. **Multi-hop Reasoning Chain Explanations**: PARTIAL -- `explain_connection` does single-path; no multi-hop chains. Blocked by KG disconnection
4. ~~**Semantic Chunking**~~ -> Done (Phase AH + AJ: Docling/Granite-258M)
5. ~~**Concept Deduplication at Scale**~~ -> Done (Phase AF: 312K->310K)
6. ~~**Automated Literature Review**~~ -> Done (Phase AI: MCP tool #22)
7. **Temporal Reasoning / Contradiction Detection**: Not started
8. **The other 10 Codex domains**: Not started -- only Domain 3 built

---

## 2. Prioritized Value Delivery (All 5 Tiers Complete)

### Tier 1: JSON output for MCP tools (Phase Z) -- Done
### Tier 2: Synthesis layer (Phases AB, AC) -- Done
### Tier 3: Interview prep fix (Phase AE) -- Done
### Tier 4: Codex audit fixes (Phase AD) -- Done
### Tier 5: Concept deduplication (Phase AF) -- Done

---

## 3. What NOT to do (updated 2026-03-30)

- No more coverage gate raises (70% is sufficient)
- No more doc alignment phases (audit_docs.py + generate_status.py handle ongoing drift)
- No more test fortification phases (2,815+ tests is enough)
- No more mypy/black/ruff phases (enforced via pre-commit hooks)
- No more eval test case writing without running eval first
- No more domain tagging sprints (36 domains is broad cross-disciplinary coverage)
- Defer KG re-extraction until Anthropic credits replenished (see Section 7)

---

## 4. The Test

> Can you ask a cross-disciplinary question -- spanning two or more domains -- and get a synthesized answer with source citations from each domain?

**Status (validated 2026-03-25)**: 3-way search (FTS + vector + citation) returns multi-domain results for explicit cross-domain topics. KG-dependent synthesis tools (`explain_connection`, `literature_review`, `audit_assumptions`) produce degraded results because `chunk_concepts = 0`.

---

## 5. Phase Log

| Phase | Date | Focus |
|-------|------|-------|
| 1-4 | 2025-11 to 2025-12 | Foundation, KG, Enhanced Retrieval, API/Dashboard |
| D | 2025-12 | KuzuDB, Prometheus, benchmarking |
| E | 2025-12 | RAG/LLM concept extraction (Haiku 4.5, ~$30) |
| F | 2026-01 | Cross-repo integration |
| G-K | 2026-01 to 2026-02 | Hygiene, multi-domain, CI, eval, docs |
| M-O | 2026-02 | Quality gates, domain gaps, eval hardening |
| P-S | 2026-02 | Infrastructure hardening, coverage |
| T-U | 2026-02 | Domain acquisition, concept extraction |
| V-Y | 2026-02 | Doc trust, CLI, data accuracy, test fort, mypy zero |
| Z | 2026-02-26 | JSON MCP output (Tier 1) |
| AB | 2026-02-26 | Scoped assumption audit (Tier 2) |
| AC | 2026-02-26 | explain_connection synthesis (Tier 2 crown jewel) |
| AD | 2026-02-27 | Codex audit cleanup (Tier 4) |
| AE | 2026-02-27 | Interview prep fix -- 100% Hit@10 (Tier 3) |
| AF | 2026-02-27 | Concept deduplication -- 312K->310K (Tier 5) |
| AG | 2026-02-27 | Documentation trust alignment |
| AH | 2026-03-01 | Semantic chunking -- heading-aware PDF splitting |
| AI | 2026-03-01 | Literature review + operational scripts (MCP tool #22) |
| AJ | 2026-03-06 | Docling migration -- LaTeX-preserving PDF extraction |
| Sprint 1 | 2026-03-09 | output_format on get_source + cross_domain_concepts |
| RAG Opt | 2026-03-21 | Embedding backfill (67%->100%), 41K citation edges, 3-way defaults |
| Cleanup | 2026-03-21 | Removed 22 interview_prep code_repos |
| Catalog | 2026-03-22 | 552 books ingested (Tier 1+2), 12 new domains |
| Citations | 2026-03-23 | Full citation rebuild, PageRank authority |
| Batch 1 | 2026-03-24 | Gap-fill ingestion + 66 arXiv papers |
| Batch 2 | 2026-03-25 | Cleanup + 25 remaining books |
| Audit | 2026-03-25 | Unified graph defaults OFF, reference-only metrics |
| Optimizer | 2026-03-25 | Weight optimizer refactor (32min -> 2s cached) |
| Validation | 2026-03-25 | Cross-disciplinary validation (3-way pass, KG broken) |
| Bulk Ingest | 2026-03-29 | 408 new sources, 1.4M chunks, 100% embedded |
| Full Audit | 2026-03-30 | Dual-audit remediation, ROADMAP consolidated, test contract fix |
| Eval P1 | 2026-04-01 | Fix v1 limit bug, reranking confirmed harmful (-44% true MRR) |
| Eval P2-3 | 2026-04-03 | 3 eval scripts, 71-query set, TREC pooling (3138 candidates), pre-labeling |

---

## 6. Current State

**Do not hardcode metrics here.** See `docs/status/CURRENT_STATUS.md` (auto-generated) for live numbers.
See `docs/DOMAIN_COVERAGE.md` (auto-generated) for per-domain breakdown.

Qualitative status:
- **Embeddings**: Complete (100%, 0 NULL as of 2026-03-29)
- **KG**: Fully disconnected (chunk_concepts = 0). See Section 7
- **Citations**: Active, PageRank-scored (49K+ edges)
- **Search**: 3-way default (FTS + vector + citation). Graph OFF (all surfaces unified 2026-03-25)
- **Ingestion**: 1,756 sources ingested, but only 785/3,697 catalog books (21%) completed. 2,284 actionable books remain. See `.mass_ingest_checkpoint.json`
- **CI**: PR checks automated (black, ruff, mypy, unit+integration tests, doc freshness). Integration and weekly rebuild workflows are **manual** (`workflow_dispatch`), not scheduled

### Sprint History

| Sprint | Status | Cost | Description |
|--------|--------|------|-------------|
| 1. Friction fixes | Done | $0 | MCP output_format, research-agent venv/stats fixes |
| 2. RAG optimization | Done | $0 | 100% embeddings, 41K citation edges, 3-way defaults |
| 3. Interview prep cleanup | Done | $0 | Removed 22 derivative code_repo sources |
| 4. Catalog ingestion | Done | $0 | 552 books (Tier 1+2), 12 new domains, 107 eval cases |
| 5. Weight optimization | Done | $0 | Disk cache + parallel precompute (~2s cached) |
| 6. Bulk ingestion | Done | $0 | 408 new sources, 1.4M chunks, 100% embedded |
| 7. Full audit | Done | $0 | Dual-audit remediation, doc consolidation, test contract fix |
| 8. Eval overhaul P1-3 | Done | $0 | v2 eval infrastructure: 71 queries, TREC pooling, pre-labeling |

---

## 7. Knowledge Graph: Deferred Decision

> **⛔ CLOSED / SUPERSEDED (RS4, 2026-06-13).** The chunk-level KG re-extraction
> backlog described below is **closed, not deferred**: research-kb's concept-graph
> ambition was retired (decision R3, `docs/decisions/0001-scope-retrieval-citation-backbone.md`).
> KuzuDB + the graph-query/score machinery were removed; the stale concept/graph MCP tools are
> fail-loud retirement stubs; `research_kb_search` is FTS + vector + citation (3-way). The concept
> layer is now **synthesis-kb** (claim-level, eval-gated). The $150–300 re-extraction line item is
> dropped. Falsifier (reopen): a concrete chunk-level-graph use case that claim-level synthesis-kb
> cannot serve. The section below is historical.

**Status**: ~~Deferred until Anthropic credits are replenished~~ **CLOSED (retired, RS4)** — see banner above.

### Current State

- **310,063** concept nodes and **743,984** relationship edges exist in KuzuDB
- **chunk_concepts = 0** -- the junction table linking chunks to concepts is empty
- Concepts exist for only **3/36 domains**: causal_inference (271K), rag_llm (21K), time_series (17K)
- All other 33 domains have zero concept extraction

### Why Deferred

1. **Anthropic API credits exhausted** (key ...cQAA has insufficient credits). LLM extraction requires working API access
2. **Corpus grew 3.5x** since last extraction (495 -> 1,756 sources). Previous concepts reference old chunk IDs
3. **Ingestion was the priority** -- cross-disciplinary coverage takes precedence over graph depth

### Cost Estimates (Haiku 4.5)

| Scope | Sources | Est. Chunks | Est. Cost | Timeline |
|-------|---------|-------------|-----------|----------|
| Top 5 domains (causal_inference, machine_learning, deep_learning, rag_llm, time_series) | 820 | ~500K | $50-80 | 4-6 hours |
| Full corpus (all 36 domains) | 1,756 | ~1.4M | $150-300 | 8-12 hours |

### Trigger Condition

Re-evaluate when ALL of:
1. Anthropic API credits are replenished
2. Ingestion volume is stable (no major batch ingestion planned)
3. A concrete use case demands graph-backed synthesis (e.g., research-agent needs `explain_connection` with text grounding)

### What It Unlocks

- `--graph` flag produces real results (4-way hybrid search)
- `explain_connection` grounds concept paths in actual text evidence
- `literature_review` graph phase returns real concept neighborhoods
- `audit_assumptions` graph path returns domain-filtered assumptions
- Multi-hop reasoning chains become feasible

---

## 8. Prioritized Roadmap

| Priority | Item | Cost | Dependency | Status |
|----------|------|------|------------|--------|
| ~~1~~ | ~~Weight optimizer refactor~~ | ~~$0~~ | | Done (Sprint 5) |
| ~~2~~ | ~~Embedding backfill~~ | ~~$0~~ | | Done (Sprint 6) |
| 1 | **Eval v2: annotate + baseline** | $0 | Phases 1-3 done | In progress — Phase 4-5 next |
| 2 | **Eval v2: citation ablation + CI** | $0 | Priority 1 | Phases 6-8 |
| 3 | KG re-extraction (phased, top 5 domains first) | $50-80 | Anthropic credits | Deferred (Section 7) |
| 4 | Wire `literature_review` to research-agent | $0 | None | Not started |
| 5 | Full-corpus KG re-extraction | $150-300 | Priority 3 validated | Deferred |
| 6 | Multi-hop reasoning chains | $0 | Priority 3+5 | Blocked |
| 7 | Interactive citation network (D3.js) | $0 | None | Low priority |
| 8 | Temporal reasoning / contradiction detection | $0 | Priority 5 | Not started |
