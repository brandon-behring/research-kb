# Strategic Assessment: Research-KB Value Delivery

**Date**: 2026-02-27
**Context**: Post-Phase AF (33 phases completed, all 5 tiers done)

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

### 1.3 What was never built (updated 2026-03-23)

Features from original plans, Gemini audit, or ROADMAP "Future Work":

1. ~~**Path-Augmented Synthesis**~~ → ✅ DONE (Phase AC: `explain_connection` — graph path + evidence hydration + LLM synthesis)
2. **Learned Weight Optimization**: Script exists (`optimize_weights.py`) but blocked — 18s/query makes Nelder-Mead impractical. Needs refactor to precompute scores
3. **Multi-hop Reasoning Chain Explanations**: PARTIAL — `explain_connection` does single-path; no multi-hop chains
4. ~~**Semantic Chunking**~~ → ✅ DONE (Phase AH: heading-aware splitting; Phase AJ: Docling/Granite-258M LaTeX-preserving extraction)
5. ~~**Concept Deduplication at Scale**~~ → ✅ DONE (Phase AF: 2,370 singular/plural pairs merged, 312K→310K)
6. ~~**Automated Literature Review**~~ → ✅ DONE (Phase AI: `generate_literature_review()`, MCP tool #22, CLI `review generate`)
7. **Temporal Reasoning / Contradiction Detection**: Not started
8. **The other 10 Codex domains**: Not started — only Domain 3 built

### 1.4 Evidence the platform serves itself, not users

- interview_prep domain *had* 10-30% Hit@10 -- fixed in Phase AE (now 100% Hit@10, MRR 0.636)
- The only real consumer (research-agent) is blocked by brittle markdown parsing
- 92.9% Hit@K is tracked religiously. "Did this help me understand DML better?" appears nowhere

---

## 2. Prioritized Value Delivery

### Tier 1: Unblock the only real consumer (Phase Z)

**JSON output for MCP tools** -- research-agent parses markdown with ~70 lines of regex. Add `output_format` parameter to 7 MCP tools, return structured JSON.

### Tier 2: Build the synthesis layer

**`explain_connection(concept_a, concept_b)`**: Find shortest path, hydrate with text chunks, LLM-generate explanation. Expose as MCP tool and CLI command. **DONE (Phase AC).**

**Scoped assumption audit**: Add `scope` and `domain` parameters to `audit_assumptions`. **DONE (Phase AB).**

### Tier 3: Fix the weakest domain ✅ DONE (Phase AE)

interview_prep fixed: 7 synonym groups, 15 eval test cases, 100% Hit@10, MRR 0.636.

### Tier 4: Codex audit fixes ✅ DONE (Phase AD)

CI cadence labels, stale golden_dataset refs, MRR threshold correction, README tool count.

### Tier 5: Concept deduplication ✅ DONE (Phase AF)

310K concepts (from 312K). 2,370 singular/plural pairs merged, zero eval regression.

---

## 3. What NOT to do (updated 2026-03-23)

- No more coverage gate raises (70% is sufficient)
- No more doc alignment phases (audit_docs.py exists)
- No more test fortification phases (2,700+ tests is enough)
- No more mypy/black/ruff phases (all at zero baseline)
- No more ingestion until KG is restored (997 sources is sufficient corpus)
- No more eval test case writing without running eval first (validate patterns against real results)

---

## 4. The Test

> Can you sit down, ask "What assumptions does a DML estimator require for valid inference in a time-series setting?", get a synthesized answer with source citations and a graph-traced explanation chain, and learn something you didn't know?

Until that works, the platform is not serving its purpose.

---

## 5. Phase Log

| Phase | Date | Focus |
|-------|------|-------|
| Z | 2026-02-26 | JSON MCP output (Tier 1) |
| AB | 2026-02-26 | Scoped assumption audit (Tier 2) |
| AC | 2026-02-26 | explain_connection synthesis (Tier 2 crown jewel) |
| AD | 2026-02-27 | Codex audit cleanup (Tier 4) |
| AE | 2026-02-27 | Interview prep fix — 100% Hit@10, MRR 0.636 (Tier 3) |
| AF | 2026-02-27 | Concept deduplication — 312K→310K, zero eval regression (Tier 5) |
| AG | 2026-02-27 | Documentation trust alignment — 11 stale claims fixed |
| AH | 2026-03-01 | Semantic chunking — heading-aware PDF splitting |
| AI | 2026-03-01 | Literature review + operational scripts (MCP tool #22, weight opt, rechunk) |

| AJ | 2026-03-06 | Docling migration — LaTeX-preserving PDF extraction |
| Sprint 1 | 2026-03-09 | output_format on get_source + cross_domain_concepts, research-agent friction fixes |
| RAG Opt | 2026-03-21 | 100% embeddings (was 67%), 41K citation edges (was 5K), 3-way search defaults |
| Cleanup | 2026-03-21 | Removed 22 interview_prep code_repos, retagged ML Interviews to machine_learning |
| Catalog | 2026-03-22 | 552 books ingested (Tier 1+2), 12 new domains, 211 new sources |
| Citations | 2026-03-23 | Full citation rebuild: 41,852 edges, 997/997 sources with authority |

All 5 original tiers + RAG optimization + catalog ingestion complete. 1,198 sources, 1,012K chunks, 36 domains. Next unlock: KG re-extraction (~$250-300).

---

## 6. Execution Roadmap

Updated 2026-03-23. All $0 sprints complete. KG re-extraction remains the next major unlock.

### Current State (2026-03-23)

| Metric | Value |
|--------|-------|
| Sources | 997 (~771 textbooks, 226 papers) |
| Chunks | 857,024 (100% embedded, 0 null) |
| Citation edges | 41,852 (all 997 sources have authority) |
| Concepts | 310K (stale chunk IDs — KG disabled, 0/997 sources linked) |
| Eval | 107 test cases, 35 domains, MRR 0.771, Hit Rate 94.4% |
| Search mode | 3-way default (FTS + vector + citation). Graph OFF |

### Sprint History

| Sprint | Status | Cost | Description |
|--------|--------|------|-------------|
| 1. Friction fixes | ✅ Done | $0 | MCP output_format, research-agent venv/stats fixes |
| 2. RAG optimization | ✅ Done | $0 | 100% embeddings, 41K citation edges, 3-way defaults |
| 3. Interview prep cleanup | ✅ Done | $0 | Removed 22 derivative code_repo sources |
| 4. Catalog ingestion | ✅ Done | $0 | 552 books (Tier 1+2), 12 new domains, 107 eval cases |
| 5. Weight optimization | Blocked | $0 | Script runs 18s/query (40h total). Needs precompute refactor |
| 6. Live validation | Pending | ~$5 | North Star: research-agent DML query |

### Knowledge Graph: Deferred (~$250-300)

The single biggest remaining capability gap:
- 310K concepts, 744K relationships exist but reference stale chunk IDs
- 0/997 sources have working concept→chunk links
- Graph search, concept neighborhood, and graph-traced synthesis all degraded
- Re-extraction: Haiku 4.5 on 857K chunks (~$250-300)
- Revisit when budget allows

### Prioritized Roadmap

| Priority | Item | Cost | Dependency |
|----------|------|------|------------|
| 1 | Refactor `optimize_weights.py` (precompute scores) | $0 | None |
| 2 | Live validation: North Star DML query | ~$5 | None |
| 3 | KG re-extraction (Haiku 4.5, 857K chunks) | ~$250-300 | Depends on #2 results |
| 4 | Research-agent: wire `literature_review` tool | $0 | None |
| 5 | Interactive citation network (Streamlit/D3.js) | $0 | Low priority |
| 6 | Multi-hop reasoning chains | $0 | Blocked by #3 |
| 7 | Temporal reasoning / contradiction detection | $0 | Low priority |
