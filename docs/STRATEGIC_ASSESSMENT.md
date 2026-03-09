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

### 1.3 What was never built

Features from original plans, Gemini audit, or ROADMAP "Future Work":

1. **Path-Augmented Synthesis**: `graph_path` returns node IDs but never fetches chunk text or produces natural-language synthesis
2. **Learned Weight Optimization**: Search weights are hand-tuned magic numbers
3. **Multi-hop Reasoning Chain Explanations**: Designed in Phase 3 notes, not implemented
4. **Semantic Chunking**: Still using 300-token fixed windows
5. **Concept Deduplication at Scale**: `_normalize_concept` handles casing only; 312K concepts with significant duplication
6. **Automated Literature Review**: ROADMAP future work, never started
7. **Temporal Reasoning / Contradiction Detection**: Never started
8. **The other 10 Codex domains**: Only Domain 3 built

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

## 3. What NOT to do

- No more coverage gate raises (70% is sufficient)
- No more doc alignment phases (audit_docs.py exists)
- No more test fortification phases (2,630 tests is enough)
- No more mypy/black/ruff phases (all at zero baseline)
- No new domain acquisition until existing domains deliver value

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

All 5 tiers complete. Semantic chunking done, literature review done, weight optimization scripts ready. Full corpus rechunk in progress (Docling).

---

## 6. Execution Roadmap

Decided 2026-03-02 via `/iterate`. Budget: $50-100.

### Sprint Sequence

| Sprint | Status | Cost | Description |
|--------|--------|------|-------------|
| 1. Friction fixes | ✅ Done | $0 | MCP output_format, research-agent venv/stats fixes |
| 2. Weight optimization | Blocked | $0 | Requires post-rechunk concept re-extraction |
| 3. Live validation | Pending | ~$5 | North Star: research-agent DML query + KB reviews |
| 4. time_series pilot | Superseded | ~$15-30 | Full corpus rechunk running instead |

### Budget Estimate

| Activity | Cost |
|----------|------|
| Cross-repo fixes (Sprint 1) | $0 |
| Weight optimization (Sprint 2) | $0 |
| Live validation (~100 queries) | ~$5.50 |
| Full corpus rechunk (Docling) | $0 (GPU) |
| Concept re-extraction (Haiku) | ~$250-300 (full) or $0 (if FTS+vector MRR ≥ 0.7) |
| **Total** | **$5-305** (decision gate at post-rechunk eval) |

### Post-Rechunk Pipeline (Critical Path)

1. Backfill embeddings (~4-6h, GPU)
2. Regenerate golden candidates
3. Eval retrieval (FTS+vector only) — **decision gate**
4. If MRR < 0.7: concept re-extraction (Haiku 4.5)
5. KuzuDB sync
6. Sprint 2 weight optimization
7. Sprint 3 live validation

### Backlog

- Integrate `literature_review` into research-agent ($0, medium priority)
- Multi-hop reasoning chains (low priority)
- Embed 26.5K missing concept embeddings (low)
- Temporal reasoning / contradiction detection (low)
- Remaining 10 codex domains (low)
