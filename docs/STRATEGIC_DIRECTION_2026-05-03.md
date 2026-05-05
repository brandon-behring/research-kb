# Strategic Direction Memo — 2026-05-03

**Status**: Tier 1 closure session in progress (sessions #23–#25). Stage B residual closed
(deferred via priority marker rather than retried), embedding backfill running, Tier 1
audit + #9 closure pending.

**Companion to**: [`STRATEGIC_ASSESSMENT.md`](STRATEGIC_ASSESSMENT.md) (canonical roadmap).
This memo captures the diagnostic + strategic insights surfaced today, particularly
around **corpus quality** (redundancy, pollution) and the **un-ingested backlog**.

---

## 1. Corpus State (verified 2026-05-03)

| metric | value |
|---|---:|
| sources in DB | 2,195 |
| chunks total | 1,684,506 |
| chunks NULL embedding | 318,106 (19% — backfill running) |
| sources marked low-priority | 203 (155 `low_redundant` + 48 `low_review_pending`) |
| chunks under low-priority sources | 165,091 (10%) |
| formula gap chunks remaining | 174,922 (Tier 1 alone was 111k pre-redo) |
| domains active | 36 |
| PDFs on disk (excluding split parts) | 7,644 |
| PDFs un-ingested (no DB source) | 6,037 (37.9 GB) |

### 1.1 Domain distribution skew

```
top 5 by chunks       : mathematics 296k, software_eng 152k, analysis 138k,
                        topology_geometry 109k, machine_learning 88k
bottom 6 (<10 sources): adtech 2, recommender 3, economics 3, forecasting 5,
                        sql 8, healthcare 35-by-source-but-1k-chunks
```

### 1.2 Where the work was today

- **Stage B closure**: 287 ok / 13 OOM (original Apr 26-27) + 3 ok / 8 OOM / 5 skipped (today's recovery on items 301-316). The 16 never-ran are now checkpoint-final; the 21 OOM cohort is residual for split-and-reingest.
- **All 16 today-failed sources were redundant calculus textbooks** (Salas, Hass-Pearson 2017, Hass/Heil/Thomas Pearson 2018) — exactly the kind of corpus that should be deprioritized rather than re-extracted.

---

## 2. Diagnostic — Why the Tier 1 redo got expensive

1.1 **Stage A/B campaign succeeded only when desktop was idle** (laptop-closed window per memory). Today's run with VS Code holding ~1-2 GB GPU produced 64% OOM rate vs <5% historical. **VRAM contention, not source difficulty, was the primary failure mode.**

1.11 Implication: *future MinerU campaigns must be scheduled when the desktop is idle (overnight, headless) — not as a foreground task during work hours.*

2.1 **A nontrivial chunk of the Tier 1 work was on books we shouldn't have ingested.** 79 sources marked `low_redundant` today (Schaum's, ISMs, freshman calc, MEAP older versions, duplicate editions); these contributed 25,571 NULL chunks (8% of backfill scope) and ~50k existing embeddings. Re-extracting and embedding redundant intro material **inflates corpus cost without lifting research signal**.

2.11 The Tier 1 redo audit (issue #9) didn't apply a corpus-quality filter before scoping. Tier 1 was defined by domain (`mathematics ∪ linear_algebra ∪ optimization ∪ probability_theory ∪ analysis ∪ topology_geometry`); within those domains, generic intro books were on equal footing with graduate canonical texts.

3.1 **The `ingestion_priority` marker (added today) addresses a structural gap**: the pipeline previously had no "should we ingest this?" filter beyond domain assignment. The marker is auditable (manifest in `data/exclusion_lists/`), reversible (single SQL UPDATE), and now wired into search (`search.py:_apply_priority_multiplier`).

---

## 3. Retrieval Pollution — How bad is it?

3.1 **Pollution is real but domain-scoped.** The 79 `low_redundant` markers concentrate in `mathematics` (41) and `analysis` (38). Querying `physics`, `machine_learning`, `software_engineering`, `causal_inference`, etc. is unaffected — those domains have no marked sources yet.

3.2 **For math/analysis queries, the impact pre-fix was material**:
- Vector retrieval surfaces calculus chunks for Banach/Hilbert/measure queries because of shared elementary vocabulary.
- Citation-authority signal *backfires* on textbooks (heavily-cited as intro material → high PageRank).
- The reranker (BGE-reranker-v2-m3) only operates on top-K candidates, so cannot recover specialized monographs drowned in initial retrieval.

3.3 **Today's mitigation**: `_apply_priority_multiplier` in search.py applies 0.5× to `low_redundant`, 0.75× to `low_review_pending` after score fusion. No-op for normal-priority sources. Wired into both `search_hybrid` and `search_hybrid_v2`. Smoke-tested; 83 storage tests pass.

3.4 **Limits**: This is heuristic, not eval-validated. We don't yet know whether 0.5/0.75 are the right multipliers, or whether RRF fusion is more robust than weighted-sum to the downweight. Eval-driven tuning is a follow-up.

---

## 4. Un-Ingested Backlog (6,037 PDFs, 37.9 GB)

| location | un-ingested | size |
|---|---:|---:|
| `fixtures/library_books/*` | 4,669 | 33 GB |
| `fixtures/library_books/migrated/` | 88 | 1.2 GB |
| `fixtures/textbooks/` | 93 | 1.2 GB |
| `fixtures/library_books/needs_ocr/` | 16 | 329 MB |
| `fixtures/papers/` | 152 | 293 MB |
| (other 18 domain subdirs) | ~970 | ~1.5 GB |

### 4.1 What's in the backlog

Spot-check of top-20 largest un-ingested PDFs reveals a 60/40 mix:

- **High-value specialized**: Cohen-Tannoudji *Quantum Mechanics* (117 MB), Schroeder *Thermal Physics* (116 MB), Aliprantis *Real Analysis* (90 MB), Murphy *PML Advanced Topics 2025* (144 MB), Hubbard *Vector Calculus* (101 MB, was OCR-pending), GARP *FRM Part 1* (154 MB), neuroscience canonical (Principles of Neural Science, 258 MB).

- **Low-value redundant or off-topic**: Stewart *Calculus Concepts* (235 MB), Solutions_Manual variants (101 MB), Mueller report (139 MB — political document, not research), Campbell Biology textbook (282 MB — undergrad bio), Rippetoe *Starting Strength* (230 MB — fitness, not research).

### 4.2 Implication

A naive "ingest everything" pass would double the corpus (+38 GB → ~62 GB) AND amplify the pollution problem we just bandaged. Instead:

1. Apply the `ingestion_priority` filter at *acquisition time*, not just retrospectively.
2. Pre-classify the 6,037 by likely value — graduate monograph / canonical textbook / intro / solution-manual / off-topic — using filename heuristics + small-LLM classification.
3. Ingest only `priority ∈ {high, normal}` (specialized monographs + new editions of canonical texts).
4. The `low_*` material can be flagged as "owned but not indexed" — discoverable via filesystem, not via search.

---

## 5. Strategic Options for Next 2-4 Weeks

### Option A — Finish Tier 1, defer everything else
- This session: backfill (running) + audit + #9 update.
- Tier 2 + Tier 3 redo deferred indefinitely.
- Un-ingested backlog deferred.
- **Pro**: Low scope creep. Closes the longest-running campaign.
- **Con**: Tier 2 + Tier 3 still have ~125k formula-gap chunks. Backlog stays untouched. North Star (cross-domain synthesis) is unblocked but uneven.

### Option B — Marker-first, then selective ingest (recommended)
- This session: backfill + audit + #9 update + first 200-300 priority markers (done) + search wire-in (done).
- Next session: extend markers to remaining ~70 trouble parents + ~100 likely-redundant from un-marked corpus. Target: 350-450 marked sources total.
- Session +2: triage the 6,037 un-ingested PDFs into 4 buckets (high/normal/low/excluded) using filename heuristics + spot-check. Output: `data/exclusion_lists/acquisition_triage_2026-05-N.json`.
- Session +3: ingest the high-priority bucket (~200-400 specialized monographs from the un-ingested pile). Use `--auto-stop-services` + `--mineru-vram-min-mib 4000` overnight.
- Session +4: Tier 2 redo with the marker filter applied (skip already-marked; focus on physics/numerical/algebra gaps).
- **Pro**: Each session has a single-feature scope. Corpus quality improves *before* it grows. Cross-domain synthesis benefits early.
- **Con**: Slower path to "all 174k formula gaps closed". Requires discipline to not ingest everything.

### Option C — Eval-driven (reset to validation)
- Build a small retrieval eval suite (10 queries × 36 domains × 3 difficulty tiers) before any new ingestion.
- Score current corpus on the suite (baseline).
- Score after each marker-batch, after search.py changes, after each Tier 2 sub-batch.
- Use eval delta to decide: is this marker pattern net-positive? Is this domain underserved?
- **Pro**: Highest-leverage long-term — turns "I think pollution is bad" into "we know X% of math queries hit junk pre-fix, Y% post-fix". Aligns with `feedback_evaluation_trust`.
- **Con**: 1-2 sessions of pure infrastructure before any visible corpus progress. Requires eval ground-truth.

### Option D — KG revival (orthogonal)
- Re-run concept extraction on the new MinerU chunks. Cost: ~$250 + compute (per memory).
- Re-enable `use_graph=True` defaults across all 7 surfaces (currently OFF; per `project_graph_defaults_unified.md`).
- This is orthogonal to A/B/C — can run in parallel.
- **Trigger criteria** (per Strategic Assessment §7): ≥80% of post-redo Tier 1 chunks have non-zero concept count after re-extraction trial.

---

## 6. Recommendation: B + C in parallel

**Concretely**:
1. **This session**: Finish backfill → audit → close #9 with completion summary + Tier 3 hand-off.
2. **Next session (~2 days)**: Build a minimal eval suite (10 × 6 = 60 queries; ground-truth from the user). Run baseline on current corpus. Establishes the metric.
3. **Session +2**: Extend markers (target 400+); re-run eval. Decision point: did markers + downweight improve precision? If yes, scale; if no, retune multipliers (0.3 / 0.6) or switch to RRF-only (rank-based, less sensitive to magnitude).
4. **Session +3**: Triage un-ingested backlog. Filename + spot-check classification. Output JSON manifest + acquisition policy.
5. **Session +4**: Ingest top 200-400 specialized monographs from the high-priority bucket. Re-run eval.
6. **Session +5**: Decide on Tier 2 / Tier 3 redo scope based on eval results + domain-coverage gaps.

KG revival (Option D) can run as a separate weekend campaign once Tier 1 redo and high-priority new ingest are done — it benefits most from a high-quality corpus, so do it after corpus hygiene is locked in.

---

## 7. Anti-patterns to avoid

7.1 **Don't re-extract redundant material.** The 16 calculus parts I tried to recover today were a wasted ~90 minutes. Lesson: priority-mark *before* scoping a redo campaign, not retrospectively.

7.2 **Don't ingest during VS Code session.** Today's 64% Stage B failure rate vs the original Apr 26-27 run's 4% confirms the "laptop-closed window" pattern from memory. *Schedule MinerU campaigns via `cron` or `/loop` for off-hours.*

7.3 **Don't conflate corpus completeness with corpus quality.** 6,037 un-ingested PDFs ≠ 6,037 valuable additions. Filtering at acquisition time is cheaper than filtering at retrieval time.

7.4 **Don't rely on memory recall for live state.** Multiple times this session I checked memory ("X is at Y status") and verified against DB ("actually it's Z now"). Live `psql` / Python query before acting on a memory-derived assumption.

---

## 8. Open questions / decisions for the user

8.1 **Multiplier values**: 0.5 / 0.75 are placeholders. Once eval suite exists, tune empirically. Possible alternatives: 0.3 / 0.6 (more aggressive) or 0.7 / 0.85 (more conservative).

> **Update 2026-05-05** (eval baseline session, see `evaluation_runs/baseline_2026-05-05.md`): A/B run on the 71-query v2 suite produced **null effect** at p>0.05 (NDCG@10 0.5572 ON vs 0.5673 OFF; Fisher's randomization test). On the targeted domains (math, analysis), the marker effect is exactly **zero** — top-K retrievals do not include any marked sources for those queries. Where markers had any effect at all, it was a regression: dynamical_systems -0.150 NDCG@10, biology_neuroscience -0.127, reinforcement_learning -0.062. The label-quality audit (`fixtures/eval/v2_pool_audit_2026-05-05.yaml`) found grade-0 hit-rate of only 37% — auto-grades cannot reliably distinguish marker effect from label noise. **Do not retune multipliers based on this null result; instead: (a) re-grade the eval pool, (b) investigate the dyn_sys/RL/bio regressions for mistagged sources, (c) build marker-stress-test queries before re-running A/B.**

8.2 **Do `low_review_pending` items deserve the marker at all?** Some (Lang's *Short Calculus*, Klambauer's *Aspects of Calculus*) are intro-but-pedagogically-different. User judgment; eval suite would settle it.

8.3 **MEAP older versions kept as `low_redundant`** — but newest version of a Manning book may be DRAFT (typos, incomplete chapters). Should newest-MEAP get downweighted vs the published edition (when both exist)?

8.4 **Mueller report and similar non-research PDFs**: should they be deleted from fixtures or just left un-ingested? Currently they sit in fixtures consuming disk but not corpus.

8.5 **Tier 2/Tier 3 strategy**: now that Tier 1 closure has revealed how much of Tier 1 was redundant, is a lighter Tier 2/3 redo (priority-filtered, ~50% the books) the right move? Or full coverage like the original plan?

---

## 9. Artifacts produced today

- `data/exclusion_lists/low_priority_redundant_2026-05-03.json` — manifest of 203 markers (155 + 48), with hits + reasons.
- `packages/storage/src/research_kb_storage/search.py` — `_apply_priority_multiplier` helper + 2 call sites.
- `backups/research_kb_20260502_101022.sql` — 23 GB pre-session backup.
- `data/checkpoints/tier1_redo_stage_b.jsonl` — 316 lines (300 historical + 16 new today, all final state).
- `data/logs/tier1_redo_stage_b_recover-20260503-063142.log` + `data/logs/backfill_embeddings-20260503-081923.log` — run logs.
- `docs/STRATEGIC_DIRECTION_2026-05-03.md` — this memo.

---

*Companion docs to update after backfill completes:*
- `docs/STRATEGIC_ASSESSMENT.md` (single source of truth — incorporate §6 recommendation).
- `docs/status/CURRENT_STATUS.md` (auto-generated; will reflect new NULL count).
- Issue #9 (closure summary).
- Memory: `project_phase_c_stage_ab_status.md` (Tier 1 redo COMPLETE entry).
