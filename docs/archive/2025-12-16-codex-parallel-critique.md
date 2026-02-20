# Critique: `parallel-waddling-hanrahan.md` (codex)

This document critiques the current llama.cpp integration plan at `$HOME/.claude/plans/parallel-waddling-hanrahan.md` and evaluates it against Research-KB’s goals (hybrid retrieval + knowledge graph extraction) and the current extraction architecture.

## 1) Repo goals and constraints (what “success” means)

Research-KB’s core value proposition is *reliable* extraction of concepts/relationships to support graph-boosted retrieval (docs/SYSTEM_DESIGN.md#L8; README.md#L1; packages/storage/migrations/002_knowledge_graph.sql#L6).

That implies the extraction backend must optimize for:
- **Schema fidelity**: concept/relationship types must stay within a small ontology (packages/extraction/src/research_kb_extraction/models.py#L13; packages/extraction/src/research_kb_extraction/prompts.py#L31; packages/storage/migrations/002_knowledge_graph.sql#L17).
- **Operational safety**: backups are treated as a default guardrail (scripts/extract_concepts.py#L216; CLAUDE.md#L192).
- **Repeatable throughput**: long runs need stable performance and observability (scripts/extract_concepts.py#L79; scripts/extract_concepts.py#L590).

## 2) Plan summary (what it proposes)

The plan proposes adding a new `LlamaCppClient` using `llama-cpp-python`, with the stated intent to “replace Ollama” for ~1.8× speed on RTX 2070 SUPER (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L4`). It also gives **inconsistent timeline estimates**: “~9 days → ~2 days” (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L6`) and later “~17 days → ~2 days” (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L277`).

It also claims:
- Lower overhead by avoiding HTTP (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L9`).
- “Grammar-based JSON output” and “schema enforced” output (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L12`; `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L279`).

## 3) What’s good about the plan (aligned with repo architecture)

### 3.1 It matches the repo’s abstraction direction
The extraction package explicitly anticipates a future `LlamaCppClient` backend (packages/extraction/src/research_kb_extraction/base_client.py#L13), so adding it is consistent with the existing design.

### 3.2 It chooses a model format consistent with current Ollama usage
The plan uses a GGUF “Q4_K_M” Llama 3.1 8B instruct file (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L34`), which matches the current Ollama model’s quantization (runtime `curl http://localhost:11434/api/show …` reports `quantization_level Q4_K_M`).

### 3.3 It keeps structured validation in the pipeline
Using `ChunkExtraction.model_validate(...)` (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L176`) is compatible with existing backends (packages/extraction/src/research_kb_extraction/ollama_client.py#L156; packages/extraction/src/research_kb_extraction/anthropic_client.py#L208) and aligns with DB constraints (packages/storage/migrations/002_knowledge_graph.sql#L17).

## 4) Critical gaps / skeptical questions (things that will break or mislead)

### 4.1 The plan is missing at least one *required* repo change: CLI backend support
The plan shows running:
```bash
python3 scripts/extract_concepts.py --backend llamacpp ...
```
(plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L242`)

But the script currently restricts `--backend` to `["ollama", "anthropic"]` (scripts/extract_concepts.py#L657). Without updating this argparse choice (and help text), the plan’s Phase 3 commands will fail immediately.

**Constructive fix:** treat “add `llamacpp` to CLI choices” as Phase 2, not Phase 3 (scripts/extract_concepts.py#L657).

### 4.2 “Replace Ollama” is not the safest framing
System design and docs assume Ollama as the local default (“extraction via Ollama LLM”) (docs/SYSTEM_DESIGN.md#L75; CLAUDE.md#L131), and the code currently ships a tested Ollama client + mocks (packages/extraction/tests/test_ollama_client.py#L1).

**Recommendation:** implement llama.cpp as an *additional backend* (like Anthropic), not a replacement. Keep Ollama as a stable fallback (packages/extraction/src/research_kb_extraction/__init__.py#L29).

### 4.3 The “schema-enforced JSON” claim is not demonstrated (and may be wrong in practice)
The plan’s code uses:
```python
response_format={"type": "json_object", "schema": EXTRACTION_SCHEMA}
```
(plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L188`)

Two skeptical questions you should answer before betting the pipeline on this:
1) **Does your target `llama-cpp-python` version actually support this exact `response_format` shape?** (API compatibility risk; plan assumes OpenAI-style schema support.)
2) **Even if supported, is it “hard validation” or “model guidance”?** In this repo, both Ollama and Anthropic ultimately validate with Pydantic and silently return empty extraction on failure (packages/extraction/src/research_kb_extraction/ollama_client.py#L187; packages/extraction/src/research_kb_extraction/anthropic_client.py#L238). If llama.cpp produces malformed or truncated JSON, you’ll lose recall unless you add retries/repair.

**Constructive fix options:**
- **A (strict):** use llama.cpp grammar/GBNF (true constrained decoding) and keep Pydantic validation as a second line of defense.
- **B (pragmatic):** keep “JSON object” mode if available, but add *retry-on-validation-failure* and log the failure rate as a first-class metric (see §6.4).

### 4.4 The performance math is optimistic and not apples-to-apples
The plan claims ~1.8× faster (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L4`) and ~17 days → ~2 days (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L277`).

Key skepticism:
- **HTTP overhead is rarely the dominant cost** in GPU inference; prompt evaluation dominates for long prompts. The extraction prompt is large and includes strict type constraints (packages/extraction/src/research_kb_extraction/prompts.py#L20).
- The plan’s llama.cpp defaults include `n_ctx=2048` (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L111`), while the current Ollama client defaults to `num_ctx=4096` (packages/extraction/src/research_kb_extraction/ollama_client.py#L38). That alone can produce a sizeable speedup without switching backends.

**Constructive fix:** benchmark llama.cpp against a *properly tuned Ollama baseline* (same model quant, same context length, same concurrency) before concluding 1.8× is real.

### 4.5 Concurrency and thread-safety assumptions are underspecified
The plan recommends `--concurrency 1` because llama-cpp-python “handles batching internally” (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L261`).

Two concerns:
- In this repo, `--concurrency` means “number of concurrent LLM requests” (scripts/extract_concepts.py#L672). llama.cpp can be fast per request but still be single-request-at-a-time unless you add true multi-sequence batching or multiple processes.
- If the pipeline accidentally runs llama.cpp with concurrency > 1, you may hit model thread-safety issues (the plan code does not include an internal lock).

**Constructive fix:** enforce `concurrency=1` for llamacpp in code, or add a per-client `asyncio.Lock` so the model is never called concurrently.

### 4.6 Chat template / instruct formatting is a hidden quality footgun
Ollama will apply a model-appropriate chat/instruct template. With llama-cpp-python + GGUF, you may need to set `chat_format` or otherwise ensure the correct prompt template is used.

If template handling differs, you can get:
- worse instruction following (schema drift),
- higher hallucination rate,
- more JSON formatting errors,
even with the “same model + same quant”.

**Constructive fix:** add an explicit chat format / template configuration knob and A/B on a fixed chunk set. Treat “output schema fidelity rate” as a gating metric.

### 4.7 Rollback plan is incomplete / potentially misleading
Rollback suggests:
```bash
python3 scripts/extract_concepts.py --backend ollama --concurrency 3
```
(plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L284`)

But:
- Ollama parallelism also depends on server-side settings; the repo’s systemd service currently has no `OLLAMA_*` overrides (/etc/systemd/system/ollama.service#L1).
- Your GPU VRAM budget may not support 3 concurrent requests with 8GB unless context is reduced.

**Constructive fix:** update rollback to a known-safe recipe (e.g., server `OLLAMA_NUM_PARALLEL=2` + client `--concurrency 2` + `num_ctx=2048`), and explicitly include the required server configuration steps.

### 4.8 Phase 1 installation steps are under-specified (reproducibility + supply chain)
The plan suggests installing `llama-cpp-python` from a custom wheel index and optionally building with CUDA via `apt install nvidia-cuda-toolkit` (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L17`).

Skeptical concerns:
- The custom wheel index is a dependency trust decision; the plan should pin versions and document the expected CUDA/Python matrix.
- The Ubuntu `nvidia-cuda-toolkit` package is not always aligned with the CUDA version implied by the wheel (`cu125`); “build from source” often takes longer than 15 minutes if you hit toolchain issues.
- If this is meant to be repeatable for future rebuilds, it should live in repo docs/scripts (CLAUDE.md#L5) rather than only in a one-off plan file.

Constructive options:
- **A (lowest friction):** use the prebuilt wheel + pin a known-good version in documentation.
- **B (most reproducible):** provide a small `scripts/install_llamacpp.sh` that checks GPU arch, python version, and installs the right wheel/build flags.
- **C (containers):** encapsulate llama.cpp in a container (or run `llama-server`) to keep host toolchains stable.

## 5) What’s missing for a production-quality `LlamaCppClient` in this repo

### 5.1 Dependency management (optional extras)
The extraction package currently has a small dependency set and uses lazy imports for optional backends (packages/extraction/pyproject.toml#L1; packages/extraction/src/research_kb_extraction/anthropic_client.py#L150). The plan installs `llama-cpp-python` manually (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L19`), but doesn’t define an optional dependency extra.

**Option:**
- Add `llamacpp` extra: `pip install -e "packages/extraction[llamacpp]"`.
  - Pros: reproducible installs; clearer docs.
  - Cons: you must decide which CUDA wheel index/version is supported (and that’s platform-specific).

### 5.2 Schema single-sourcing (avoid 3 diverging schemas)
Right now, you have schema in three places:
- prompt constraints (packages/extraction/src/research_kb_extraction/prompts.py#L31),
- Pydantic model literals (packages/extraction/src/research_kb_extraction/models.py#L24),
- Anthropic tool schema (packages/extraction/src/research_kb_extraction/anthropic_client.py#L23).

The plan adds a *fourth* copy: `EXTRACTION_SCHEMA` (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L62`).

**Constructive fix:** generate the JSON schema from the Pydantic model (or from a single canonical definition) and feed it to:
- Anthropic tools,
- llama.cpp grammar/schema,
- (optionally) prompt docs for Ollama.

Pros: less drift, easier ontology changes.
Cons: more upfront plumbing; Pydantic schema may be verbose and need simplification.

### 5.3 Integration points beyond `get_llm_client()`
The plan updates the factory (plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L220`), but misses:
- `scripts/extract_concepts.py` backend choices/help text (scripts/extract_concepts.py#L657),
- docs stating “extraction via Ollama” (docs/SYSTEM_DESIGN.md#L75; CLAUDE.md#L131),
- tests for the new client (packages/extraction/tests/test_ollama_client.py#L1 provides a pattern).

### 5.4 Observability for silent failure modes (must-have)
Both existing backends return `ChunkExtraction()` on validation error (Ollama: packages/extraction/src/research_kb_extraction/ollama_client.py#L200; Anthropic: packages/extraction/src/research_kb_extraction/anthropic_client.py#L251).

If llama.cpp introduces more parse/validation failures (e.g., truncation), you can silently tank recall while the pipeline “succeeds”.

**Minimum recommendation:** record and report, per backend:
- JSON/validation failure rate,
- mean/median/p95 latency per chunk,
- concepts/chunk and relationships/chunk,
- number of “empty extractions” per N chunks.

## 6) Options to move forward (with pros/cons)

### Option 1: Optimize Ollama first (lowest effort, validates assumptions)
**What:** enable server-side performance settings + reduce context; then reassess whether llama.cpp is still worth it.
**Pros:** minimal code; keeps current tested client; reduces risk.
**Cons:** still depends on systemd service + HTTP; “schema enforcement” remains prompt+validation.

Relevant code: Ollama JSON mode and `num_ctx` (packages/extraction/src/research_kb_extraction/ollama_client.py#L105).

### Option 2: Add `LlamaCppClient` as a third backend (recommended framing)
**What:** implement llamacpp backend as optional; keep Ollama and Anthropic.
**Pros:** flexibility; can benchmark and choose best backend per job.
**Cons:** more surface area (deps, CI, docs, support).

Alignment: repo explicitly anticipates it (packages/extraction/src/research_kb_extraction/base_client.py#L13).

### Option 3: Use `llama-server` (HTTP) instead of embedding llama.cpp in-process
**What:** run llama.cpp as a server and talk over HTTP, similar to Ollama.
**Pros:** isolates crashes/leaks; easier concurrency control; simpler Python client.
**Cons:** gives back some “no HTTP overhead” benefit; still an extra service.

### Option 4: Use Anthropic for extraction runs; reserve local LLMs for dev
**What:** if cost is acceptable, Anthropic gives much faster latency (packages/extraction/src/research_kb_extraction/anthropic_client.py#L110).
**Pros:** fastest time-to-results; no GPU constraints.
**Cons:** cost + rate limits; still needs robustness (tool schema is not guaranteed).

### Option 5: Scale out with multi-GPU (bigger win than 1.8×)
**What:** run two extraction workers pinned to different GPUs (process-level parallelism).
**Pros:** near-linear throughput scaling; avoids thread-safety issues.
**Cons:** requires 2nd GPU + thermal planning; more operational complexity.

## 7) Suggested revised plan (go/no-go gates)

1) **Benchmark baseline**: optimized Ollama with `num_ctx=2048` and safe parallel settings; record throughput + failure rate.
2) **Prototype llamacpp**: verify the exact llama-cpp-python API for schema/grammar output; confirm chat template correctness.
3) **Implement**: `LlamaCppClient` + factory + script CLI backend + docs updates + tests.
4) **Add robustness**: retry/repair on validation failures and report failure rate explicitly.
5) **A/B on a fixed chunk set**: compare quality (concept count, typing accuracy) and speed.
6) **Decide**: keep as optional backend, switch default only if quality is equal and speed gain is real under your workload.

## References
- Plan: `$HOME/.claude/plans/parallel-waddling-hanrahan.md#L1`
- Repo goals (graph-boosted retrieval): `README.md#L1` and `docs/SYSTEM_DESIGN.md#L8`
- LLM client interface + future llama.cpp mention: `packages/extraction/src/research_kb_extraction/base_client.py#L13`
- Ollama JSON mode + num_ctx default: `packages/extraction/src/research_kb_extraction/ollama_client.py#L38`
- Anthropic tool schema + tool_choice: `packages/extraction/src/research_kb_extraction/anthropic_client.py#L23`
- Prompt constraints (enum values): `packages/extraction/src/research_kb_extraction/prompts.py#L31`
- Pydantic enum types: `packages/extraction/src/research_kb_extraction/models.py#L13`
- Extraction script backend choices and concurrency semantics: `scripts/extract_concepts.py#L653`
- Backup guardrail: `scripts/extract_concepts.py#L216` and `CLAUDE.md#L192`
- DB enum constraints: `packages/storage/migrations/002_knowledge_graph.sql#L6`
- Ollama systemd service (no overrides): `/etc/systemd/system/ollama.service#L1`
