# Research-KB Critique (codex)

## Snapshot (cited)
- Purpose: semantic search over causal inference literature using FTS + pgvector + knowledge-graph re-ranking (docs/SYSTEM_DESIGN.md; README.md “Graph-Boosted Search”).
- Ingestion basics: PDF → chunk (~300 tokens) → embed (BGE-large) → store (docs/SYSTEM_DESIGN.md “Ingestion Flow”).
- Infra: Postgres + pgvector via docker-compose; optional GROBID, API, pgAdmin (docker-compose.yml). Neo4j sync is optional in extraction (scripts/extract_concepts.py).
- Extraction stack: Ollama client default `num_ctx=4096`, JSON mode, temperature 0.1 (packages/extraction/src/research_kb_extraction/ollama_client.py). Runner script uses `--concurrency 1` and `--skip-backup` (scripts/extract_top10.sh).
- Models installed: only `llama3.1:8b` (ollama list). Service file has no OLLAMA_* env overrides (/etc/systemd/system/ollama.service). Ollama 0.13.0 (ollama --version).
- Hardware: Threadripper 3990X (128 threads), 251 GiB RAM, RTX 2070 SUPER 8GB (nvidia-smi --query).
- Plan context: $HOME/.claude/plans/parallel-waddling-hanrahan.md reports Haiku enum/validation errors and attributes the root cause to “missing structured output”; in this repo, `AnthropicClient` already uses tool schema + forced tool choice (packages/extraction/src/research_kb_extraction/anthropic_client.py) and defaults to `haiku-3.5` for better schema adherence (packages/extraction/src/research_kb_extraction/__init__.py).
- User-provided A/B: llama3.1-8B vs qwen2.5-7B shows +33% concepts and better typing for Llama with ~11% slower latency; Qwen had a JSON error.

## Strengths (cited)
- Clear modular packaging and async storage with pgvector (docs/SYSTEM_DESIGN.md “Package Dependency Graph”; packages layout).
- Graph-boosted retrieval first-class with defaults and fallbacks (README.md; docs/SYSTEM_DESIGN.md “Key Data Flow”).
- Extraction pipeline has checkpoints, DLQ, dedup cache, and Neo4j sync hook (scripts/extract_concepts.py).
- Dev ergonomics: docker-compose for DB/GROBID/API and CLI scripts for ingest/extract (README.md; docker-compose.yml; scripts/*).

## Issues & Risks (cited, ordered by impact)
1) **Extraction throughput under-utilizes GPU**  
   - Ollama service lacks `OLLAMA_*` performance envs (/etc/systemd/system/ollama.service).  
   - Client runs `num_ctx=4096` (ollama_client.py) and script pins `--concurrency 1` (scripts/extract_top10.sh) despite 8GB VRAM (nvidia-smi). For ~300-token chunks (docs/SYSTEM_DESIGN.md ingestion flow), this wastes capacity and matches the reported 26–50s/chunk (user measurement).

2) **Model quality vs speed trade-off**  
   - User A/B shows Llama3.1-8B yields ~33% more concepts and better typing; Qwen2.5-7B is ~11% faster but mislabels and had a JSON error (user result). Quality likely regresses if switching solely for speed.

3) **Anthropic backend reliability + plan/code mismatch**  
   - Plan reports Haiku returning invalid enum values and blames “missing structured output” ($HOME/.claude/plans/parallel-waddling-hanrahan.md), but the repo’s `AnthropicClient` already uses a tool schema with enums and forces tool use (packages/extraction/src/research_kb_extraction/anthropic_client.py).  
   - Even with tools, failures still happen in practice (no tool use, malformed/invalid inputs): the current implementation returns empty extraction on validation failure (packages/extraction/src/research_kb_extraction/anthropic_client.py; packages/extraction/src/research_kb_extraction/models.py). If you don’t track this rate, you can silently lose recall.

4) **Extraction tuning knobs are partly ineffective**  
   - `--batch-size` exists but doesn’t currently control parallel batch size: parallel mode hardcodes `batch_size = concurrency * 2` (scripts/extract_concepts.py). This makes throughput tuning less predictable and the CLI misleading.

5) **Safety and ops gaps**  
   - Long-run script uses `--skip-backup` and `--no-neo4j` (scripts/extract_top10.sh), bypassing the default pre-extraction backup guardrail in scripts/extract_concepts.py.  
   - No built-in metrics/telemetry for throughput/failures/GPU utilization in extraction code; debugging requires manual logs.

6) **Performance tuning not codified**  
   - No docs or defaults for flash attention, KV cache quantization, or parallel streams; service file is unmodified.  
   - Context window larger than needed for typical chunk length, increasing latency/VRAM.

7) **Housekeeping**  
   - License/contributing are placeholders (README.md). Only `llama3.1:8b` installed (ollama list), so no ready fallback/fast model.

## Recommendations with options (pros/cons, cited)
1) **Enable Ollama performance flags (service override)**  
   - Action: add `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_NUM_PARALLEL=2`, `OLLAMA_KV_CACHE_TYPE=q8_0`; daemon-reload/restart.  
   - Pros: No code changes; typically ~1.5–2x throughput on 8GB cards.  
   - Cons: Slight memory overhead; must ensure 2 streams fit VRAM.  
   - Source: current service lacks overrides (/etc/systemd/system/ollama.service).

2) **Right-size context and concurrency**  
   - Action: set `num_ctx` to 2048 (ollama_client.py or CLI option) and run with `--concurrency 2` for extraction runs.  
   - Pros: Lower per-request VRAM/latency; better GPU utilization for ~300-token chunks (docs/SYSTEM_DESIGN.md).  
   - Cons: If future prompts need >2048 tokens, may truncate; monitor for parsing edge cases.

3) **Codify a “fast vs quality” profile in scripts**  
   - Action: update scripts/extract_top10.sh (or add a new preset) to: enable backups by default, allow a `--fast` flag that sets `concurrency=2` and `num_ctx=2048`, and document when `--skip-backup` is acceptable.  
   - Pros: Repeatable, safer runs; clearer operator intent.  
   - Cons: Minor script maintenance; adds flag surface area.

4) **Harden Anthropic mode (tools already exist)**
   - Option A: Prefer `haiku-3.5` (already the factory default) or `sonnet` when you need schema adherence (packages/extraction/src/research_kb_extraction/__init__.py; packages/extraction/src/research_kb_extraction/anthropic_client.py).
     - Pros: Higher likelihood of valid tool calls; fast per-chunk latency.
     - Cons: Costs money; still not a hard guarantee.
   - Option B: Add retries + “repair” path when validation fails or no tool_use appears (log the failure rate and re-request with stricter instructions / smaller output).
     - Pros: Higher recall; failures become observable.
     - Cons: More API calls; increases cost; more code paths.
   - Option C: If the plan’s “missing structured output” reflects your runtime reality, verify you’re running the repo’s current `AnthropicClient` (pip install/editable mismatch is a common culprit).
     - Pros: Often fixes the issue without model switching.
     - Cons: Requires environment hygiene (editable installs, restart jobs).

5) **Lightweight observability**  
   - Action: emit per-chunk timing/failure counters and model/version into logs or a Prometheus textfile; optionally log GPU utilization snapshots during runs.  
   - Pros: Faster triage of slowdowns/errors; evidence for future tuning.  
   - Cons: Small code additions; need a place to scrape/store metrics.

6) **Model hygiene and documentation**  
   - Action: keep llama3.1-8B primary; optionally pre-pull `llama3.2:3b` or `qwen2.5:7b` for controlled fast-mode tests; fill license/contributing and document the recommended Ollama/systemd settings and extraction command in README/docs.  
   - Pros: Clear defaults and fallbacks; reduces tribal knowledge.  
   - Cons: Additional doc work; storage for extra models.

## Quick Win Playbook (cited)
- Today: add Ollama overrides and restart; drop `num_ctx` to 2048; run extraction with `--concurrency 2` (sources: ollama_client.py, scripts/extract_top10.sh, service file).  
- This week: make `--batch-size` effective; add a “fast” preset with backup guardrails; add extraction failure-rate logging (sources: scripts/extract_concepts.py; scripts/extract_top10.sh).  
- Next: add minimal metrics; document operational defaults and license/contrib so practices persist.
