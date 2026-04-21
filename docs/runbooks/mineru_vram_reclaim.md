# MinerU VRAM Reclaim Runbook

MinerU extraction (magic_pdf layout + UniMERNet formula recognizer)
requires at least **4000 MiB** of free VRAM to initialize reliably.
On the RTX 2070 SUPER workstation (7.6 GB total, ~590 MiB free at
rest with default services running), the rerank_server is the single
dominant consumer that must be reclaimed before running MinerU.

This runbook documents (1) the VRAM budget, (2) the reclaim
protocol, and (3) the preflight integration in
`scripts/reextract_with_mineru.py`.

## VRAM budget on RTX 2070 SUPER (7600 MiB total)

| Consumer | MiB | Service file | Notes |
|---|---:|---|---|
| `research_kb_rerank.service` (BGE-reranker-v2-m3) | ~5774 | `research_kb_rerank.service:25` | **GPU** via `CUDA_VISIBLE_DEVICES=0`. Reclaim target. |
| `research_kb_embed.service` (BGE-large-en-v1.5) | 0 on GPU | `research_kb_embed.service:27` | Runs on CPU via `CUDA_VISIBLE_DEVICES=""`. Not a GPU consumer. |
| `gnome-shell` + Xwayland | ~590 | — | Desktop baseline. Cannot reclaim without logging out. |
| Ollama | 0 at idle | — | Not persistent; loads on first query and unloads per `OLLAMA_KEEP_ALIVE`. |
| **MinerU floor** | **4000** | `gpu_guard.py` (`DEFAULT_MINERU_VRAM_MIN_MB`) | Minimum free VRAM to start MinerU. |

At rest with rerank running: ~590 MiB free, which is **below** the
MinerU floor — MinerU will OOM during model load or during a
formula-heavy page batch.

## Reclaim protocol

### Option A — let the script manage services (recommended)

Use the `--auto-stop-services` flag. The script runs a VRAM preflight
via `abort_if_vram_insufficient_for_mineru` (see
`packages/common/src/research_kb_common/gpu_guard.py`), stops
`research_kb_rerank.service` if needed, runs the MinerU batch, and
restarts rerank in a `finally` block regardless of success or
failure.

```bash
python scripts/reextract_with_mineru.py \
    --source-id <uuid> \
    --no-embed \
    --auto-stop-services
```

### Option B — manage services manually

Stop rerank, run MinerU, restart rerank:

```bash
systemctl --user stop research_kb_rerank.service
python scripts/reextract_with_mineru.py --source-id <uuid> --no-embed
systemctl --user start research_kb_rerank.service
```

This is the path the preflight recommends when called **without**
`--auto-stop-services`. The preflight aborts with:

```
[mineru preflight] VRAM 590 MiB free, need 4000 MiB.
detected managed GPU consumer(s): research_kb_rerank.service

Reclaim and re-run:
  systemctl --user stop research_kb_rerank.service

Or let the script manage it:
  python scripts/reextract_with_mineru.py ... --auto-stop-services

aborting.
```

## Verifying VRAM state

```bash
# Real-time reading used by the preflight (wraps torch.cuda.mem_get_info)
.venv/bin/python -c "
from research_kb_common.gpu_guard import check_vram_for_mineru
ok, free = check_vram_for_mineru()
print(f'ok={ok}, free={free} MiB')
"

# Or with nvidia-smi
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

# Or with the richer stats helper
.venv/bin/python -c "
from research_kb_common.gpu_guard import get_vram_stats
print(get_vram_stats())
"
```

## Post-batch embedding backfill

MinerU extraction runs `--no-embed` by default (chunks are written
with NULL embeddings). After the batch:

```bash
# 1. Restart rerank_server if you stopped it manually
systemctl --user start research_kb_rerank.service

# 2. Ensure embed_server is up (it stays on CPU — safe to run always)
systemctl --user start research_kb_embed.service

# 3. Backfill embeddings with adaptive batching
python scripts/backfill_embeddings.py --batch-size 8
```

Backfill uses the VRAM ceiling + adaptive batch sizing in
`gpu_guard.py` (`set_vram_ceiling`, `get_safe_batch_size`) to stay
within the 35% fraction cap.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Preflight reports ok but MinerU still OOMs | Another consumer (e.g. Ollama loaded a model) claimed VRAM during MinerU model load | Run `nvidia-smi` and stop the consumer. Consider raising `--mineru-vram-min-mib` to add headroom. |
| Script aborts even with `--auto-stop-services` | An unmanaged consumer (outside `MINERU_MANAGED_SERVICES`) is holding VRAM | Stop it manually. File an issue to add it to the managed list if it's a regular consumer. |
| Restart fails after batch | `systemctl --user start` returned non-zero | The preflight logs the error but does not raise. Check `systemctl --user status research_kb_rerank.service` and restart manually. |

## Scope

The preflight currently manages only `research_kb_rerank.service`.
Other ingestion scripts (`ingest_missing_textbooks.py`,
`ingest_missing_papers.py`, `mass_ingest_catalog.py`) use the
older `abort_if_embed_server_running()` contract. Unifying both
patterns is tracked separately — see the follow-up issue filed
from the 2026-04-21 plan (label: `tracked,improvement,P3`).
