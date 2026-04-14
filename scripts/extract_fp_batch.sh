#!/bin/bash
# Two-phase MinerU extraction for FP + SE-with-FP books.
#
# Phase 1a: Docling-ingest 5 new books via ingest_fp_batch.py
#           (creates sources rows with Docling chunks)
# Phase 1b: MinerU re-extract all FP + software_engineering sources
#           via reextract_with_mineru.py (overwrites chunks with MinerU quality)
#
# GPU lifecycle: stops Ollama + embed_server + rerank_server before Phase 1
# to free VRAM for MinerU/Docling. Trap-based EXIT handler guarantees
# service restart even on crash.
#
# Usage:
#     bash scripts/extract_fp_batch.sh
#
# Sudo: requires passwordless sudo for `systemctl stop/start ollama` OR
# invoke via Claude Code `!` REPL prefix so the terminal handles the prompt.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    VENV_PYTHON=$(command -v python3)
fi

OLLAMA_WAS_RUNNING=0
if systemctl is-active --quiet ollama; then
    OLLAMA_WAS_RUNNING=1
fi

cleanup() {
    local exit_code=$?
    echo ""
    echo "=== CLEANUP (exit code $exit_code) ==="
    if [ "$OLLAMA_WAS_RUNNING" = "1" ]; then
        echo "Restarting Ollama..."
        sudo systemctl start ollama || echo "  WARN: failed to restart ollama"
    fi
    echo "Done. embed_server and rerank_server NOT auto-restarted — restart manually if needed."
    exit $exit_code
}
trap cleanup EXIT INT TERM

echo "=== Pre-flight ==="
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{printf "VRAM free: %s MiB\n", $1}'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | head -20

OLLAMA_MODELS=$(curl -s http://localhost:11434/api/ps 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get("models",[])))' 2>/dev/null || echo "?")
echo "Ollama loaded models: $OLLAMA_MODELS"

echo ""
echo "=== Phase 1: Free GPU ==="
if [ "$OLLAMA_WAS_RUNNING" = "1" ]; then
    echo "Stopping ollama.service (requires sudo)..."
    sudo systemctl stop ollama
fi

echo "Killing embed_server + rerank_server..."
pkill -f 'research_kb_pdf.embed_server' 2>/dev/null || echo "  embed_server not running"
pkill -f 'research_kb_pdf.rerank_server' 2>/dev/null || echo "  rerank_server not running"

sleep 3
echo "Post-kill VRAM state:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{printf "  free: %s MiB (need >=4500)\n", $1}'

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}')
if [ "$FREE_MIB" -lt 4500 ]; then
    echo "ERROR: insufficient VRAM ($FREE_MIB MiB < 4500 MiB). Aborting."
    exit 2
fi

SOURCE_IDS_FILE="$(mktemp -t fp_batch_source_ids.XXXXXX)"
trap 'rm -f "$SOURCE_IDS_FILE"; cleanup' EXIT INT TERM

echo ""
echo "=== Phase 1a: Docling-ingest 5 new books ==="
"$VENV_PYTHON" scripts/ingest_fp_batch.py --no-embed --source-ids-out "$SOURCE_IDS_FILE"

echo ""
echo "Newly created source_ids:"
cat "$SOURCE_IDS_FILE"

echo ""
echo "=== Phase 1b: MinerU re-extract all FP sources (2 new, clean slate) ==="
"$VENV_PYTHON" scripts/reextract_with_mineru.py \
    --domains functional_programming \
    --no-embed

echo ""
echo "=== Phase 1c: MinerU re-extract ONLY the 3 new SE books (by source_id) ==="
# Avoid touching the 162 other software_engineering sources
while IFS=$'\t' read -r domain source_id title; do
    if [ "$domain" = "software_engineering" ]; then
        echo ""
        echo "  Re-extracting: $title"
        "$VENV_PYTHON" scripts/reextract_with_mineru.py \
            --source-id "$source_id" \
            --no-embed
    fi
done < "$SOURCE_IDS_FILE"

echo ""
echo "=== Phase 1 COMPLETE ==="
echo "Next steps (manual):"
echo "  1. Restart embed_server: python -m research_kb_pdf.embed_server &"
echo "  2. Restart rerank_server: python -m research_kb_pdf.rerank_server &"
echo "  3. Backfill embeddings: python scripts/backfill_embeddings.py --batch-size 8"
echo "  4. Rebuild citations: python scripts/extract_citations.py && python scripts/build_citation_graph.py"
echo "  5. Regenerate status: python scripts/generate_status.py"
