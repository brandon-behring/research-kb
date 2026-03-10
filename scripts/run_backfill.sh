#!/bin/bash
# Backfill embeddings with desktop notification on completion
# Usage: ./scripts/run_backfill.sh [batch_size]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
BATCH_SIZE=${1:-32}
echo "Starting backfill with batch_size=$BATCH_SIZE at $(date)"

"$PYTHON" -u scripts/backfill_embeddings.py \
  --batch-size "$BATCH_SIZE" --json \
  > /tmp/backfill_report.json 2> /tmp/backfill_stderr.log

STATUS=$?
SUMMARY=$(tail -1 /tmp/backfill_report.json 2>/dev/null)
if [ $STATUS -eq 0 ]; then
    notify-send -u normal "research-kb" "Backfill complete: $SUMMARY"
else
    notify-send -u critical "research-kb" "Backfill FAILED (exit $STATUS). Check /tmp/backfill_stderr.log"
fi
echo "Finished at $(date) with exit $STATUS"
