#!/bin/bash
# Batch ingestion: all gap-fill domains + SE sweep + papers
# Run with: nohup bash scripts/run_batch_ingest.sh > /tmp/batch_ingest.log 2>&1 &
#
# Uses --no-embed for two-phase GPU workflow.
# After completion, run: python scripts/backfill_embeddings.py

set -euo pipefail
cd "$(dirname "$0")/.."

CATALOG="fixtures/library_catalog/catalog_highmed.json"
BASE_DIR="$HOME/Claude/research-kb/fixtures/library_books/"
STARTED=$(date +%s)

echo "=== Batch Ingestion Started: $(date) ==="
echo ""

# Phase 1: Gap-fill sparse domains
# (numerical_methods already done in prior run, checkpoint will skip it)
for domain in linear_algebra machine_learning; do
    echo "--- Phase 1: $domain ---"
    python -u scripts/mass_ingest_catalog.py \
        --domain "$domain" --no-embed --quiet \
        --catalog "$CATALOG" \
        --base-dir "$BASE_DIR" 2>&1
    echo "  Completed: $(date)"
    echo ""
done

# Phase 2: SE/Python sweep
echo "--- Phase 2: software_engineering ---"
python -u scripts/mass_ingest_catalog.py \
    --domain software_engineering --no-embed --quiet \
    --catalog "$CATALOG" \
    --base-dir "$BASE_DIR" 2>&1
echo "  Completed: $(date)"
echo ""

# Phase 3: Papers (arXiv + annuity)
echo "--- Phase 3: Papers (arxiv + annuity + misc) ---"
python -u scripts/ingest_missing_papers.py --no-embed 2>&1
echo "  Completed: $(date)"
echo ""

# Phase 3d: Vibe Engineering textbook
echo "--- Phase 3d: Vibe Engineering ---"
python -u scripts/ingest_missing_textbooks.py --no-embed --quiet 2>&1
echo "  Completed: $(date)"
echo ""

ENDED=$(date +%s)
ELAPSED=$(( ENDED - STARTED ))
echo "=== Batch Ingestion Done: $(date) (${ELAPSED}s) ==="
echo "Next: python scripts/backfill_embeddings.py"
