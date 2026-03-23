#!/usr/bin/env bash
# Post-Rechunk Pipeline: Phases B through F
#
# Usage:
#   ./scripts/run_postrechunk_pipeline.sh [phase]
#
# Phases:
#   ingest       - Phase B: Ingest new textbooks from fixtures/textbooks/
#   extract      - Phase C: Extract concepts for weak domains (finance, sql, recsys)
#   extract-all  - Phase C+: Extract concepts for ALL domains (full restoration)
#   sync         - Phase D: Sync KuzuDB
#   optimize     - Phase E: Optimize search weights
#   restart      - Phase F: Restart services and validate
#   all          - Run phases B through F sequentially
#
# Examples:
#   ./scripts/run_postrechunk_pipeline.sh extract     # Just extraction
#   ./scripts/run_postrechunk_pipeline.sh all          # Full pipeline
#   nohup ./scripts/run_postrechunk_pipeline.sh extract-all > /tmp/pipeline.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_DIR/.venv/bin"
LOG_DIR="/tmp/research_kb_pipeline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()  { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
err() { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"; }

# ---- Phase B: Ingest new textbooks ----
phase_ingest() {
    log "Phase B: Ingesting new textbooks..."

    # Count PDFs without matching sidecar (warns about missing sidecars)
    local pdf_count
    pdf_count=$(find "$PROJECT_DIR/fixtures/textbooks/sql" "$PROJECT_DIR/fixtures/textbooks/recommender_systems" \
                     -name "*.pdf" 2>/dev/null | wc -l)

    if [ "$pdf_count" -eq 0 ]; then
        warn "No new PDFs found in sql/ or recommender_systems/ directories"
        warn "Place PDFs with sidecar JSONs in fixtures/textbooks/{sql,recommender_systems}/"
        return 1
    fi

    log "Found $pdf_count PDFs to ingest"

    # Check for GPU contention: Docling needs ~3GB, embed_server needs ~4GB
    if pgrep -f "embed_server" > /dev/null 2>&1; then
        warn "embed_server is running — Docling may OOM on GPU"
        warn "Consider: kill embed_server → ingest → restart embed_server → backfill"
        log "Proceeding with ingestion (Docling may fall back to CPU)..."
    fi

    cd "$PROJECT_DIR"
    "$VENV/python" -u scripts/ingest_missing_textbooks.py --quiet 2>&1 | tee "$LOG_DIR/ingest_${TIMESTAMP}.log"

    ok "Phase B complete — check log: $LOG_DIR/ingest_${TIMESTAMP}.log"
}

# ---- Phase C: Extract concepts for weak domains ----
phase_extract() {
    log "Phase C: Extracting concepts for weak domains..."

    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        err "ANTHROPIC_API_KEY not set — extraction requires Anthropic API"
        return 1
    fi

    # Clear stale checkpoint (all old chunk IDs from pre-rechunk)
    if [ -f "$PROJECT_DIR/.extraction_checkpoint.json" ]; then
        log "Clearing stale extraction checkpoint..."
        rm "$PROJECT_DIR/.extraction_checkpoint.json"
    fi

    cd "$PROJECT_DIR"

    local domains=("finance" "sql" "recommender_systems")
    for domain in "${domains[@]}"; do
        log "Extracting: $domain"

        # Check if domain has chunks
        local chunk_count
        chunk_count=$("$VENV/python" -c "
import asyncio
from research_kb_storage import DatabaseConfig, get_connection_pool
async def check():
    config = DatabaseConfig()
    pool = await get_connection_pool(config)
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            \"SELECT count(*) FROM chunks c JOIN sources s ON c.source_id = s.id WHERE s.domain_id = \\\$1\",
            '$domain'
        )
        print(count)
    await pool.close()
asyncio.run(check())
" 2>/dev/null | tail -1)

        if [ "${chunk_count:-0}" -eq 0 ]; then
            warn "  $domain: 0 chunks — skipping"
            continue
        fi

        log "  $domain: $chunk_count chunks to extract"

        "$VENV/python" -u scripts/extract_concepts.py \
            --backend anthropic --model haiku-4.5 --concurrency 4 \
            --source-domain "$domain" --domain "$domain" \
            --clear-checkpoint --skip-backup \
            --metrics-file "$LOG_DIR/extraction_${domain}_${TIMESTAMP}.txt" \
            2>&1 | tee "$LOG_DIR/extraction_${domain}_${TIMESTAMP}.log"

        ok "  $domain extraction complete"
    done

    ok "Phase C complete"
}

# ---- Phase C+: Extract ALL domains ----
phase_extract_all() {
    log "Phase C+: Extracting concepts for ALL domains..."

    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        err "ANTHROPIC_API_KEY not set — extraction requires Anthropic API"
        return 1
    fi

    # Clear stale checkpoint
    if [ -f "$PROJECT_DIR/.extraction_checkpoint.json" ]; then
        log "Clearing stale extraction checkpoint..."
        rm "$PROJECT_DIR/.extraction_checkpoint.json"
    fi

    cd "$PROJECT_DIR"

    # Get all domains with unextracted chunks
    local domains
    domains=$("$VENV/python" -c "
import asyncio
from research_kb_storage import DatabaseConfig, get_connection_pool
async def check():
    config = DatabaseConfig()
    pool = await get_connection_pool(config)
    async with pool.acquire() as conn:
        rows = await conn.fetch('''
            SELECT s.domain_id, count(c.id) as cnt
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE c.id NOT IN (SELECT DISTINCT chunk_id FROM chunk_concepts)
            AND s.domain_id IS NOT NULL
            GROUP BY s.domain_id
            HAVING count(c.id) > 0
            ORDER BY count(c.id) DESC
        ''')
        for r in rows:
            print(f\"{r['domain_id']}:{r['cnt']}\")
    await pool.close()
asyncio.run(check())
" 2>/dev/null | grep -v '^\[')

    if [ -z "$domains" ]; then
        warn "No unextracted chunks found"
        return 0
    fi

    log "Domains to extract:"
    echo "$domains" | while IFS=: read -r domain count; do
        log "  $domain: $count chunks"
    done

    echo "$domains" | while IFS=: read -r domain count; do
        log "Extracting: $domain ($count chunks)"

        "$VENV/python" -u scripts/extract_concepts.py \
            --backend anthropic --model haiku-4.5 --concurrency 4 \
            --source-domain "$domain" --domain "$domain" \
            --clear-checkpoint --skip-backup \
            --metrics-file "$LOG_DIR/extraction_${domain}_${TIMESTAMP}.txt" \
            2>&1 | tee "$LOG_DIR/extraction_${domain}_${TIMESTAMP}.log"

        ok "  $domain extraction complete"
    done

    ok "Phase C+ complete (all domains extracted)"
}

# ---- Phase D: KuzuDB Sync ----
phase_sync() {
    log "Phase D: Syncing KuzuDB..."

    # Kill daemon if running (holds KuzuDB lock)
    if pgrep -f "research-kb-daemon" > /dev/null 2>&1; then
        warn "Daemon running — stopping to release KuzuDB lock"
        systemctl --user stop research-kb-daemon 2>/dev/null || true
        sleep 2
    fi

    cd "$PROJECT_DIR"
    "$VENV/python" scripts/sync_kuzu.py 2>&1 | tee "$LOG_DIR/sync_${TIMESTAMP}.log"

    log "Verifying sync..."
    "$VENV/python" scripts/sync_kuzu.py --verify-only 2>&1

    ok "Phase D complete"
}

# ---- Phase E: Weight Optimization ----
phase_optimize() {
    log "Phase E: Optimizing search weights..."

    cd "$PROJECT_DIR"

    # Dry run first
    log "Baseline (current weights):"
    "$VENV/python" scripts/optimize_weights.py --dry-run 2>&1 | tee "$LOG_DIR/optimize_dryrun_${TIMESTAMP}.log"

    # Full optimization with cross-validation
    log "Running optimization with cross-validation..."
    "$VENV/python" scripts/optimize_weights.py --cross-validate \
        --output "$LOG_DIR/weights_${TIMESTAMP}.json" \
        2>&1 | tee "$LOG_DIR/optimize_${TIMESTAMP}.log"

    # Also run RRF comparison
    log "RRF scoring comparison:"
    "$VENV/python" scripts/eval_retrieval.py --scoring rrf --per-domain \
        2>&1 | tee "$LOG_DIR/eval_rrf_${TIMESTAMP}.log"

    ok "Phase E complete — review $LOG_DIR/weights_${TIMESTAMP}.json"
    warn "Apply optimized weights manually to:"
    warn "  1. packages/cli/src/research_kb_cli/_shared.py:74-90"
    warn "  2. packages/mcp-server/src/research_kb_mcp/tools/search.py:20-34"
    warn "  3. packages/api/src/research_kb_api/service.py"
}

# ---- Phase F: Service Restart + Validation ----
phase_restart() {
    log "Phase F: Restarting services and validating..."

    # Start daemon
    systemctl --user start research-kb-daemon 2>/dev/null || {
        warn "systemd start failed — starting daemon directly"
        cd "$PROJECT_DIR"
        "$VENV/python" -m research_kb_daemon &
        sleep 3
    }

    cd "$PROJECT_DIR"

    # Validation queries
    log "Running validation queries..."

    local queries=(
        "instrumental variables"
        "Black-Scholes option pricing"
        "collaborative filtering matrix factorization"
        "database normalization"
    )

    for q in "${queries[@]}"; do
        log "  Query: '$q'"
        "$VENV/research-kb" search query "$q" --limit 3 2>&1 | head -20
        echo "---"
    done

    # Stats
    log "Database stats:"
    "$VENV/research-kb" sources stats

    # Graph check
    log "Graph check:"
    "$VENV/research-kb" graph concepts "DML" 2>&1 | head -10

    # Full eval
    log "Running full eval..."
    "$VENV/python" scripts/eval_retrieval.py --per-domain --fail-below 0.7 \
        2>&1 | tee "$LOG_DIR/eval_final_${TIMESTAMP}.log"

    ok "Phase F complete"
}

# ---- Main ----
main() {
    local phase="${1:-help}"

    case "$phase" in
        ingest)
            phase_ingest
            ;;
        extract)
            phase_extract
            ;;
        extract-all)
            phase_extract_all
            ;;
        sync)
            phase_sync
            ;;
        optimize)
            phase_optimize
            ;;
        restart)
            phase_restart
            ;;
        all)
            phase_ingest || warn "Ingest skipped (no new textbooks)"
            phase_extract
            phase_sync
            phase_optimize
            phase_restart
            ok "Full pipeline complete!"
            ;;
        help|*)
            echo "Usage: $0 {ingest|extract|extract-all|sync|optimize|restart|all}"
            echo ""
            echo "Phases:"
            echo "  ingest       Phase B: Ingest new textbooks"
            echo "  extract      Phase C: Extract concepts (3 weak domains)"
            echo "  extract-all  Phase C+: Extract concepts (ALL domains)"
            echo "  sync         Phase D: Sync KuzuDB"
            echo "  optimize     Phase E: Optimize search weights"
            echo "  restart      Phase F: Restart services + validate"
            echo "  all          Run B through F sequentially"
            echo ""
            echo "Logs: $LOG_DIR/"
            ;;
    esac
}

main "$@"
