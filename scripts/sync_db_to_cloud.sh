#!/bin/bash
# Nightly compressed DB sync: PostgreSQL -> Google Drive via rclone.
#
# Creates a pg_dump custom-format binary dump (~4-6 GB vs 24 GB plain SQL)
# and uploads it to gdrive:/research-kb-db/ with retries. Writes heartbeat
# markers on success and failure markers on any error. Designed for nightly
# cron; no silent failures.
#
# Usage:
#   ./scripts/sync_db_to_cloud.sh               # dump + upload
#   ./scripts/sync_db_to_cloud.sh --dump-only   # skip upload
#   ./scripts/sync_db_to_cloud.sh --no-upload   # alias for --dump-only
#
# Exit codes:
#   0 - Success
#   1 - Dump failed (Postgres down, disk full, etc.)
#   2 - Upload failed after retries
#   3 - Post-upload size verification failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="$REPO_ROOT/backups"
DUMP_FILE="$BACKUP_DIR/research_kb_latest.dump"
TMP_FILE="$DUMP_FILE.tmp"
SUCCESS_MARKER="$BACKUP_DIR/.last_sync_ok"
FAILURE_MARKER="$BACKUP_DIR/.last_sync_FAILED"

REMOTE="gdrive:/research-kb-db"
CONTAINER="research-kb-postgres"

DUMP_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --dump-only|--no-upload) DUMP_ONLY=true ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
    esac
done

mkdir -p "$BACKUP_DIR"

log() {
    echo "[$(date -Iseconds)] $*"
}

fail() {
    local code="$1"; shift
    local msg="$*"
    log "ERROR: $msg" >&2
    printf '%s\n%s\n' "$(date -Iseconds)" "$msg" > "$FAILURE_MARKER"
    rm -f "$TMP_FILE"
    exit "$code"
}

# 1. Verify Postgres container is up and responsive
if ! docker exec "$CONTAINER" psql -U postgres -d research_kb -c "SELECT 1" &>/dev/null; then
    fail 1 "Postgres container '$CONTAINER' not running or not accepting queries"
fi

# 2. Dump to temp file (atomic rename on success)
log "Dumping research_kb (pg_dump -Fc -Z 9) -> $TMP_FILE"
DUMP_START=$(date +%s)
if ! docker exec "$CONTAINER" pg_dump -U postgres -Fc -Z 9 research_kb > "$TMP_FILE"; then
    fail 1 "pg_dump failed"
fi
DUMP_END=$(date +%s)
DUMP_SECS=$((DUMP_END - DUMP_START))

# Sanity check: dump must be non-trivial
DUMP_SIZE=$(stat -c%s "$TMP_FILE")
if (( DUMP_SIZE < 10000000 )); then
    fail 1 "Dump suspiciously small ($DUMP_SIZE bytes) - likely failed"
fi

# Atomic swap
mv "$TMP_FILE" "$DUMP_FILE"
DUMP_HUMAN=$(du -h "$DUMP_FILE" | cut -f1)
log "Dump complete: $DUMP_HUMAN in ${DUMP_SECS}s"

# 3. Upload unless --dump-only
if [ "$DUMP_ONLY" = true ]; then
    log "Skipping upload (--dump-only)"
    rm -f "$FAILURE_MARKER"
    date -Iseconds > "$SUCCESS_MARKER"
    exit 0
fi

if ! command -v rclone &>/dev/null; then
    fail 2 "rclone not installed"
fi

UPLOAD_START=$(date +%s)
RETRY_WAITS=(30 120 480)
UPLOAD_OK=false
for attempt in 1 2 3; do
    log "Upload attempt $attempt/3 -> $REMOTE"
    if rclone copy --checksum "$DUMP_FILE" "$REMOTE/" 2>&1; then
        UPLOAD_OK=true
        break
    fi
    if (( attempt < 3 )); then
        wait_s=${RETRY_WAITS[$((attempt - 1))]}
        log "Upload attempt $attempt failed, retrying in ${wait_s}s"
        sleep "$wait_s"
    fi
done
UPLOAD_END=$(date +%s)
UPLOAD_SECS=$((UPLOAD_END - UPLOAD_START))

if [ "$UPLOAD_OK" != true ]; then
    fail 2 "rclone upload failed after 3 attempts ($((UPLOAD_SECS))s elapsed)"
fi

# 4. Verify remote size matches local
REMOTE_SIZE=$(rclone size --json "$REMOTE/research_kb_latest.dump" 2>/dev/null | grep -o '"bytes":[0-9]*' | head -1 | cut -d: -f2 || echo "0")
if [ "$REMOTE_SIZE" != "$DUMP_SIZE" ]; then
    fail 3 "Remote size mismatch: local=$DUMP_SIZE remote=$REMOTE_SIZE"
fi

log "Upload verified: $DUMP_HUMAN in ${UPLOAD_SECS}s"

# 5. Success: clear failure marker, write heartbeat
rm -f "$FAILURE_MARKER"
date -Iseconds > "$SUCCESS_MARKER"

log "Sync complete (dump ${DUMP_SECS}s + upload ${UPLOAD_SECS}s = $((DUMP_SECS + UPLOAD_SECS))s total)"
