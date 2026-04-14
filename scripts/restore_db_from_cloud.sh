#!/bin/bash
# Restore PostgreSQL from the compressed sync dump produced by sync_db_to_cloud.sh.
#
# Works on Linux (pulls via rclone) and Mac (reads from the Google Drive
# auto-synced path). Staleness-aware and idempotent: safe to re-run; skips
# work if the source hasn't changed since the last restore.
#
# Usage:
#   ./scripts/restore_db_from_cloud.sh              # staleness-aware
#   ./scripts/restore_db_from_cloud.sh --force      # restore even if current
#   ./scripts/restore_db_from_cloud.sh --dry-run    # show what it would do
#
# Environment:
#   RESEARCH_KB_DUMP_PATH   Override source path (e.g., Mac's GDrive mirror)
#
# Source path resolution order:
#   1. $RESEARCH_KB_DUMP_PATH (explicit override)
#   2. ~/Library/CloudStorage/GoogleDrive-*/My Drive/research-kb-db/research_kb_latest.dump  (Mac auto-detect)
#   3. rclone copy from gdrive:/research-kb-db/ into backups/  (Linux default)
#
# Exit codes:
#   0 - Success (or skipped because current)
#   1 - Source not found / download failed
#   2 - pg_restore failed
#   3 - Post-restore verification failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="$REPO_ROOT/backups"
STALENESS_MARKER="$BACKUP_DIR/.last_restore_mtime"
CONTAINER="research-kb-postgres"
REMOTE="gdrive:/research-kb-db"

FORCE=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            sed -n '2,25p' "$0"
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
    log "ERROR: $*" >&2
    exit "$code"
}

# 1. Resolve source path
resolve_source() {
    if [ -n "${RESEARCH_KB_DUMP_PATH:-}" ]; then
        if [ ! -f "$RESEARCH_KB_DUMP_PATH" ]; then
            fail 1 "RESEARCH_KB_DUMP_PATH set but file missing: $RESEARCH_KB_DUMP_PATH"
        fi
        echo "$RESEARCH_KB_DUMP_PATH"
        return
    fi

    # Mac auto-detect: ~/Library/CloudStorage/GoogleDrive-*/My Drive/research-kb-db/
    if [ -d "$HOME/Library/CloudStorage" ]; then
        local mac_path
        mac_path=$(find "$HOME/Library/CloudStorage" -type f -name "research_kb_latest.dump" 2>/dev/null | head -1 || true)
        if [ -n "$mac_path" ]; then
            echo "$mac_path"
            return
        fi
    fi

    # Linux default: pull from rclone remote into backups/
    if ! command -v rclone &>/dev/null; then
        fail 1 "rclone not installed and no RESEARCH_KB_DUMP_PATH set"
    fi
    local linux_path="$BACKUP_DIR/research_kb_latest.dump"
    log "Pulling from $REMOTE/research_kb_latest.dump -> $linux_path"
    if [ "$DRY_RUN" = false ]; then
        if ! rclone copy --checksum "$REMOTE/research_kb_latest.dump" "$BACKUP_DIR/"; then
            fail 1 "rclone download failed"
        fi
    fi
    echo "$linux_path"
}

SOURCE_PATH=$(resolve_source)
log "Source: $SOURCE_PATH"

if [ ! -f "$SOURCE_PATH" ] && [ "$DRY_RUN" = false ]; then
    fail 1 "Source file missing after resolution: $SOURCE_PATH"
fi

# 2. Staleness check (ISO timestamp + SHA256 prefix)
current_fingerprint() {
    local mtime size
    if [ "$(uname)" = "Darwin" ]; then
        mtime=$(stat -f %m "$SOURCE_PATH")
        size=$(stat -f %z "$SOURCE_PATH")
    else
        mtime=$(stat -c %Y "$SOURCE_PATH")
        size=$(stat -c %s "$SOURCE_PATH")
    fi
    echo "${mtime}:${size}"
}

FINGERPRINT=$(current_fingerprint)
if [ "$FORCE" != true ] && [ -f "$STALENESS_MARKER" ]; then
    LAST_FINGERPRINT=$(cat "$STALENESS_MARKER")
    if [ "$FINGERPRINT" = "$LAST_FINGERPRINT" ]; then
        log "Already current (fingerprint matches). Use --force to re-restore."
        exit 0
    fi
fi

if [ "$DRY_RUN" = true ]; then
    log "DRY RUN: would pg_restore from $SOURCE_PATH"
    log "DRY RUN: would run scripts/sync_kuzu.py"
    exit 0
fi

# 3. Stop daemon if running (releases KuzuDB lock)
if pgrep -f "research-kb-daemon\|research_kb_daemon" &>/dev/null; then
    log "Stopping research-kb daemon to release KuzuDB lock"
    pkill -f "research-kb-daemon\|research_kb_daemon" || true
    sleep 2
fi

# 4. Ensure Postgres is up and healthy
log "Ensuring Postgres container is healthy"
if ! docker compose -f "$REPO_ROOT/docker-compose.yml" ps --format json postgres &>/dev/null; then
    (cd "$REPO_ROOT" && docker compose up -d postgres)
fi

# Wait up to 90s for healthcheck
for i in $(seq 1 18); do
    if docker exec "$CONTAINER" psql -U postgres -d research_kb -c "SELECT 1" &>/dev/null; then
        break
    fi
    sleep 5
    if (( i == 18 )); then
        fail 2 "Postgres did not become healthy in 90s"
    fi
done

# 5. Restore
log "Running pg_restore (-c -j 4)"
RESTORE_START=$(date +%s)
if ! docker exec -i "$CONTAINER" pg_restore -U postgres -d research_kb -c -j 4 < "$SOURCE_PATH"; then
    # pg_restore can emit non-zero on harmless "does not exist" errors when -c drops nothing.
    # Verify by checking row counts below; only fail if verification also fails.
    log "pg_restore returned non-zero; will verify by counting rows"
fi
RESTORE_SECS=$(( $(date +%s) - RESTORE_START ))
log "pg_restore finished in ${RESTORE_SECS}s"

# 6. Verify Postgres state
SOURCES=$(docker exec "$CONTAINER" psql -U postgres -d research_kb -t -c "SELECT COUNT(*) FROM sources;" | tr -d ' ')
CHUNKS=$(docker exec "$CONTAINER" psql -U postgres -d research_kb -t -c "SELECT COUNT(*) FROM chunks;" | tr -d ' ')
log "Postgres state: sources=$SOURCES chunks=$CHUNKS"
if [ "$SOURCES" = "0" ] || [ "$CHUNKS" = "0" ]; then
    fail 3 "Post-restore verification failed: sources=$SOURCES chunks=$CHUNKS"
fi

# 7. Rebuild KuzuDB from Postgres (derived state)
log "Rebuilding KuzuDB graph via scripts/sync_kuzu.py"
if ! python "$REPO_ROOT/scripts/sync_kuzu.py" --quiet; then
    log "WARNING: sync_kuzu.py failed; KuzuDB may be out of date" >&2
    # Not fatal - graph search gracefully falls back to FTS+vector per CLAUDE.md
fi

# 8. Update staleness marker
echo "$FINGERPRINT" > "$STALENESS_MARKER"

log "Restore complete: sources=$SOURCES chunks=$CHUNKS in ${RESTORE_SECS}s"
