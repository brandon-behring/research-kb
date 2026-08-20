#!/bin/bash
# Archive old PostgreSQL backups to external drive, keeping local only the
# minimum set needed for day-to-day recovery.
#
# Keeps LOCAL in backups/:
#   - research_kb_latest.dump   (the compressed sync artifact, if present)
#   - Most recent pre_extraction_*.sql  (rollback invariant for extraction runs)
#
# Moves to EXTERNAL (/run/media/brandon_behring/backup/research-kb-backups/):
#   - All other research_kb_*.sql timestamped dumps
#   - All older pre_extraction_*.sql dumps
#
# Safe to re-run as ongoing rotation.
#
# Usage:
#   ./scripts/archive_backups.sh --dry-run   # show plan, no changes
#   ./scripts/archive_backups.sh             # execute after confirmation
#   ./scripts/archive_backups.sh --yes       # skip confirmation (for scripting)
#
# Exit codes:
#   0 - Success (or nothing to do)
#   1 - External drive not mounted / target dir not writable
#   2 - rsync transfer failed (nothing deleted on source)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="$REPO_ROOT/backups"
ARCHIVE_ROOT="/run/media/brandon_behring/backup/research-kb-backups"

DRY_RUN=false
ASSUME_YES=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --yes|-y) ASSUME_YES=true ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
    esac
done

log() {
    echo "[$(date -Iseconds)] $*"
}

# 1. Verify external drive is mounted and writable
if [ ! -d "$(dirname "$ARCHIVE_ROOT")" ]; then
    echo "ERROR: backup drive not mounted at $(dirname "$ARCHIVE_ROOT")" >&2
    exit 1
fi

if [ "$DRY_RUN" = false ]; then
    mkdir -p "$ARCHIVE_ROOT"
    if ! touch "$ARCHIVE_ROOT/.write_test" 2>/dev/null; then
        echo "ERROR: Cannot write to $ARCHIVE_ROOT" >&2
        exit 1
    fi
    rm -f "$ARCHIVE_ROOT/.write_test"
fi

# 2. Compute keep lists
KEEP_FILES=()

# Keep the compressed sync dump if present
if [ -f "$BACKUP_DIR/research_kb_latest.dump" ]; then
    KEEP_FILES+=("research_kb_latest.dump")
fi

# Keep the single most recent pre_extraction dump
LATEST_PRE_EXTRACTION=$(ls -t "$BACKUP_DIR"/pre_extraction_*.sql 2>/dev/null | head -1 || true)
if [ -n "$LATEST_PRE_EXTRACTION" ]; then
    KEEP_FILES+=("$(basename "$LATEST_PRE_EXTRACTION")")
fi

# Keep the "latest.sql" CI cache file if it exists (used by CI workflows)
if [ -f "$BACKUP_DIR/research_kb_latest.sql" ]; then
    KEEP_FILES+=("research_kb_latest.sql")
fi

# 3. Build move list (everything in backups/ except keep + hidden files + non-dumps)
MOVE_FILES=()
while IFS= read -r -d '' path; do
    name=$(basename "$path")
    case "$name" in
        .*|*.log) continue ;;
    esac
    keep=false
    for k in "${KEEP_FILES[@]}"; do
        [ "$name" = "$k" ] && keep=true && break
    done
    if [ "$keep" = false ]; then
        MOVE_FILES+=("$path")
    fi
done < <(find "$BACKUP_DIR" -maxdepth 1 -type f \( -name "*.sql" -o -name "*.dump" \) -print0)

if [ ${#MOVE_FILES[@]} -eq 0 ]; then
    log "Nothing to archive — backups/ is already minimal"
    exit 0
fi

# 4. Print plan
echo
echo "=== Archive Plan ==="
echo "Target: $ARCHIVE_ROOT"
echo
echo "Will KEEP in backups/:"
for k in "${KEEP_FILES[@]}"; do
    if [ -f "$BACKUP_DIR/$k" ]; then
        printf "  %-60s %s\n" "$k" "$(du -h "$BACKUP_DIR/$k" | cut -f1)"
    fi
done
echo
echo "Will MOVE to $ARCHIVE_ROOT:"
TOTAL_BYTES=0
for f in "${MOVE_FILES[@]}"; do
    size=$(du -h "$f" | cut -f1)
    bytes=$(stat -c %s "$f")
    TOTAL_BYTES=$((TOTAL_BYTES + bytes))
    printf "  %-60s %s\n" "$(basename "$f")" "$size"
done
echo
TOTAL_GB=$(awk "BEGIN{printf \"%.1f\", $TOTAL_BYTES/1024/1024/1024}")
echo "Total to move: ${TOTAL_GB} GB"
echo

if [ "$DRY_RUN" = true ]; then
    log "Dry run complete — no changes made"
    exit 0
fi

if [ "$ASSUME_YES" != true ]; then
    read -rp "Proceed with archive? [y/N] " answer
    case "$answer" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "Aborted." ; exit 0 ;;
    esac
fi

# 5. Execute move with rsync --remove-source-files (verifies before deleting)
log "Moving ${#MOVE_FILES[@]} files to $ARCHIVE_ROOT"
if ! rsync -av --remove-source-files --progress "${MOVE_FILES[@]}" "$ARCHIVE_ROOT/"; then
    echo "ERROR: rsync failed — source files untouched" >&2
    exit 2
fi

log "Archive complete"
echo
echo "Post-archive sizes:"
du -sh "$BACKUP_DIR" "$ARCHIVE_ROOT"
echo
df -h / | tail -1
