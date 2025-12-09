#!/bin/bash
# Backup PostgreSQL database to timestamped file
#
# Usage:
#   ./scripts/backup_db.sh                  # Creates backups/research_kb_YYYYMMDD_HHMMSS.sql
#   ./scripts/backup_db.sh latest           # Creates backups/research_kb_latest.sql (for CI cache)
#   ./scripts/backup_db.sh --pre-extraction # Creates backups/pre_extraction_YYYYMMDD_HHMMSS.sql
#   ./scripts/backup_db.sh --path-only      # Only print backup path, no stats (for scripting)
#
# Restore:
#   docker exec -i research-kb-postgres psql -U postgres -d research_kb < backups/research_kb_latest.sql
#
# Exit codes:
#   0 - Success
#   1 - Database not running or dump failed
#   2 - Backup verification failed (file empty or missing)

set -e

BACKUP_DIR="$(dirname "$0")/../backups"
mkdir -p "$BACKUP_DIR"

# Parse arguments
PATH_ONLY=false
BACKUP_TYPE="timestamped"

for arg in "$@"; do
    case "$arg" in
        --path-only)
            PATH_ONLY=true
            ;;
        --pre-extraction)
            BACKUP_TYPE="pre-extraction"
            ;;
        latest)
            BACKUP_TYPE="latest"
            ;;
    esac
done

# Determine backup filename
case "$BACKUP_TYPE" in
    latest)
        BACKUP_FILE="$BACKUP_DIR/research_kb_latest.sql"
        ;;
    pre-extraction)
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_FILE="$BACKUP_DIR/pre_extraction_$TIMESTAMP.sql"
        ;;
    *)
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_FILE="$BACKUP_DIR/research_kb_$TIMESTAMP.sql"
        ;;
esac

# Check database is running
if ! docker exec research-kb-postgres psql -U postgres -d research_kb -c "SELECT 1" &>/dev/null; then
    echo "ERROR: Database not running or not accessible" >&2
    exit 1
fi

if [ "$PATH_ONLY" = false ]; then
    echo "Creating backup: $BACKUP_FILE"
fi

# Create backup
docker exec research-kb-postgres pg_dump -U postgres research_kb > "$BACKUP_FILE"

# Verify backup file exists and has content
if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file was not created" >&2
    exit 2
fi

BACKUP_SIZE=$(stat -c%s "$BACKUP_FILE" 2>/dev/null || stat -f%z "$BACKUP_FILE" 2>/dev/null)
if [ "$BACKUP_SIZE" -lt 1000 ]; then
    echo "ERROR: Backup file is too small ($BACKUP_SIZE bytes) - likely failed" >&2
    exit 2
fi

# Get stats
CHUNKS=$(docker exec research-kb-postgres psql -U postgres -d research_kb -t -c "SELECT COUNT(*) FROM chunks;" | tr -d ' ')
CONCEPTS=$(docker exec research-kb-postgres psql -U postgres -d research_kb -t -c "SELECT COUNT(*) FROM concepts;" | tr -d ' ')
RELS=$(docker exec research-kb-postgres psql -U postgres -d research_kb -t -c "SELECT COUNT(*) FROM concept_relationships;" | tr -d ' ')

if [ "$PATH_ONLY" = true ]; then
    # For scripting: just output the path
    echo "$BACKUP_FILE"
else
    echo "Backup complete!"
    echo "  Chunks: $CHUNKS"
    echo "  Concepts: $CONCEPTS"
    echo "  Relationships: $RELS"
    echo "  File: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"
fi

# Keep only last 5 timestamped backups (but keep pre_extraction backups separate)
ls -t "$BACKUP_DIR"/research_kb_[0-9]*.sql 2>/dev/null | tail -n +6 | xargs -r rm -f
ls -t "$BACKUP_DIR"/pre_extraction_[0-9]*.sql 2>/dev/null | tail -n +3 | xargs -r rm -f
