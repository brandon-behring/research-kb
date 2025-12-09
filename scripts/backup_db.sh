#!/bin/bash
# Backup PostgreSQL database to timestamped file
#
# Usage:
#   ./scripts/backup_db.sh           # Creates backups/research_kb_YYYYMMDD_HHMMSS.sql
#   ./scripts/backup_db.sh latest    # Creates backups/research_kb_latest.sql (for CI cache)
#
# Restore:
#   docker exec -i research-kb-postgres psql -U postgres -d research_kb < backups/research_kb_latest.sql

set -e

BACKUP_DIR="$(dirname "$0")/../backups"
mkdir -p "$BACKUP_DIR"

if [ "$1" = "latest" ]; then
    BACKUP_FILE="$BACKUP_DIR/research_kb_latest.sql"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/research_kb_$TIMESTAMP.sql"
fi

echo "Creating backup: $BACKUP_FILE"

docker exec research-kb-postgres pg_dump -U postgres research_kb > "$BACKUP_FILE"

# Get stats
CHUNKS=$(docker exec research-kb-postgres psql -U postgres -d research_kb -t -c "SELECT COUNT(*) FROM chunks;" | tr -d ' ')
CONCEPTS=$(docker exec research-kb-postgres psql -U postgres -d research_kb -t -c "SELECT COUNT(*) FROM concepts;" | tr -d ' ')
RELS=$(docker exec research-kb-postgres psql -U postgres -d research_kb -t -c "SELECT COUNT(*) FROM concept_relationships;" | tr -d ' ')

echo "Backup complete!"
echo "  Chunks: $CHUNKS"
echo "  Concepts: $CONCEPTS"
echo "  Relationships: $RELS"
echo "  File: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"

# Keep only last 5 timestamped backups
ls -t "$BACKUP_DIR"/research_kb_[0-9]*.sql 2>/dev/null | tail -n +6 | xargs -r rm -f
