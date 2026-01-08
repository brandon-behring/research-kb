#!/bin/bash
# Test backup and restore functionality
#
# This script validates that:
# 1. backup_db.sh creates valid backups
# 2. Backups can be restored successfully
# 3. Data integrity is preserved after restore
# 4. Exit codes work correctly
#
# WARNING: This script will DROP and RECREATE the database!
# Only run this in development/CI environments.
#
# Usage:
#   ./scripts/test_backup_restore.sh
#   ./scripts/test_backup_restore.sh --skip-restore  # Only test backup creation
#
# Exit codes:
#   0 - All tests passed
#   1 - Backup creation failed
#   2 - Restore failed
#   3 - Data integrity check failed

set -e

SCRIPT_DIR="$(dirname "$0")"
BACKUP_DIR="$SCRIPT_DIR/../backups"
TEST_BACKUP="$BACKUP_DIR/test_backup_restore.sql"

# Parse arguments
SKIP_RESTORE=false
for arg in "$@"; do
    case "$arg" in
        --skip-restore)
            SKIP_RESTORE=true
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check database is running
check_database() {
    if ! docker exec research-kb-postgres psql -U postgres -d research_kb -c "SELECT 1" &>/dev/null; then
        log_error "Database not running or not accessible"
        exit 1
    fi
}

# Get table counts
get_counts() {
    local prefix=$1
    eval "${prefix}_SOURCES=$(docker exec research-kb-postgres psql -U postgres -d research_kb -t -c 'SELECT COUNT(*) FROM sources;' | tr -d ' ')"
    eval "${prefix}_CHUNKS=$(docker exec research-kb-postgres psql -U postgres -d research_kb -t -c 'SELECT COUNT(*) FROM chunks;' | tr -d ' ')"
    eval "${prefix}_CONCEPTS=$(docker exec research-kb-postgres psql -U postgres -d research_kb -t -c 'SELECT COUNT(*) FROM concepts;' | tr -d ' ')"
    eval "${prefix}_RELS=$(docker exec research-kb-postgres psql -U postgres -d research_kb -t -c 'SELECT COUNT(*) FROM concept_relationships;' | tr -d ' ')"
}

cleanup() {
    if [ -f "$TEST_BACKUP" ]; then
        rm -f "$TEST_BACKUP"
        log_info "Cleaned up test backup file"
    fi
}

trap cleanup EXIT

echo "======================================"
echo "  Backup/Restore Test Suite"
echo "======================================"
echo

# Test 1: Database accessibility
log_info "Test 1: Checking database accessibility..."
check_database
echo "  Database is accessible"

# Test 2: Get baseline counts
log_info "Test 2: Recording baseline counts..."
get_counts "BEFORE"
echo "  Sources: $BEFORE_SOURCES"
echo "  Chunks: $BEFORE_CHUNKS"
echo "  Concepts: $BEFORE_CONCEPTS"
echo "  Relationships: $BEFORE_RELS"

if [ "$BEFORE_SOURCES" -eq 0 ]; then
    log_warn "Database appears to be empty. Skipping integrity tests."
    SKIP_RESTORE=true
fi

# Test 3: Create backup
log_info "Test 3: Creating test backup..."
docker exec research-kb-postgres pg_dump -U postgres research_kb > "$TEST_BACKUP"

if [ ! -f "$TEST_BACKUP" ]; then
    log_error "Backup file was not created"
    exit 1
fi

BACKUP_SIZE=$(stat -c%s "$TEST_BACKUP" 2>/dev/null || stat -f%z "$TEST_BACKUP" 2>/dev/null)
if [ "$BACKUP_SIZE" -lt 1000 ]; then
    log_error "Backup file is too small ($BACKUP_SIZE bytes)"
    exit 1
fi
echo "  Backup created: $(du -h "$TEST_BACKUP" | cut -f1)"

# Test 4: Verify backup_db.sh exit codes
log_info "Test 4: Testing backup_db.sh exit codes..."

# Test successful backup (should return 0)
if "$SCRIPT_DIR/backup_db.sh" --path-only > /dev/null 2>&1; then
    echo "  Exit code 0 (success): PASS"
else
    log_error "backup_db.sh returned non-zero for valid backup"
    exit 1
fi

# Test 5: Verify retention policy (optional - just check command runs)
log_info "Test 5: Checking backup retention..."
TIMESTAMPED_COUNT=$(ls -1 "$BACKUP_DIR"/research_kb_[0-9]*.sql 2>/dev/null | wc -l)
PRE_EXTRACTION_COUNT=$(ls -1 "$BACKUP_DIR"/pre_extraction_[0-9]*.sql 2>/dev/null | wc -l)
echo "  Timestamped backups: $TIMESTAMPED_COUNT (max 5)"
echo "  Pre-extraction backups: $PRE_EXTRACTION_COUNT (max 3)"

if [ "$SKIP_RESTORE" = true ]; then
    log_warn "Skipping restore test (--skip-restore or empty database)"
    echo
    echo "======================================"
    echo "  Backup Tests PASSED (restore skipped)"
    echo "======================================"
    exit 0
fi

# Test 6: Drop and recreate database
log_info "Test 6: Dropping and recreating database..."
log_warn "This will DELETE all data temporarily!"

# Create a fresh database
docker exec research-kb-postgres psql -U postgres -c "DROP DATABASE IF EXISTS research_kb_test;" 2>/dev/null || true
docker exec research-kb-postgres psql -U postgres -c "CREATE DATABASE research_kb_test;"

# Test 7: Restore backup to test database
log_info "Test 7: Restoring backup to test database..."
docker exec -i research-kb-postgres psql -U postgres -d research_kb_test < "$TEST_BACKUP"
echo "  Restore completed"

# Test 8: Verify restored counts
log_info "Test 8: Verifying data integrity..."

AFTER_SOURCES=$(docker exec research-kb-postgres psql -U postgres -d research_kb_test -t -c 'SELECT COUNT(*) FROM sources;' | tr -d ' ')
AFTER_CHUNKS=$(docker exec research-kb-postgres psql -U postgres -d research_kb_test -t -c 'SELECT COUNT(*) FROM chunks;' | tr -d ' ')
AFTER_CONCEPTS=$(docker exec research-kb-postgres psql -U postgres -d research_kb_test -t -c 'SELECT COUNT(*) FROM concepts;' | tr -d ' ')
AFTER_RELS=$(docker exec research-kb-postgres psql -U postgres -d research_kb_test -t -c 'SELECT COUNT(*) FROM concept_relationships;' | tr -d ' ')

echo "  Before → After:"
echo "    Sources:       $BEFORE_SOURCES → $AFTER_SOURCES"
echo "    Chunks:        $BEFORE_CHUNKS → $AFTER_CHUNKS"
echo "    Concepts:      $BEFORE_CONCEPTS → $AFTER_CONCEPTS"
echo "    Relationships: $BEFORE_RELS → $AFTER_RELS"

# Verify counts match
INTEGRITY_OK=true
if [ "$BEFORE_SOURCES" != "$AFTER_SOURCES" ]; then
    log_error "Sources count mismatch!"
    INTEGRITY_OK=false
fi
if [ "$BEFORE_CHUNKS" != "$AFTER_CHUNKS" ]; then
    log_error "Chunks count mismatch!"
    INTEGRITY_OK=false
fi
if [ "$BEFORE_CONCEPTS" != "$AFTER_CONCEPTS" ]; then
    log_error "Concepts count mismatch!"
    INTEGRITY_OK=false
fi
if [ "$BEFORE_RELS" != "$AFTER_RELS" ]; then
    log_error "Relationships count mismatch!"
    INTEGRITY_OK=false
fi

# Cleanup test database
log_info "Cleaning up test database..."
docker exec research-kb-postgres psql -U postgres -c "DROP DATABASE IF EXISTS research_kb_test;" 2>/dev/null || true

if [ "$INTEGRITY_OK" = false ]; then
    log_error "Data integrity check FAILED!"
    exit 3
fi

echo
echo "======================================"
echo "  All Tests PASSED"
echo "======================================"
echo "  - Backup creation: OK"
echo "  - Restore: OK"
echo "  - Data integrity: OK"
echo "======================================"
