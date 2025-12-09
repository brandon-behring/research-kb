#!/bin/bash
# Safe docker compose wrapper - intercepts destructive operations
# Covers: down -v, down --volumes, rm -v, volume rm, system prune
#
# Usage: ./scripts/docker-safe.sh [docker compose args]
# Recommended alias: alias dc='./scripts/docker-safe.sh'

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for warnings
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

warn_and_confirm() {
    echo -e "${RED}⚠️  WARNING: This operation will DELETE ALL DATA${NC}"
    echo ""

    # Try to show current data counts
    if docker exec research-kb-postgres psql -U postgres -d research_kb -t -c "SELECT 1" &>/dev/null; then
        echo "Current database contents:"
        docker exec research-kb-postgres psql -U postgres -d research_kb -t -c \
            "SELECT '  concepts: ' || COUNT(*) FROM concepts UNION ALL
             SELECT '  chunks: ' || COUNT(*) FROM chunks UNION ALL
             SELECT '  relationships: ' || COUNT(*) FROM concept_relationships" 2>/dev/null
    else
        echo -e "${YELLOW}  (database not running - cannot show counts)${NC}"
    fi

    echo ""
    read -p "Create backup first? [Y/n] " answer
    if [[ "$answer" != "n" && "$answer" != "N" ]]; then
        echo "Creating backup..."
        "$SCRIPT_DIR/backup_db.sh" || {
            echo -e "${RED}Backup failed! Aborting.${NC}"
            exit 1
        }
    fi

    echo ""
    echo -e "${RED}This will permanently delete all data in the research-kb database.${NC}"
    read -p "Type 'DELETE' to confirm: " confirm
    if [[ "$confirm" != "DELETE" ]]; then
        echo "Aborted."
        exit 1
    fi

    echo ""
    echo "Proceeding with destructive operation..."
}

# Check for destructive patterns in arguments
case "$*" in
    *"down -v"*|*"down --volumes"*|*"rm -v"*|*"volume rm"*|*"system prune"*)
        warn_and_confirm
        ;;
esac

# Execute docker compose with all arguments
cd "$PROJECT_DIR"
docker compose "$@"
