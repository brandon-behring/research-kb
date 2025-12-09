#!/bin/bash
# Create test database for pytest
# This script runs on container initialization after schema.sql
#
# IMPORTANT: Tests use research_kb_test database to protect production data.
# The test fixtures will REFUSE to connect to research_kb.

set -e

echo "=== Creating test database research_kb_test ==="

# Check if database exists
if psql -U "$POSTGRES_USER" -lqt | cut -d \| -f 1 | grep -qw research_kb_test; then
    echo "Test database research_kb_test already exists"
else
    echo "Creating database research_kb_test..."
    psql -U "$POSTGRES_USER" -c "CREATE DATABASE research_kb_test"
fi

# Apply base schema to test database
echo "Applying schema to research_kb_test..."
psql -U "$POSTGRES_USER" -d "research_kb_test" -f /docker-entrypoint-initdb.d/01_schema.sql

echo "=== Test database research_kb_test ready ==="
echo ""
echo "SAFETY REMINDER: pytest fixtures will REFUSE to connect to research_kb (production)."
echo "All tests automatically use research_kb_test."
