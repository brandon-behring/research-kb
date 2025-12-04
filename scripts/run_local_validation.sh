#!/bin/bash
# Local validation script to test the full pipeline before CI runs
# This simulates the weekly full rebuild workflow locally

set -e  # Exit on error

echo "=========================================="
echo "Research KB - Local Validation Runner"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0
WARNINGS=0

function print_step() {
    echo ""
    echo "=========================================="
    echo "STEP: $1"
    echo "=========================================="
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED++))
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED++))
}

function print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((WARNINGS++))
}

# Check prerequisites
print_step "Checking Prerequisites"

# Check PostgreSQL
if command -v psql &> /dev/null; then
    print_success "PostgreSQL client installed"
else
    print_error "PostgreSQL client not found (install postgresql-client)"
fi

# Check Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python installed: $PYTHON_VERSION"
else
    print_error "Python not found"
fi

# Check if in venv
if [ -n "$VIRTUAL_ENV" ]; then
    print_success "Virtual environment active: $VIRTUAL_ENV"
else
    print_warning "No virtual environment active (consider activating)"
fi

# Check database connection
if [ -z "$POSTGRES_HOST" ]; then
    export POSTGRES_HOST=localhost
    print_warning "POSTGRES_HOST not set, using localhost"
fi

if [ -z "$POSTGRES_DB" ]; then
    export POSTGRES_DB=research_kb
    print_warning "POSTGRES_DB not set, using research_kb"
fi

if [ -z "$POSTGRES_USER" ]; then
    export POSTGRES_USER=postgres
    print_warning "POSTGRES_USER not set, using postgres"
fi

# Test database connection
print_step "Testing Database Connection"

if PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres -c '\l' &> /dev/null; then
    print_success "Database connection successful"
else
    print_error "Cannot connect to database"
    echo "Check POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD environment variables"
    exit 1
fi

# Check if embedding server is running
print_step "Checking Embedding Server"

if curl --unix-socket /tmp/research_kb_embed.sock http://localhost/health 2>/dev/null; then
    print_success "Embedding server is running"
else
    print_warning "Embedding server not running"
    echo "Start with: python -m research_kb_pdf.embed_server &"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Ollama (optional)
print_step "Checking Ollama (Optional)"

if command -v ollama &> /dev/null; then
    if ollama list &> /dev/null; then
        print_success "Ollama is available"
    else
        print_warning "Ollama installed but not running (ollama serve)"
    fi
else
    print_warning "Ollama not installed (concept extraction will be skipped)"
fi

# Drop and recreate database
print_step "Recreating Database (Clean Slate)"

echo "WARNING: This will DROP the database '$POSTGRES_DB' and recreate it."
echo "Continue? (y/n)"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres -c "DROP DATABASE IF EXISTS $POSTGRES_DB" || print_warning "Could not drop database (may not exist)"
PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres -c "CREATE DATABASE $POSTGRES_DB" || print_error "Could not create database"

print_success "Database recreated"

# Apply schema
print_step "Applying Database Schema"

if [ -f packages/storage/schema.sql ]; then
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB < packages/storage/schema.sql
    print_success "Schema applied"
else
    print_error "Schema file not found: packages/storage/schema.sql"
    exit 1
fi

# Apply migrations
print_step "Applying Database Migrations"

MIGRATION_COUNT=0
for migration in packages/storage/migrations/*.sql; do
    if [ -f "$migration" ]; then
        echo "Applying: $migration"
        PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB < "$migration"
        ((MIGRATION_COUNT++))
    fi
done

if [ $MIGRATION_COUNT -eq 0 ]; then
    print_warning "No migrations found"
else
    print_success "Applied $MIGRATION_COUNT migration(s)"
fi

# Ingest corpus
print_step "Ingesting Corpus (~500 chunks target)"

if [ -f scripts/ingest_corpus.py ]; then
    START_TIME=$(date +%s)

    if python scripts/ingest_corpus.py; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        print_success "Corpus ingested in ${DURATION}s"

        # Check chunk count
        CHUNK_COUNT=$(PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -t -c "SELECT COUNT(*) FROM chunks" | xargs)

        if [ $CHUNK_COUNT -ge 450 ]; then
            print_success "Chunk count: $CHUNK_COUNT (≥450 target)"
        else
            print_warning "Chunk count: $CHUNK_COUNT (<450 target)"
        fi
    else
        print_error "Corpus ingestion failed"
        exit 1
    fi
else
    print_error "ingest_corpus.py not found"
    exit 1
fi

# Validate retrieval
print_step "Validating Retrieval Quality"

if [ -f scripts/eval_retrieval.py ]; then
    if python scripts/eval_retrieval.py; then
        print_success "Retrieval validation passed"
    else
        print_warning "Retrieval validation completed with warnings"
    fi
else
    print_warning "eval_retrieval.py not found (skipping)"
fi

# Extract concepts (optional)
print_step "Extracting Concepts (Optional)"

if [ -f scripts/extract_concepts.py ] && command -v ollama &> /dev/null; then
    echo "Extract concepts with limit? (recommended for speed)"
    echo "Enter limit (e.g., 100, 500, 1000) or 'skip':"
    read -r limit

    if [[ "$limit" != "skip" && "$limit" =~ ^[0-9]+$ ]]; then
        START_TIME=$(date +%s)

        if timeout 18m python scripts/extract_concepts.py --limit $limit; then
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            print_success "Concept extraction completed in ${DURATION}s"

            # Check concept count
            CONCEPT_COUNT=$(PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -t -c "SELECT COUNT(*) FROM concepts" | xargs)

            if [ $CONCEPT_COUNT -ge 100 ]; then
                print_success "Concept count: $CONCEPT_COUNT (≥100 target)"
            else
                print_warning "Concept count: $CONCEPT_COUNT (<100 target)"
            fi
        else
            print_warning "Concept extraction timed out or failed"
        fi
    else
        print_warning "Skipping concept extraction"
    fi
else
    print_warning "Skipping concept extraction (Ollama not available or script not found)"
fi

# Validate seed concepts
print_step "Validating Seed Concepts"

CONCEPT_COUNT=$(PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -t -c "SELECT COUNT(*) FROM concepts" | xargs)

if [ $CONCEPT_COUNT -gt 0 ] && [ -f scripts/validate_seed_concepts.py ]; then
    if python scripts/validate_seed_concepts.py --output json > /tmp/seed_validation.json; then
        print_success "Seed concept validation passed"
    else
        print_warning "Seed concept validation completed with warnings"
    fi
else
    print_warning "Skipping seed concept validation (no concepts or script not found)"
fi

# Run CLI tests
print_step "Running CLI Tests"

if pytest packages/cli/tests/ -v --tb=short -q; then
    print_success "All CLI tests passed"
else
    print_error "Some CLI tests failed"
fi

# Run script tests
print_step "Running Script Tests"

if pytest tests/scripts/ -v --tb=short -q -m "not slow and not integration"; then
    print_success "All script tests passed"
else
    print_error "Some script tests failed"
fi

# Summary
print_step "Validation Summary"

echo ""
echo "Results:"
echo -e "${GREEN}✓ Passed: $PASSED${NC}"
echo -e "${YELLOW}⚠ Warnings: $WARNINGS${NC}"
echo -e "${RED}✗ Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "✓ VALIDATION SUCCESSFUL"
    echo "==========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review the results above"
    echo "2. Update docs/VALIDATION_TRACKER.md with metrics"
    echo "3. Trigger GitHub Actions workflow for official run"
    exit 0
else
    echo -e "${RED}=========================================="
    echo "✗ VALIDATION FAILED"
    echo "==========================================${NC}"
    echo ""
    echo "Fix the issues above before running CI workflow"
    exit 1
fi
