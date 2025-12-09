# Recovery Guide

This document describes how to recover from data loss scenarios in the research-kb system.

## Quick Reference

| Scenario | Command |
|----------|---------|
| Restore from backup | `docker exec -i research-kb-postgres psql -U postgres -d research_kb < backups/research_kb_YYYYMMDD_HHMMSS.sql` |
| Resume extraction | `python scripts/extract_concepts.py --resume` |
| List backups | `ls -la backups/` |
| Check database status | `docker exec research-kb-postgres psql -U postgres -d research_kb -c "SELECT COUNT(*) FROM concepts;"` |

---

## Backup Locations

All backups are stored in the `backups/` directory:

```
backups/
├── research_kb_20251209_160006.sql     # Regular timestamped backups
├── research_kb_latest.sql              # CI/CD latest backup
└── pre_extraction_20251209_160122.sql  # Pre-extraction backups
```

**Retention policy:**
- Regular backups: Last 5 kept
- Pre-extraction backups: Last 3 kept
- Latest backup: Single file, overwritten

---

## Recovery Scenarios

### Scenario 1: Database Wiped (e.g., `docker compose down -v`)

**Symptoms:**
- All concepts/chunks/relationships count is 0
- Docker volume was deleted

**Recovery:**

1. Start the database:
   ```bash
   docker compose up -d postgres
   sleep 5
   ```

2. Find the most recent backup:
   ```bash
   ls -lt backups/*.sql | head -5
   ```

3. Restore from backup:
   ```bash
   docker exec -i research-kb-postgres psql -U postgres -d research_kb < backups/research_kb_YYYYMMDD_HHMMSS.sql
   ```

4. Verify restoration:
   ```bash
   docker exec research-kb-postgres psql -U postgres -d research_kb -c "
     SELECT 'concepts' as table_name, COUNT(*) FROM concepts
     UNION ALL SELECT 'chunks', COUNT(*) FROM chunks
     UNION ALL SELECT 'concept_relationships', COUNT(*) FROM concept_relationships;
   "
   ```

5. Create a fresh backup:
   ```bash
   ./scripts/backup_db.sh
   ```

---

### Scenario 2: Extraction Crashed Mid-Run

**Symptoms:**
- Extraction script terminated unexpectedly
- Partial data in database

**Recovery:**

1. Check checkpoint status:
   ```bash
   cat .extraction_checkpoint.json | python -m json.tool | head -20
   ```

2. Resume from checkpoint:
   ```bash
   python scripts/extract_concepts.py --resume
   ```

3. If checkpoint is corrupted, start fresh (safe since pre-extraction backup was created):
   ```bash
   python scripts/extract_concepts.py --clear-checkpoint
   python scripts/extract_concepts.py
   ```

---

### Scenario 3: Need to Rollback After Bad Extraction

**Symptoms:**
- Extraction completed but results are bad
- Want to restore to pre-extraction state

**Recovery:**

1. Find the pre-extraction backup:
   ```bash
   ls -lt backups/pre_extraction_*.sql | head -1
   ```

2. Truncate knowledge graph tables:
   ```bash
   docker exec -i research-kb-postgres psql -U postgres -d research_kb -c "
     TRUNCATE TABLE chunk_concepts CASCADE;
     TRUNCATE TABLE concept_relationships CASCADE;
     TRUNCATE TABLE methods CASCADE;
     TRUNCATE TABLE assumptions CASCADE;
     TRUNCATE TABLE concepts CASCADE;
   "
   ```

3. Restore concepts from pre-extraction backup:
   ```bash
   awk '/^COPY public.concepts /,/^\\\./' backups/pre_extraction_YYYYMMDD_HHMMSS.sql | \
     docker exec -i research-kb-postgres psql -U postgres -d research_kb

   awk '/^COPY public.concept_relationships /,/^\\\./' backups/pre_extraction_YYYYMMDD_HHMMSS.sql | \
     docker exec -i research-kb-postgres psql -U postgres -d research_kb
   ```

---

### Scenario 4: Retry Failed Chunks from DLQ

**Symptoms:**
- Some chunks failed during extraction
- DLQ directory has error files

**Recovery:**

1. Check DLQ contents:
   ```bash
   ls -la .dlq/extraction/ | wc -l  # Count failed chunks
   cat .dlq/extraction/*.json | head -50  # Sample errors
   ```

2. Common error patterns:
   - `credit balance is too low` → Add credits to Anthropic account
   - `connection refused` → Start Ollama server
   - `timeout` → Reduce batch size or increase timeout

3. After fixing the issue, retry failed chunks:
   ```bash
   # Get list of failed chunk IDs
   ls .dlq/extraction/ | sed 's/.json//' > failed_chunks.txt

   # Clear DLQ to allow retry
   rm .dlq/extraction/*.json

   # Re-run extraction (will process failed chunks)
   python scripts/extract_concepts.py --resume
   ```

---

## Prevention

### Use Safe Docker Wrapper

Instead of raw `docker compose`, use the safe wrapper:

```bash
# Add to your shell profile (~/.bashrc or ~/.zshrc)
alias dc='./scripts/docker-safe.sh'

# Usage
dc up -d      # Works normally
dc down       # Works normally
dc down -v    # Warns, requires backup confirmation, requires 'DELETE' confirmation
```

### Regular Backups

Backups are created automatically:
- **Before every extraction** (unless `--skip-backup` is used)
- **Manually** with `./scripts/backup_db.sh`

Set up scheduled backups (optional):
```bash
# Add to crontab (crontab -e)
0 */6 * * * /path/to/research-kb/scripts/backup_db.sh >> /var/log/research-kb-backup.log 2>&1
```

---

## Database Access

### Direct PostgreSQL Access

```bash
# Interactive psql session
docker exec -it research-kb-postgres psql -U postgres -d research_kb

# Single query
docker exec research-kb-postgres psql -U postgres -d research_kb -c "SELECT COUNT(*) FROM concepts;"
```

### Key Tables

| Table | Description |
|-------|-------------|
| `sources` | Ingested papers/textbooks |
| `chunks` | Text chunks with embeddings |
| `concepts` | Extracted concepts |
| `concept_relationships` | Relationships between concepts |
| `chunk_concepts` | Links chunks to concepts |
| `citations` | Extracted citations |

---

## Emergency Contacts

If you encounter a scenario not covered here:

1. Check git history for recent changes: `git log --oneline -10`
2. Check docker logs: `docker compose logs postgres`
3. Check if backups directory is intact: `ls -la backups/`
