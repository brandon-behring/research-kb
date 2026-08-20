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

## Cross-machine sync (Linux authoritative -> Mac replica)

The database itself is not in git (`*.sql` and `*.dump` are gitignored). To
keep a Mac replica in sync, nightly cron on the Linux workstation dumps the
DB in compressed custom format and uploads it to Google Drive. The Mac's
Google Drive app auto-downloads the file in the background; the user runs
`restore_db_from_cloud.sh` on the Mac when a fresh snapshot is desired.

### Architecture

```
Linux (authoritative)                       Mac (replica)
─────────────────────                       ─────────────
nightly 03:17 cron                          manual restore
     │                                          ▲
     ▼                                          │
pg_dump -Fc -Z 9 -> research_kb_latest.dump     │
     │                                          │
     └── rclone copy --> gdrive:/research-kb-db/ -> GDrive app auto-downloads
                                                        │
                                                        ▼
                                              pg_restore -c -j 4
                                                        │
                                                        ▼
                                              sync_kuzu.py (rebuild graph)
```

### On Linux (sync side)

Install nightly cron once:
```bash
( crontab -l 2>/dev/null; echo "17 3 * * * cd $HOME/Claude/research-kb && ./scripts/sync_db_to_cloud.sh >> backups/sync.log 2>&1" ) | crontab -
```

Manual operations:
```bash
./scripts/sync_db_to_cloud.sh              # dump + upload
./scripts/sync_db_to_cloud.sh --dump-only  # dump only, skip upload
tail -50 backups/sync.log                  # inspect cron output
cat backups/.last_sync_ok                  # last successful sync timestamp
cat backups/.last_sync_FAILED              # error detail if sync failed
```

Failure markers: `backups/.last_sync_FAILED` is written on any error (with
timestamp + reason) and cleared on the next successful sync.

### On Mac (first-time setup)

1. Install prerequisites:
   ```bash
   brew install --cask docker
   brew install --cask google-drive
   brew install postgresql@16 python@3.11 uv git rclone
   ```
2. Clone the repo and install deps:
   ```bash
   mkdir -p ~/Claude && cd ~/Claude
   git clone git@github.com:brandon-behring/research-kb.git
   cd research-kb && uv sync --all-packages
   ```
3. Start Postgres:
   ```bash
   docker compose up -d postgres
   docker compose ps    # wait for "healthy"
   ```
4. Sign in to the Google Drive desktop app (same account as Linux). Enable
   **Mirror mode** for `research-kb-db/` so the dump lives on local disk
   (not streamed), which makes restores instant.
5. Locate the mirrored file and set the env var in `~/.zshrc`:
   ```bash
   find ~/Library/CloudStorage -name research_kb_latest.dump 2>/dev/null
   # Copy the result and add to ~/.zshrc:
   export RESEARCH_KB_DUMP_PATH="$HOME/Library/CloudStorage/GoogleDrive-<email>/My Drive/research-kb-db/research_kb_latest.dump"
   source ~/.zshrc
   ```
6. First restore:
   ```bash
   cd ~/Claude/research-kb
   ./scripts/restore_db_from_cloud.sh    # 5-10 min on typical Mac
   research-kb sources stats             # should match Linux counts
   ```

### Mac ongoing usage

- Google Drive app keeps `research_kb_latest.dump` auto-synced in the background
- When you want fresh data on Mac: `./scripts/restore_db_from_cloud.sh`
- The script is idempotent and staleness-aware — it skips work if already
  current. Use `--force` to re-restore anyway
- `pg_restore -c` drops and recreates objects, so don't run during active queries

### Archived old backups (cold storage)

Historical plain-SQL backups older than the most recent pre-extraction dump
live on the external drive at
`/run/media/brandon_behring/backup/research-kb-backups/`. The
`scripts/archive_backups.sh` script moves old dumps there, keeping only the
compressed sync dump and the most recent pre-extraction SQL on system disk.
Re-run it as rotation whenever `backups/` grows.

```bash
./scripts/archive_backups.sh --dry-run   # preview what will move
./scripts/archive_backups.sh             # execute after confirmation
```

---

## Emergency Contacts

If you encounter a scenario not covered here:

1. Check git history for recent changes: `git log --oneline -10`
2. Check docker logs: `docker compose logs postgres`
3. Check if backups directory is intact: `ls -la backups/`
4. Check archived backups on external drive: `ls -la /run/media/brandon_behring/backup/research-kb-backups/`
