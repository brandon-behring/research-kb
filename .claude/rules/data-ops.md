---
paths:
  - "docker-compose.yml"
  - "scripts/docker-safe.sh"
  - "scripts/backup*.sh"
---

# Data Operations Safety

## Backup Automation

- **Automatic**: Created before every extraction run (unless `--skip-backup`)
- **Manual**: `./scripts/backup_db.sh`
- **Location**: `backups/` directory (last 5 kept)

## Recovery

See [`docs/RECOVERY.md`](docs/RECOVERY.md) for detailed recovery procedures including:
- Full database restore from backup
- Partial re-ingestion of failed sources
- KuzuDB graph rebuild from PostgreSQL
