---
paths:
  - "docs/**"
---

# Documentation Standards

Run `python scripts/audit_docs.py` periodically to detect drift between code and documentation.

The documentation protocol table in CLAUDE.md defines what to update for each change type.

## Entry Point

**Start here**: [`docs/INDEX.md`](docs/INDEX.md) — Navigation hub with current status, phase overview, and links to all documentation.

## Doc Generation

For API endpoint changes, regenerate package docs:
```bash
python scripts/generate_package_docs.py
```
