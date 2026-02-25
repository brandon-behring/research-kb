# Local Development Guide

---

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- ~10GB disk space (for models and database)
- NVIDIA GPU (optional, for faster embeddings)

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/brandonmbehring-dev/research-kb.git
cd research-kb
```

### 2. Start Services

```bash
# PostgreSQL + GROBID
docker-compose up -d

# Wait for PostgreSQL health check
docker-compose logs -f postgres
```

### 3. Install Packages

```bash
# Recommended: uv (single command, workspace-aware)
uv sync

# Fallback: pip (manual editable installs)
python -m venv .venv
source .venv/bin/activate
make setup-pip
```

### 4. Start Embedding Server

```bash
# Start in background
python -m research_kb_pdf.embed_server &

# Test connection
python -c "
from research_kb_pdf import EmbeddingClient
client = EmbeddingClient()
print(client.ping())
"
```

### 5. Initialize Database

```bash
# Apply schema
PGPASSWORD=postgres psql -h localhost -U postgres -d research_kb < packages/storage/schema.sql

# Apply migrations
for f in packages/storage/migrations/*.sql; do
    PGPASSWORD=postgres psql -h localhost -U postgres -d research_kb < "$f"
done
```

### 6. Verify Installation

```bash
# Run tests
pytest packages/cli/tests/ -v

# Test CLI
research-kb sources stats
```

---

## Common Tasks

### Ingest PDFs

```bash
# Single PDF
python scripts/ingest_corpus.py --file path/to/paper.pdf

# Corpus (all PDFs in fixtures/papers/)
python scripts/ingest_corpus.py
```

### Run Search

```bash
# Basic search
research-kb search query "instrumental variables"

# With graph boost
research-kb search query "instrumental variables" --context building

# Without graph
research-kb search query "test" --no-graph
```

### Extract Concepts (Requires Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.1:8b

# Run extraction
python scripts/extract_concepts.py --limit 100
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_HOST` | localhost | Database host |
| `POSTGRES_PORT` | 5432 | Database port |
| `POSTGRES_DB` | research_kb | Database name |
| `POSTGRES_USER` | postgres | Database user |
| `POSTGRES_PASSWORD` | postgres | Database password |
| `OLLAMA_HOST` | http://localhost:11434 | Ollama API endpoint |

---

## Troubleshooting

### Embedding Server Won't Start

```bash
# Check if socket exists
ls -la /tmp/research_kb_embed.sock

# Remove stale socket
rm -f /tmp/research_kb_embed.sock

# Restart server
python -m research_kb_pdf.embed_server
```

### Database Connection Failed

```bash
# Check PostgreSQL is running
docker-compose ps

# Check logs
docker-compose logs postgres

# Restart
docker-compose restart postgres
```

### Tests Failing

```bash
# Run with verbose output
pytest packages/storage/tests/ -v --tb=long

# Check database connection
PGPASSWORD=postgres psql -h localhost -U postgres -d research_kb -c "SELECT 1"
```

---

## Development Workflow

1. **Create branch**: `git checkout -b feature/my-feature`
2. **Make changes**
3. **Run tests**: `pytest packages/*/tests/ -v`
4. **Format code**: `black packages/`
5. **Lint**: `ruff check packages/`
6. **Commit**: Follow conventional commits
7. **Push**: `git push origin feature/my-feature`

---

## Useful Commands

```bash
# Database stats
research-kb sources stats

# List sources
research-kb sources list

# Concept search
research-kb graph concepts "IV"

# Graph exploration
research-kb graph neighborhood "double machine learning" --hops 2

# Extraction status
research-kb sources extraction-status
```

---

## IDE Setup

### VS Code

Recommended extensions:
- Python
- Pylance
- Black Formatter
- Ruff

`.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

---

## Links

- [CLAUDE.md](../../CLAUDE.md) - Full command reference
- [System Design](../SYSTEM_DESIGN.md) - Architecture overview
