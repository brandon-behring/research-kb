# Contributing to research-kb

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ with pgvector extension
- Docker (for GROBID and PostgreSQL)

### Quick Start

```bash
# Clone and set up
git clone https://github.com/brandonmbehring-dev/research-kb.git
cd research-kb

# Option A: uv (recommended — single command, workspace-aware)
uv sync

# Option B: pip (fallback — manual editable installs)
python -m venv venv
source venv/bin/activate
make setup-pip

# Start services
docker compose up -d

# Run tests
pytest -m unit
```

## Code Style

- **Formatter**: Black with 100-character line width
- **Linter**: Ruff
- **Type checker**: mypy
- **Test framework**: pytest with pytest-asyncio

```bash
black packages/ --line-length 100
ruff check packages/
mypy packages/
```

## Testing

All code changes should include tests. We use real assertions — no stubs or placeholder tests.

```bash
# Unit tests (fast, no external services)
pytest -m unit

# Integration tests (needs PostgreSQL)
pytest -m integration

# Full suite
pytest
```

### Test Conventions

- Async tests use `pytest-asyncio` with `asyncio_mode = auto`
- Float comparisons use `pytest.approx(value, rel=1e-5)`
- Database tests truncate tables between runs (function-scoped fixtures)
- Mock fixtures: `mock_ollama`, `mock_embedding_client`

## Architecture

See [CLAUDE.md](CLAUDE.md) for the full architecture overview. Key principles:

1. **Package boundaries**: `storage` owns all database access. Other packages go through it.
2. **Async throughout**: All storage operations are async (`asyncpg`).
3. **JSONB extensibility**: Unknown fields go to `metadata` columns.
4. **Custom errors**: Use `IngestionError`, `StorageError`, `SearchError` from `research_kb_common`.

## Pull Request Process

1. **Branch from `main`**: Create a feature branch (`feat/description` or `fix/description`)
2. **Write tests**: Cover happy path + likely error cases + critical edge cases
3. **Run linting**: `black --check packages/ && ruff check packages/`
4. **Run tests**: `pytest -m unit`
5. **Open PR**: Include a summary, test plan, and checklist

### PR Checklist

- [ ] Tests pass locally
- [ ] Code formatted with Black
- [ ] No new linting warnings
- [ ] Updated relevant documentation (if API/CLI changed)
- [ ] Commit messages follow conventional format (see below)

### Commit Message Format

Use conventional prefixes:

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature or capability |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring (no behavior change) |
| `test:` | Adding or modifying tests |
| `docs:` | Documentation only |

Example: `feat: Add citation authority scoring to hybrid search`

Include a brief body (1-2 sentences) explaining *why* if the change is non-obvious.

## Adding a New Domain

research-kb supports multiple knowledge domains. To add a new one:

1. Add domain config to `packages/extraction/src/research_kb_extraction/domain_prompts.py`
2. Add domain prompt tests in `packages/extraction/tests/test_domain_prompts.py`
3. Register domain in `scripts/add_missing_domains.py`
4. Ingest domain-specific papers using `scripts/ingest_corpus.py --domain <name>`

## Adding an MCP Tool

1. Define the tool function in `packages/mcp-server/src/research_kb_mcp/tools/`
2. Register it in `packages/mcp-server/src/research_kb_mcp/server.py`
3. Add tests in `packages/mcp-server/tests/`
4. Document in CLAUDE.md MCP Server section

## Reporting Issues

Please use [GitHub Issues](https://github.com/brandonmbehring-dev/research-kb/issues) with:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, Docker status)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
