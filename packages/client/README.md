# Research-KB Client SDK

Python client for research-kb daemon and CLI.

## Installation

```bash
pip install -e packages/client
```

## Quick Start

```python
from research_kb_client import DaemonClient

client = DaemonClient()

# Check availability
if client.is_available():
    # Search
    results = client.search("instrumental variables", limit=5)
    for r in results.results:
        print(f"[{r.score:.2f}] {r.source.title}")
        print(f"  {r.chunk.content[:100]}...")

    # Health check
    health = client.health()
    print(f"Status: {health.status}")

    # Stats
    stats = client.stats()
    print(f"Sources: {stats.sources}, Chunks: {stats.chunks}")
```

## Convenience Functions

```python
from research_kb_client import search_or_none, get_methodology_context

# Graceful failure (returns None if unavailable)
results = search_or_none("walk-forward validation")
if results:
    for r in results.results:
        print(r.source.title)

# Formatted markdown context
context = get_methodology_context("double machine learning")
print(context)
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_KB_SOCKET_PATH` | `/tmp/research_kb_daemon_$USER.sock` | Daemon socket |
| `RESEARCH_KB_CLI_PATH` | `~/Claude/research-kb/venv/bin/research-kb` | CLI fallback |

## Models

All responses use typed Pydantic models:

- `SearchResponse` - Search results with scores
- `SearchResult` - Individual result with source, chunk, scores
- `SearchResultSource` - Source metadata (title, authors, year)
- `SearchResultChunk` - Text chunk (content, page, section)
- `HealthStatus` - Component health
- `StatsResponse` - Database counts

## Error Handling

```python
from research_kb_client import ResearchKBError, ConnectionError, TimeoutError

try:
    results = client.search("query")
except ConnectionError:
    print("Cannot connect to daemon or CLI")
except TimeoutError:
    print("Request timed out")
except ResearchKBError as e:
    print(f"Other error: {e}")
```

## Architecture

1. **Daemon-first**: Tries Unix socket (`<100ms` latency)
2. **CLI fallback**: Uses subprocess if daemon unavailable (`~8s`)
3. **Typed responses**: All data via Pydantic models

## Testing

```bash
# Unit tests (mocked)
pytest packages/client/tests/ -v

# Integration tests (requires live daemon)
pytest packages/client/tests/ -v -m integration
```
