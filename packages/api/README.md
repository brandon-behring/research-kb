# Research KB API

FastAPI REST API for the research knowledge base. Provides semantic search, concept exploration, graph traversal, and source management.

## Quick Start

```bash
pip install -e packages/api

# Start the API server
uvicorn research_kb_api.main:create_app --factory --reload --port 8000

# Swagger docs at http://localhost:8000/docs
```

## Routers

| Router | Prefix | Description |
|--------|--------|-------------|
| **Health** | `/health` | Liveness and readiness checks |
| **Search** | `/search` | Hybrid search (FTS + vector + graph + citation) |
| **Sources** | `/sources` | Source management (list, get, metadata) |
| **Concepts** | `/concepts` | Concept search and details |
| **Graph** | `/graph` | Graph traversal and path finding |
| **Domains** | `/domains` | Knowledge domain listing |

## Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Search
curl "http://localhost:8000/search?q=instrumental+variables&limit=5"

# List sources
curl http://localhost:8000/sources

# Get concept details
curl http://localhost:8000/concepts/search?q=double+machine+learning

# Graph path between concepts
curl "http://localhost:8000/graph/path?source=IV&target=unconfoundedness"

# List domains
curl http://localhost:8000/domains
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://research_kb:...` | PostgreSQL connection string |
| `EMBED_SOCKET_PATH` | `/tmp/research_kb_embed.sock` | Embedding server socket |

## Architecture

- **FastAPI** with async handlers
- **asyncpg** connection pool (shared via `app.state.pool`)
- **Lifespan** context manages pool startup/shutdown
- **CORS** enabled for cross-origin access
- Swagger UI at `/docs`, ReDoc at `/redoc`
