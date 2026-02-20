# Research KB Dashboard

Streamlit visualization dashboard for exploring the research knowledge base.

## Quick Start

```bash
pip install -e packages/dashboard

streamlit run packages/dashboard/src/research_kb_dashboard/app.py
# Opens at http://localhost:8501
```

## Pages

| Page | Description |
|------|-------------|
| **Search** | Hybrid search interface with FTS + vector + graph + citation scoring |
| **Citations** | Interactive citation network visualization (PyVis) |
| **Queue** | Extraction pipeline status and throughput estimates |
| **Concepts** | Concept graph exploration with neighborhood visualization |
| **Statistics** | Corpus metrics: domain distribution, citation authority, concept types |
| **Assumptions** | Interactive assumption auditing for causal methods |

## Features

- **Hybrid search**: Test all 4 search signals with adjustable weights
- **Graph visualization**: Interactive PyVis graphs for citations and concept neighborhoods
- **Domain filtering**: Filter by any of the 19+ knowledge domains
- **Real-time stats**: Live corpus metrics from PostgreSQL

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://research_kb:...` | PostgreSQL connection string |
| `EMBED_SOCKET_PATH` | `/tmp/research_kb_embed.sock` | Embedding server socket |
