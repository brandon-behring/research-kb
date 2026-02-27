---
layout: default
title: Getting Started
---

# Getting Started

## Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Ollama (optional, for concept extraction)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/brandon-behring/research-kb.git
cd research-kb
```

### 2. Set up Python environment

```bash
# Recommended: uv (single command, workspace-aware)
uv sync

# Fallback: pip
python -m venv .venv
source .venv/bin/activate
make setup-pip
```

### 3. Start infrastructure

```bash
docker compose up -d   # PostgreSQL + GROBID
```

### 4. Set up demo corpus

```bash
# Downloads 25 open-access arXiv papers and ingests them
python scripts/setup_demo.py
```

### 5. Search

```bash
research-kb search query "instrumental variables"
research-kb sources stats
research-kb graph concepts "double machine learning"
```

## Optional Setup

### Concept Extraction (requires Ollama)

```bash
pip install -e packages/extraction

# Install Ollama and pull model
ollama pull llama3.1:8b

# Extract concepts
python scripts/extract_concepts.py --backend ollama --limit 500

# Sync to KuzuDB graph
python scripts/sync_kuzu.py
```

### MCP Server (Claude Code integration)

```bash
pip install -e packages/mcp-server

# Add to Claude Code config
cp .mcp.json.example .mcp.json
# Edit paths in .mcp.json to match your installation
```

### Dashboard

```bash
pip install -e packages/dashboard

# Start API server
uvicorn research_kb_api.main:create_app --factory --port 8000 &

# Start dashboard
streamlit run packages/dashboard/src/research_kb_dashboard/app.py
```

### Full Docker Demo

```bash
# Start everything (PostgreSQL + API + Dashboard)
docker compose --profile demo --profile api up -d

# Set up demo corpus
python scripts/setup_demo.py

# Dashboard at http://localhost:8501
# API docs at http://localhost:8000/docs
```

## Troubleshooting

See [TROUBLESHOOTING.md](https://github.com/brandon-behring/research-kb/blob/main/TROUBLESHOOTING.md) for common issues and solutions.
