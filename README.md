# Research Knowledge Base

A semantic search system for causal inference literature with graph-boosted retrieval.

## Features

- **Hybrid Search**: Combines full-text search (FTS), vector similarity, and knowledge graph signals
- **Graph-Boosted Ranking**: Leverages concept relationships for improved relevance (enabled by default)
- **Knowledge Graph**: Automatically extracted concepts and relationships from research papers
- **CLI Interface**: Simple command-line interface for querying the knowledge base
- **Multiple Output Formats**: Markdown, JSON, and agent-optimized formats

## Quick Start

### Installation

```bash
# Install all packages in development mode
pip install -e packages/cli
pip install -e packages/storage
pip install -e packages/pdf-tools
pip install -e packages/contracts
pip install -e packages/common
```

### Basic Usage

#### Search with Graph Boosting (Default)

Graph-boosted search is enabled by default and provides the best results:

```bash
research-kb query "instrumental variables"
```

This combines:
- **Full-text search** (keyword matching)
- **Vector similarity** (semantic matching)
- **Knowledge graph signals** (concept relationships)

#### Customize Graph Weight

Adjust how much the knowledge graph influences rankings:

```bash
research-kb query "backdoor criterion" --graph-weight 0.3
```

Default graph weight is 0.2 (20% influence).

#### Fallback to Non-Graph Search

If you prefer traditional FTS + vector search only:

```bash
research-kb query "double machine learning" --no-graph
```

#### Other Query Options

```bash
# Limit number of results
research-kb query "propensity score" --limit 10

# Filter by source type
research-kb query "matching" --source-type paper

# Adjust context type (affects FTS/vector weights)
research-kb query "cross-fitting" --context-type building

# JSON output
research-kb query "IV estimation" --format json

# Agent-optimized output
research-kb query "causal trees" --format agent
```

### Other CLI Commands

#### List Sources

```bash
research-kb sources
```

#### Database Statistics

```bash
research-kb stats
```

#### Concept Search

```bash
research-kb concepts "instrumental variables"
```

#### Knowledge Graph Exploration

```bash
# View concept neighborhood
research-kb graph "double machine learning" --hops 2

# Find path between concepts
research-kb path "instrumental variables" "exogeneity"
```

#### Extraction Status

```bash
research-kb extraction-status
```

## Graph-Boosted Search

### How It Works

Graph-boosted search (v2) enhances traditional hybrid search by incorporating knowledge graph signals:

1. **Extract query concepts**: Identify concepts mentioned in your query
2. **Base retrieval**: Get initial results using FTS + vector search
3. **Graph scoring**: Compute graph similarity between query concepts and document concepts
4. **Re-ranking**: Combine all three signals for final ranking

**Formula**: `score = fts_weight × fts + vector_weight × vector + graph_weight × graph`

### When to Use Graph Search

✅ **Use graph search (default) when:**
- You want the most relevant results
- You're searching for specific concepts or methods
- You care about conceptual relationships

🔄 **Use `--no-graph` when:**
- Concepts haven't been extracted yet
- You're debugging search issues
- You want to compare results

### Graceful Fallback

If graph search is requested but concepts haven't been extracted, the system automatically falls back to standard search with a warning:

```
Warning: Graph search requested but no concepts extracted.
Falling back to standard search (FTS + vector only).
To extract concepts: python scripts/extract_concepts.py
```

## Data Ingestion

### Ingest Corpus

```bash
# Ingest Phase 1 corpus (textbooks + papers)
python scripts/ingest_corpus.py
```

### Extract Concepts

```bash
# Extract concepts using Ollama (requires Ollama server)
python scripts/extract_concepts.py --limit 1000
```

### Validate Quality

```bash
# Validate retrieval quality
python scripts/eval_retrieval.py

# Validate concept extraction
python scripts/validate_seed_concepts.py

# Validate knowledge graph
python scripts/master_plan_validation.py
```

## Development

### Running Tests

```bash
# All tests
pytest

# CLI tests only
pytest packages/cli/tests/ -v

# Script tests only
pytest tests/scripts/ -v

# Skip slow/integration tests
pytest -m "not slow and not integration"
```

### CI/CD

The project uses a tiered CI/CD approach:

1. **PR Checks** (fast, <10 min): Unit tests + CLI tests with mocked services
2. **Daily Validation** (3 min): Quick quality checks using cached database
3. **Weekly Full Rebuild** (60 min): Complete from-scratch validation proving reproducibility

See `.github/workflows/` for workflow definitions.

## Architecture

### Packages

- **cli**: Command-line interface (Typer-based)
- **storage**: Database layer (PostgreSQL + pgvector)
- **pdf-tools**: PDF ingestion and embedding
- **contracts**: Shared data models (Pydantic)
- **common**: Shared utilities and logging
- **extraction**: Concept extraction (Ollama-based)

### Database Schema

- **sources**: Textbooks, papers, repositories
- **chunks**: Content units with embeddings
- **citations**: Extracted bibliographic references
- **concepts**: Extracted concepts with types
- **concept_relationships**: Graph edges (REQUIRES, USES, etc.)
- **chunk_concepts**: Links chunks to concepts
- **methods**: Method-specific attributes
- **assumptions**: Assumption-specific attributes

## Migration Guide (v1 → v2)

### Breaking Changes

**Graph search is now enabled by default.** If you were using the CLI before:

**Before (v1)**:
```bash
research-kb query "test"  # FTS + vector only
research-kb query "test" --use-graph  # Opt-in to graph
```

**After (v2)**:
```bash
research-kb query "test"  # FTS + vector + graph (default)
research-kb query "test" --no-graph  # Opt-out of graph
```

### Compatibility

- Old scripts using `--use-graph` will continue to work (flag still accepted)
- Default behavior change only affects interactive CLI usage
- Programmatic API unchanged

## Requirements

### Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- Ollama (for concept extraction, optional)

### External Services

- **Embedding server** (required for search): `python -m research_kb_pdf.embed_server`
- **Ollama server** (optional, for concept extraction): `ollama serve`

## License

[Add license information]

## Contributing

[Add contributing guidelines]

## Citation

[Add citation information]
