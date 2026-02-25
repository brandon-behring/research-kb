# CLI Reference

> This is the command-line interface reference for research-kb. For project overview, architecture, and quick start, see the [main README](../README.md).

---

## Installation

```bash
# Install all packages in development mode
pip install -e packages/cli
pip install -e packages/storage
pip install -e packages/pdf-tools
pip install -e packages/contracts
pip install -e packages/common
```

## Search

### Graph-Boosted Search (Default)

Graph-boosted search is enabled by default and provides the best results:

```bash
research-kb search query "instrumental variables"
```

This combines:
- **Full-text search** (keyword matching)
- **Vector similarity** (semantic matching)
- **Knowledge graph signals** (concept relationships)

### Customize Graph Weight

Adjust how much the knowledge graph influences rankings:

```bash
research-kb search query "backdoor criterion" --graph-weight 0.3
```

Default graph weight is 0.2 (20% influence).

### Fallback to Non-Graph Search

If you prefer traditional FTS + vector search only:

```bash
research-kb search query "double machine learning" --no-graph
```

### Other Query Options

```bash
# Limit number of results
research-kb search query "propensity score" --limit 10

# Filter by source type
research-kb search query "matching" --source-type paper

# Adjust context type (affects FTS/vector weights)
research-kb search query "cross-fitting" --context-type building

# JSON output
research-kb search query "IV estimation" --format json

# Agent-optimized output
research-kb search query "causal trees" --format agent
```

## Browsing

### List Sources

```bash
research-kb sources list
```

### Database Statistics

```bash
research-kb sources stats
```

### Concept Search

```bash
research-kb graph concepts "instrumental variables"
```

### Extraction Status

```bash
research-kb sources extraction-status
```

## Knowledge Graph

### View Concept Neighborhood

```bash
research-kb graph neighborhood "double machine learning" --hops 2
```

### Find Path Between Concepts

```bash
research-kb graph path "instrumental variables" "exogeneity"
```

## Citation Network

```bash
# List citations from a source
research-kb citations list <source>

# Find sources citing this one
research-kb citations cited-by <source>

# Find sources this one cites
research-kb citations cites <source>

# Corpus citation statistics
research-kb citations stats
```

## Assumption Auditing

The **North Star feature** — audit the statistical assumptions required by any method in the knowledge base.

### Basic Usage

```bash
research-kb search audit-assumptions "instrumental variables"
```

Returns a structured list of assumptions (e.g., exclusion restriction, relevance, monotonicity) with descriptions, testability indicators, and references to corpus sources.

### Options

```bash
# Graph-only (no LLM fallback, instant from cache)
research-kb search audit-assumptions "IV" --no-ollama

# JSON output for programmatic consumption
research-kb search audit-assumptions "double machine learning" --format json

# Combine both
research-kb search audit-assumptions "propensity score matching" --no-ollama --format json
```

### How It Works

1. Searches the `method_assumption_cache` for pre-computed assumptions (10 top methods cached)
2. Falls back to knowledge graph traversal (concept → REQUIRES → assumption edges)
3. Optionally queries LLM (Anthropic backend) for methods not yet in the cache
4. Returns assumptions ranked by importance with testability metadata

### Example Output

```
Method: Instrumental Variables (IV)

Assumptions:
  1. Exclusion Restriction — instrument affects outcome only through treatment
     Testability: Not directly testable (domain knowledge required)

  2. Relevance (First Stage) — instrument is correlated with treatment
     Testability: Testable (F-statistic > 10 rule of thumb)

  3. Independence — instrument is independent of confounders
     Testability: Partially testable (balance checks)

Sources: Angrist & Pischke (2009), Imbens & Rubin (2015)
```

## Semantic Scholar Discovery

```bash
# Search Semantic Scholar
research-kb discover search "double machine learning"

# Browse by topic
research-kb discover topics

# Find by author
research-kb discover author "Chernozhukov"

# Enrich corpus with S2 metadata
research-kb enrich citations

# Show enrichment status
research-kb enrich status
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

## Migration Guide (v1 -> v2)

### Breaking Changes

**Graph search is now enabled by default.** If you were using the CLI before:

**Before (v1)**:
```bash
research-kb query "test"  # FTS + vector only
research-kb query "test" --use-graph  # Opt-in to graph
```

**After (v2)**:
```bash
research-kb search query "test"  # FTS + vector + graph (default)
research-kb search query "test" --no-graph  # Opt-out of graph
```

### Compatibility

- Old scripts using `--use-graph` will continue to work (flag still accepted)
- Default behavior change only affects interactive CLI usage
- Programmatic API unchanged
