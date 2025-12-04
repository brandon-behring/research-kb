# Research KB CLI

Command-line interface for querying the research knowledge base.

## Installation

From the repository root:

```bash
pip install -e packages/cli
```

## Commands

### `research-kb query`

Search the knowledge base using hybrid retrieval (full-text + vector similarity).

**Usage:**
```bash
research-kb query "backdoor criterion" [OPTIONS]
```

**Options:**
- `--limit, -l` - Maximum number of results (default: 5)
- `--format, -f` - Output format: `markdown`, `json`, `agent` (default: markdown)
- `--context-type, -c` - Search mode: `building`, `auditing`, `balanced` (default: balanced)
- `--source-type, -s` - Filter by source type: `paper`, `textbook`
- `--no-content` - Hide content snippets in markdown output

**Examples:**
```bash
# Basic query
research-kb query "instrumental variables"

# Agent-friendly format with 10 results
research-kb query "difference-in-differences" --limit 10 --format agent

# Precise term matching for auditing
research-kb query "cross-fitting" --context-type auditing --source-type paper
```

---

### `research-kb concepts`

Search for concepts in the knowledge graph.

**Usage:**
```bash
research-kb concepts "instrumental variables" [OPTIONS]
```

**Options:**
- `--limit, -l` - Maximum number of results (default: 10)
- `--relationships` / `--no-relationships` - Show related concepts (default: show)

**Examples:**
```bash
# Find concept by name
research-kb concepts "instrumental variables"

# Search by alias
research-kb concepts "IV" --limit 5

# Hide relationships
research-kb concepts "matching" --no-relationships
```

---

### `research-kb graph`

Visualize concept neighborhood in the knowledge graph.

**Usage:**
```bash
research-kb graph "instrumental variables" [OPTIONS]
```

**Options:**
- `--hops, -h` - Number of hops to traverse: 1-3 (default: 1)
- `--type, -t` - Filter by relationship type: `REQUIRES`, `USES`, `ADDRESSES`, etc.

**Examples:**
```bash
# Show 1-hop neighborhood
research-kb graph "instrumental variables"

# Show 2-hop neighborhood
research-kb graph "IV" --hops 2

# Filter by relationship type
research-kb graph "matching" --type REQUIRES --hops 1
```

---

### `research-kb path`

Find shortest path between two concepts in the knowledge graph.

**Usage:**
```bash
research-kb path "start concept" "end concept" [OPTIONS]
```

**Options:**
- `--max-hops, -m` - Maximum path length to search (default: 5)

**Examples:**
```bash
# Find path between concepts
research-kb path "double machine learning" "k-fold cross-validation"

# Search with shorter max hops
research-kb path "IV" "endogeneity" --max-hops 3

# Check if concepts are connected
research-kb path "matching" "propensity score" --max-hops 2
```

---

### `research-kb extraction-status`

Show extraction pipeline statistics.

**Usage:**
```bash
research-kb extraction-status
```

Displays:
- Total extracted concepts by type
- Total relationships by type
- Concept validation status
- Extraction quality metrics (confidence distribution)
- Chunk coverage

**Example:**
```bash
research-kb extraction-status
```

---

### `research-kb sources`

List all ingested sources in the knowledge base.

**Usage:**
```bash
research-kb sources
```

**Example:**
```bash
research-kb sources
```

---

### `research-kb stats`

Show knowledge base statistics.

**Usage:**
```bash
research-kb stats
```

Displays:
- Total sources and chunks
- Sources by type (paper, textbook)

**Example:**
```bash
research-kb stats
```

---

## Context Types

The `query` command supports different context modes:

### Building Context
**Use case**: Initial research, broad exploration
**Behavior**: Favors semantic similarity for breadth
**Weights**: 20% FTS, 80% vector

```bash
research-kb query "matching methods" --context-type building
```

### Auditing Context
**Use case**: Verification, precise citations
**Behavior**: Favors exact term matching for precision
**Weights**: 50% FTS, 50% vector

```bash
research-kb query "Theorem 3.1" --context-type auditing
```

### Balanced Context (Default)
**Use case**: General research
**Behavior**: Balanced approach
**Weights**: 30% FTS, 70% vector

```bash
research-kb query "backdoor criterion"  # Uses balanced by default
```

---

## Output Formats

### Markdown (Default)
Human-readable format with provenance and content snippets.

```bash
research-kb query "IV" --format markdown
```

### JSON
Machine-parseable format for programmatic use.

```bash
research-kb query "IV" --format json > results.json
```

### Agent
Optimized format for AI agent consumption with structured metadata.

```bash
research-kb query "IV" --format agent
```

---

## Graph Query Concepts

### Relationships
The knowledge graph supports several relationship types:

- **REQUIRES** - Method requires assumption (e.g., IV → relevance)
- **USES** - Method uses technique (e.g., matching → propensity scores)
- **ADDRESSES** - Method solves problem (e.g., IV → endogeneity)
- **GENERALIZES** - Broader concept (e.g., panel → DiD)
- **SPECIALIZES** - Narrower concept (e.g., LATE → treatment effect)
- **ALTERNATIVE_TO** - Competing approaches (e.g., matching vs regression)
- **EXTENDS** - Builds upon (e.g., DML → ML + CI)

### Graph Traversal
All graph commands traverse directed edges from source to target. Use different starting points to explore different perspectives.

**Example:** To see what methods require a specific assumption:
```bash
research-kb graph "unconfoundedness" --hops 1
```

---

## Tips

1. **Fuzzy matching**: Concept search supports partial matching and aliases
   ```bash
   research-kb concepts "DiD"  # Finds "difference-in-differences"
   ```

2. **Relationship exploration**: Start from problems to find solving methods
   ```bash
   research-kb graph "endogeneity" --hops 1  # Shows IV and other solutions
   ```

3. **Path finding**: Discover conceptual connections
   ```bash
   research-kb path "double ML" "cross-fitting"  # Shows how concepts relate
   ```

4. **Quality monitoring**: Check extraction pipeline health
   ```bash
   research-kb extraction-status  # View confidence scores and coverage
   ```

---

## Master Plan Reference

**Phase 2 Step 7**: CLI Extensions (lines 616-673)
**Query command**: Lines 588-590
**Graph operations**: Knowledge graph traversal requirements
