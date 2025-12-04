# Research Context Retrieval Skill

This skill teaches agents how to query the research knowledge base for relevant context.

## When to Query

### Before Implementation
- Understanding theoretical foundations
- Finding canonical definitions
- Checking established approaches

### During Implementation
- Verifying mathematical formulations
- Finding implementation details
- Cross-referencing methods

### After Implementation
- Auditing for correctness
- Finding additional citations
- Identifying related work

## Context Types

### Building Mode (`--context-type building`)
- **Weight**: 20% FTS, 80% vector
- **Use when**: Exploring a topic, gathering initial context
- **Characteristics**: Broader semantic matching, more diverse results

### Auditing Mode (`--context-type auditing`)
- **Weight**: 50% FTS, 50% vector
- **Use when**: Verifying specific claims, finding exact definitions
- **Characteristics**: Precise term matching, focused results

### Balanced Mode (`--context-type balanced`) [default]
- **Weight**: 30% FTS, 70% vector
- **Use when**: General research queries
- **Characteristics**: Good balance of precision and recall

## CLI Usage

### Basic Query
```bash
research-kb query "backdoor criterion" --limit 5
```

### Agent-Friendly Format
```bash
research-kb query "instrumental variables" --format agent
```

Output:
```
RESEARCH_KB_RESULTS
QUERY: instrumental variables
CONTEXT: balanced
COUNT: 5
---
[1] Mostly Harmless Econometrics: An Empiricist's Companion
    CITE: (Angrist 2009)
    TYPE: textbook | AUTH: canonical
    LOC: Chapter 4, pp. 113-147
    SCORE: 0.892

    The instrumental variables (IV) strategy...
```

### JSON for Programmatic Use
```bash
research-kb query "propensity scores" --format json
```

### Source Filtering
```bash
research-kb query "causal trees" --source-type paper
```

## Interpreting Authority Levels

| Level | Meaning | Example |
|-------|---------|---------|
| `canonical` | Foundational/definitive source | Pearl's Causality |
| `survey` | Comprehensive review | Athey/Imbens ML overview |
| `standard` | Peer-reviewed publication | arXiv papers |
| `frontier` | Recent/cutting-edge work | 2024 papers |

## Provenance Tracking

**Always cite sources properly:**

1. Note the author and year: `(Pearl 2009)`
2. Include page numbers when available: `p. 101`
3. Reference section for context: `Section 3.3`

Example provenance string:
```
According to Pearl (2009, Section 3.3, p. 101), the backdoor criterion...
```

## Handling Gaps

When results are insufficient:
1. Try broader/narrower query terms
2. Switch context types
3. Check if source type filter is too restrictive
4. Query may be outside knowledge base scope

## Python API Usage

```python
from research_kb_storage import SearchQuery, search_hybrid
from research_kb_pdf import EmbeddingClient

# Generate query embedding
embed_client = EmbeddingClient()
query_embedding = embed_client.embed("backdoor criterion")

# Execute search
query = SearchQuery(
    text="backdoor criterion",
    embedding=query_embedding,
    fts_weight=0.3,
    vector_weight=0.7,
    limit=10,
)
results = await search_hybrid(query)

# Process results
for result in results:
    print(f"{result.source.title} (p. {result.chunk.page_start})")
    print(f"  Score: {result.combined_score:.3f}")  # Higher = better
    print(f"  {result.chunk.content[:200]}...")
```

## Score Semantics

All scores follow "higher is better" semantics:

| Score | Range | Meaning |
|-------|-------|---------|
| `fts_score` | 0+ | PostgreSQL ts_rank (higher = more keyword matches) |
| `vector_score` | 0-1 | Cosine similarity (1 = identical, 0 = opposite) |
| `combined_score` | 0-1 | Weighted combination (higher = more relevant) |

Example interpretation:
- `vector_score = 0.95` → Highly similar to query embedding
- `vector_score = 0.50` → Moderately similar
- `vector_score = 0.10` → Low similarity

## Best Practices

1. **Start broad, then narrow**: Use building mode first, then auditing
2. **Cross-reference sources**: Verify claims across textbooks and papers
3. **Check authority**: Prefer canonical sources for definitions
4. **Include provenance**: Always cite with page numbers
5. **Update knowledge base**: Ingest new relevant papers

## Statistics Command

Check knowledge base coverage:
```bash
research-kb stats
```

Output:
```
Research KB Statistics
========================================
Total sources: 14
Total chunks:  3184

By source type:
  textbook         2
  paper           12
```

---

## Knowledge Graph Features

### Concept Retrieval

Search for concepts in the knowledge graph by name or alias:

```bash
research-kb concepts "instrumental variables"
```

Output:
```
Found 1 concept(s) matching 'instrumental variables':

[1] Instrumental Variables
    Type: method
    Category: identification
    Aliases: IV, IVs, instrument, instruments
    Confidence: 0.92
    Definition: Econometric technique using exogenous variables...
    Relationships (5):
      → REQUIRES (relevance)
      → REQUIRES (exclusion restriction)
      → ADDRESSES (endogeneity)
```

**Options:**
- `--limit N` - Maximum results
- `--no-relationships` - Hide related concepts

**Use cases:**
- Finding concept definitions
- Checking aliases and alternative names
- Understanding concept relationships
- Verifying concept exists in graph

### Graph Neighborhood Visualization

Explore concepts within N hops:

```bash
research-kb graph "instrumental variables" --hops 2
```

Output:
```
Graph neighborhood for: Instrumental Variables
Hops: 2
============================================================

CENTER: Instrumental Variables (method)

Connected concepts (6):
  [1] Relevance (assumption)
  [2] Exclusion Restriction (assumption)
  [3] Exogeneity (assumption)
  [4] Endogeneity (problem)
  [5] Two-Stage Least Squares (method)
  [6] Confounding (problem)

Relationships (8):
  Instrumental Variables -[REQUIRES]-> Relevance
  Instrumental Variables -[REQUIRES]-> Exclusion Restriction
  Instrumental Variables -[USES]-> Two-Stage Least Squares
  Instrumental Variables -[ADDRESSES]-> Endogeneity
  ...
```

**Options:**
- `--hops N` - Number of hops (1-3)
- `--type RELATIONSHIP_TYPE` - Filter by relationship type

**Use cases:**
- Understanding method requirements
- Finding related techniques
- Discovering assumptions
- Mapping conceptual landscape

**Relationship types:**
- `REQUIRES` - Method needs assumption
- `USES` - Method employs technique
- `ADDRESSES` - Method solves problem
- `GENERALIZES` / `SPECIALIZES` - Hierarchical
- `ALTERNATIVE_TO` - Competing approaches
- `EXTENDS` - Builds upon

### Path Finding

Find conceptual connections between concepts:

```bash
research-kb path "double machine learning" "k-fold cross-validation"
```

Output:
```
Path from 'Double Machine Learning' to 'K-Fold Cross-Validation':
============================================================

Path length: 2 hop(s)

START: Double Machine Learning (method)
  ↓ [USES]
  Cross-Fitting (method)
  ↓ [USES]
  K-Fold Cross-Validation (method)

END: K-Fold Cross-Validation
```

**Options:**
- `--max-hops N` - Maximum path length to search

**Use cases:**
- Understanding how concepts relate
- Finding conceptual bridges
- Tracing method evolution
- Discovering indirect connections

### Extraction Status

Monitor concept extraction pipeline:

```bash
research-kb extraction-status
```

Output:
```
Extraction Pipeline Status
============================================================

Total concepts extracted: 73
Validated concepts:       4 (5.5%)

Concepts by type:
  method             53
  assumption         10
  problem             6
  theorem             2
  definition          2

Total relationships: 18

Relationships by type:
  REQUIRES            8
  ADDRESSES           5
  USES                3
  GENERALIZES         2

Extraction Quality:
  Average confidence: 0.86

  Confidence distribution:
    High (>=0.9)             7
    Medium (0.7-0.9)        16

Chunk coverage: 4/3631 (0.1%)
```

**Use cases:**
- Checking extraction progress
- Monitoring quality metrics
- Identifying coverage gaps
- Validating graph completeness

---

## Hybrid Retrieval Strategy

### When to Use Each Method

**Text Search (`research-kb query`):**
- Finding specific passages or proofs
- Searching for exact terminology
- Broad exploratory research
- When you need content snippets

**Concept Lookup (`research-kb concepts`):**
- Finding definitions
- Checking aliases and synonyms
- Understanding relationships
- Quick concept verification

**Graph Exploration (`research-kb graph`):**
- Understanding method requirements
- Finding related techniques
- Mapping conceptual dependencies
- Discovering alternatives

**Path Finding (`research-kb path`):**
- Understanding connections
- Tracing method evolution
- Finding conceptual bridges
- Validating relationships

### Example Workflows

#### Understanding a New Method

```bash
# 1. Find the method concept
research-kb concepts "difference-in-differences"

# 2. Explore requirements and relationships
research-kb graph "difference-in-differences" --hops 1

# 3. Get detailed content from papers
research-kb query "difference-in-differences" --limit 5 --format markdown
```

#### Verifying Assumptions

```bash
# 1. Find required assumptions for method
research-kb graph "propensity score matching" --type REQUIRES

# 2. Look up each assumption
research-kb concepts "unconfoundedness"

# 3. Find how to test assumption
research-kb query "testing unconfoundedness" --context-type auditing
```

#### Finding Alternative Methods

```bash
# 1. Understand problem
research-kb concepts "endogeneity"

# 2. Find methods that address it
research-kb graph "endogeneity" --type ADDRESSES --hops 1

# 3. Compare alternatives
research-kb concepts "instrumental variables"
research-kb concepts "regression discontinuity"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No results (query) | Try broader query, check spelling |
| No results (concepts) | Try aliases or partial names |
| Low relevance scores | Query may be outside KB scope |
| No graph connections | Concepts may not be extracted yet |
| Empty neighborhood | Increase hops or check relationships exist |
| No path found | Concepts not connected or max hops too low |
| Embedding server error | Start with `python -m research_kb_pdf.embed_server` |
| Database error | Check PostgreSQL: `docker ps` |
