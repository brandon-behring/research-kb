# Research KB MCP Server

MCP (Model Context Protocol) server exposing the research-kb causal inference knowledge base to Claude Code.

## Overview

This package provides 16 MCP tools for:
- **Hybrid search** combining full-text, vector similarity, knowledge graph, and citation signals
- **Source browsing** with citation network exploration
- **Concept discovery** via knowledge graph traversal
- **Assumption auditing** for method implementation guidance
- **Health monitoring** and corpus statistics

## Installation

```bash
# From research-kb root
pip install -e packages/mcp-server

# Or with Poetry
cd packages/mcp-server && poetry install
```

**Prerequisites:**
- PostgreSQL with pgvector (via docker-compose)
- Embedding server running (`embed_server.py`)
- Research-kb database populated

## Configuration

### Claude Code Integration

Add to your `.mcp.json` or Claude Code configuration:

```json
{
  "mcpServers": {
    "research-kb": {
      "command": "/path/to/research-kb/venv/bin/research-kb-mcp",
      "args": []
    }
  }
}
```

Or for system-wide installation:

```json
{
  "mcpServers": {
    "research-kb": {
      "command": "research-kb-mcp",
      "args": []
    }
  }
}
```

## Tools Reference

### Search

#### `research_kb_search`

Hybrid search combining FTS, vector similarity, knowledge graph, and citation authority.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Natural language or keyword query |
| `limit` | int | 10 | Maximum results (1-50) |
| `context_type` | str | "balanced" | Weight strategy: "building", "auditing", "balanced" |
| `use_graph` | bool | True | Include knowledge graph signals |
| `use_rerank` | bool | True | Apply cross-encoder reranking |
| `use_expand` | bool | True | Expand query with synonyms |
| `use_citations` | bool | True | Enable citation authority boosting |
| `citation_weight` | float | 0.15 | Citation signal weight (0-1) |

**Context Types:**
- `building`: Favor semantic breadth (20% FTS, 80% vector)
- `auditing`: Favor precision (50% FTS, 50% vector)
- `balanced`: Default balance (30% FTS, 70% vector)

**Example queries:**
- "instrumental variables assumptions"
- "double machine learning implementation"
- "heterogeneous treatment effects forests"

---

### Sources

#### `research_kb_list_sources`

List papers and textbooks in the knowledge base.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 50 | Maximum sources (1-100) |
| `offset` | int | 0 | Pagination offset |
| `source_type` | str | None | Filter: "paper" or "textbook" |

#### `research_kb_get_source`

Get detailed information about a specific source.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_id` | str | required | UUID of the source |
| `include_chunks` | bool | False | Include content chunks |
| `chunk_limit` | int | 10 | Max chunks to include (1-50) |

**Returns:** Title, authors, year, type, metadata (DOI, ISBN), and optionally content chunks.

#### `research_kb_get_source_citations`

Get bidirectional citation relationships for a source.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_id` | str | required | UUID of the source |

**Returns:** Papers citing this source (downstream influence) and papers cited by this source (foundations).

#### `research_kb_get_citing_sources`

Find all sources that cite a given source.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_id` | str | required | UUID of the source |

**Use case:** Find downstream influence and papers that built on this work.

#### `research_kb_get_cited_sources`

Find all sources that a given source cites.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_id` | str | required | UUID of the source |

**Use case:** Find foundations, context, and related work.

---

### Concepts

#### `research_kb_list_concepts`

List or search concepts in the knowledge graph.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | None | Search filter (optional) |
| `limit` | int | 50 | Maximum concepts (1-100) |
| `concept_type` | str | None | Filter by type |

**Concept Types:**
- `METHOD`: Statistical/ML methods (e.g., "causal forest", "DML")
- `ASSUMPTION`: Identifying assumptions (e.g., "unconfoundedness")
- `PROBLEM`: Research problems (e.g., "selection bias")
- `DEFINITION`: Key terms (e.g., "average treatment effect")
- `THEOREM`: Mathematical results (e.g., "Neyman orthogonality")

#### `research_kb_get_concept`

Get detailed information about a specific concept.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `concept_id` | str | required | UUID of the concept |
| `include_relationships` | bool | True | Include graph relationships |

**Relationship Types:**
- `REQUIRES`: Prerequisites
- `USES`: Method uses assumption/technique
- `ADDRESSES`: Addresses a problem
- `GENERALIZES`/`SPECIALIZES`: Hierarchy
- `ALTERNATIVE_TO`: Competing approaches
- `EXTENDS`: Extensions or improvements

#### `research_kb_chunk_concepts`

Get all concepts linked to a specific text chunk.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_id` | str | required | UUID of the chunk |

**Mention Types:**
- `defines`: Chunk provides definition/explanation
- `reference`: Chunk mentions or uses the concept
- `example`: Chunk provides an example

---

### Assumptions

#### `research_kb_audit_assumptions`

**North Star Feature**: Get required assumptions for a statistical/ML method to guide implementation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method_name` | str | required | Method name, abbreviation, or alias |
| `include_docstring` | bool | True | Include ready-to-paste code docstring |

**Example queries:**
- "double machine learning" or "DML"
- "instrumental variables" or "IV"
- "difference in differences" or "DiD"
- "regression discontinuity" or "RDD"
- "propensity score matching"

**Returns:**
- Method name and aliases (for code comments)
- Required assumptions with:
  - Formal mathematical statement (for documentation)
  - Plain English explanation (for comments)
  - Importance level: critical/standard/technical
  - Verification approaches (what to check)
  - Source citations (for references)
- Ready-to-paste docstring snippet

**Use case:** When implementing a method like DML, call this tool to understand what assumptions must hold for valid inference. Paste the docstring snippet into your implementation.

**Example output structure:**
```markdown
## Assumptions for: double machine learning
**Aliases**: DML, debiased ML

### Required Assumptions (8 found)

#### Critical (identification fails if violated)

**1. Unconfoundedness** [CRITICAL]
   - **Formal**: `Y(t) ⊥ T | X for all t`
   - **Plain English**: No unmeasured confounders
   - **Verify**: DAG review, sensitivity analysis
   - **Citation**: Chernozhukov et al. (2018), Section 2.1

### Code Docstring Snippet
```python
Assumptions:
    [CRITICAL] - unconfoundedness: No unmeasured confounders
    - overlap: Treatment probability bounded away from 0,1
```
```

---

### Graph

#### `research_kb_graph_neighborhood`

Explore the neighborhood of a concept in the knowledge graph.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `concept_name` | str | required | Concept name (fuzzy matched) |
| `hops` | int | 2 | Relationship hops (1-3) |
| `limit` | int | 50 | Max connected concepts (1-100) |

**Example:** Exploring "double machine learning" with 2 hops might reveal:
- Direct: Neyman orthogonality, cross-fitting, nuisance estimation
- 2-hop: propensity score, sample splitting, regularization

#### `research_kb_graph_path`

Find the shortest path between two concepts.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `concept_a` | str | required | First concept (fuzzy matched) |
| `concept_b` | str | required | Second concept (fuzzy matched) |

**Example:** Path from "regression discontinuity" to "instrumental variables" might show:
RD → local average treatment effect → IV

---

### Citations

#### `research_kb_citation_network`

Get bidirectional citation network for a source.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_id` | str | required | UUID of the source |
| `limit` | int | 20 | Max sources per direction (1-50) |

**Use case:** Understand a paper's position in the literature by seeing who built on it and what it builds on.

#### `research_kb_biblio_coupling`

Find sources similar by bibliographic coupling (shared references).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_id` | str | required | UUID of the source |
| `limit` | int | 10 | Max similar sources (1-50) |
| `min_coupling` | float | 0.1 | Minimum coupling threshold (0-1) |

**Coupling Formula:** `coupling = shared_refs / (refs_A + refs_B - shared_refs)` (Jaccard similarity)

**Use cases:**
- Find methodologically similar papers
- Discover papers from the same research tradition
- Identify potential related work you may have missed

---

### Health

#### `research_kb_stats`

Get statistics about the research knowledge base.

**Returns:** Counts for sources, chunks, concepts, relationships, citations, and chunk-concept mappings.

#### `research_kb_health`

Check system health with connectivity and availability checks.

**Returns:** Overall status (Healthy/Unhealthy) and component details.

---

## Usage Examples

### Search for Causal Inference Methods

```
Query: "instrumental variables assumptions"
Context: balanced
```

Returns relevant chunks with:
- Source title, authors, year
- Page numbers and section headers
- Text excerpt with score breakdown
- Source and chunk IDs for follow-up

### Explore a Concept's Context

1. `research_kb_list_concepts(query="double machine learning")`
2. Get concept ID from results
3. `research_kb_get_concept(concept_id="...", include_relationships=True)`
4. `research_kb_graph_neighborhood(concept_name="double machine learning", hops=2)`

### Understand Citation Impact

1. `research_kb_list_sources(source_type="paper")`
2. `research_kb_citation_network(source_id="...")`
3. `research_kb_biblio_coupling(source_id="...", min_coupling=0.2)`

---

## Development

### Running Tests

```bash
# All tests
pytest packages/mcp-server/tests/ -v

# With coverage
pytest packages/mcp-server/tests/ --cov=research_kb_mcp --cov-report=term-missing
```

### Adding New Tools

1. Create tool module in `src/research_kb_mcp/tools/`
2. Register tools via `register_*_tools(mcp: FastMCP)` function
3. Add formatters in `formatters.py`
4. Import and call register function in `server.py`
5. Add tests in `tests/test_*_tools.py`

### Tool Structure Pattern

```python
from fastmcp import FastMCP

def register_my_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    async def research_kb_my_tool(param: str) -> str:
        """Tool docstring (shown to Claude).

        Args:
            param: Description of parameter

        Returns:
            Markdown-formatted result
        """
        # Implementation
        return format_result(data)
```

---

## Architecture

```
research_kb_mcp/
├── server.py          # Entry point, tool registration
├── formatters.py      # Markdown output formatting
└── tools/
    ├── search.py      # Hybrid search (1 tool)
    ├── sources.py     # Source management (5 tools)
    ├── concepts.py    # Concept exploration (4 tools)
    ├── graph.py       # Knowledge graph (2 tools)
    ├── citations.py   # Citation network (2 tools)
    ├── assumptions.py # Assumption auditing (1 tool)
    └── health.py      # Health & stats (2 tools)
```

**Dependencies:**
- `fastmcp` - MCP server framework
- `research-kb-api` - Service layer
- `research-kb-storage` - Database access
- `research-kb-contracts` - Pydantic models
- `research-kb-common` - Shared utilities
