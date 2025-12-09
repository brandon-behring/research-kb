# Phase 3: Enhanced Retrieval

**Status**: ✅ **COMPLETE**
**Completed**: December 2024

---

## Overview

Phase 3 enhances retrieval quality through cross-encoder re-ranking, query expansion, and multi-hop reasoning chains.

**Research Notes**: See `docs/design/phase3_research_notes.md` for detailed research.

---

## Planned Deliverables

### 1. Cross-Encoder Re-ranking

**Approach**: Two-stage retrieval
1. **Stage 1**: Bi-encoder (fast) retrieves top-50 candidates
2. **Stage 2**: Cross-encoder (accurate) re-ranks to top-10

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parameters, fast, good accuracy)

**Implementation Pattern**:
```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query: str, chunks: list[Chunk], top_k: int = 10) -> list[Chunk]:
    pairs = [[query, chunk.content] for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_k]]
```

**Expected Impact**: +5-15% improvement in P@5

**Files to Create**:
- `packages/pdf-tools/src/research_kb_pdf/reranker.py`

---

### 2. Query Expansion

**Hybrid Strategy**:
1. **Explicit synonyms** for common abbreviations
2. **Graph expansion** for related concepts
3. **NO automatic embedding expansion** (risk of noise)

**Synonym Map** (`fixtures/concepts/synonym_map.yaml`):
```yaml
iv: ["instrumental variables", "instrumental variable", "2sls"]
did: ["difference-in-differences", "diff-in-diff"]
ate: ["average treatment effect"]
dml: ["double machine learning", "debiased ml"]
```

**Implementation**:
```python
class QueryExpander:
    def expand(self, query: str, use_graph: bool = True) -> ExpandedQuery:
        # 1. Synonym expansion
        expanded_terms = self._expand_synonyms(query)

        # 2. Graph expansion (if enabled)
        if use_graph:
            concepts = extract_concepts(query)
            for concept in concepts:
                related = self.graph.get_related(concept.id, hops=1)
                expanded_terms.extend(related)

        return ExpandedQuery(original=query, expanded_terms=expanded_terms)
```

**Expected Impact**: +3-5% improvement in recall

**Files to Create**:
- `packages/storage/src/research_kb_storage/query_expander.py`
- `fixtures/concepts/synonym_map.yaml`

---

### 3. Multi-hop Reasoning Chains

**Current Capability** (already in `graph_queries.py`):
- `find_shortest_path()` - paths between concepts
- `get_neighborhood()` - N-hop expansion
- `compute_graph_score()` - graph-based relevance

**Enhancements**:
1. **Chain Explanation**: Return the reasoning path, not just the score
2. **Path Weighting**: Weight paths by relationship types (REQUIRES > USES)
3. **Confidence Propagation**: Decay confidence along path

**Example**:
```python
def explain_connection(concept_a: str, concept_b: str) -> str:
    path = find_shortest_path(concept_a, concept_b)
    # Output: "DML → (requires) → cross-fitting → (requires) → sample splitting"
```

---

### 4. CLI Integration

**New Flags**:
```bash
research-kb query "IV" --rerank          # Enable cross-encoder re-ranking
research-kb query "IV" --no-rerank       # Disable re-ranking
research-kb query "IV" --expand-query    # Enable query expansion
research-kb query "IV" --no-expand       # Disable expansion
```

---

## Expected Improvements

| Technique | Expected P@5 Improvement | Latency Impact |
|-----------|-------------------------|----------------|
| Cross-encoder re-ranking | +5-15% | +50-100ms |
| Synonym expansion | +3-5% | Negligible |
| Graph expansion | Already integrated | Already measured |
| **Combined** | +10-20% | +50-100ms |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| P@5 improvement | ≥10% over baseline |
| Query latency | <500ms with re-ranking |
| Synonym coverage | 50+ abbreviation mappings |

---

## Previous Phase

← [Phase 2: Knowledge Graph](../phase2/KNOWLEDGE_GRAPH.md)

## Next Phase

→ [Phase 4: Production](../phase4/PRODUCTION.md)
