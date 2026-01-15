# Gemini Audit Report: Research-KB

**Date**: January 8, 2026
**Subject**: methodology and code audit of `research-kb`
**Auditor**: Gemini (CLI Agent)

## 1. Executive Summary

The `research-kb` repository represents a sophisticated, domain-specific Retrieval-Augmented Generation (RAG) system tailored for causal inference. It goes beyond standard "naive RAG" by implementing a **Hybrid Search V2** architecture that combines four distinct signals: Full-Text Search (FTS), Vector Similarity (BGE-Large), Knowledge Graph (Concept Co-occurrence), and Citation Analysis (PageRank).

**Verdict**: **High Potential / Production-Grade Architecture**, but suffers from **Research-Grade Fragility** in data ingestion and evaluation consistency.

The system is well-positioned as the "Long-Term Memory" for the `lever_of_archimedes` ecosystem, but its current reliance on heuristic chunking and split evaluation metrics limits its reliability for purely autonomous operation.

---

## 2. Methodology Audit

### 2.1. Ingestion & Chunking (Critical Weakness)
**File**: `packages/pdf-tools/src/research_kb_pdf/chunker.py`

*   **Problem**: **"Fill-the-Bucket" Chunking**. The current strategy (`chunk_document`) simply aggregates paragraphs until `target_tokens` (300) is reached.
    *   *Critique*: This disregards semantic boundaries. A 300-token chunk might start with the conclusion of one idea and end with the setup of another. This dilutes the vector embedding, making it "muddy" (representing the average of two distinct topics).
    *   *Suggestion*: Implement **Semantic Chunking**. Calculate cosine similarity between consecutive sentences. If similarity drops below a threshold (e.g., 0.7), force a split.
    *   *Pros*: Higher precision embeddings. *Cons*: Variable chunk sizes.

*   **Problem**: **Brittle Metadata Alignment**.
    *   *Critique*: `chunk_with_sections` uses `full_text.find(chunk.content[:50])` to locate chunks for section metadata. If PDF extraction introduces even a single extra space or invisible character in `full_text` vs `chunk.content`, this search fails, and the chunk silently loses its section metadata (`chunk.metadata["section"] = None`).
    *   *Suggestion*: Use fuzzy matching (e.g., `thefuzz` or `rapidfuzz`) or carry character offsets through the extraction pipeline instead of re-discovering them.

### 2.2. Concept Extraction
**File**: `packages/extraction/src/research_kb_extraction/concept_extractor.py`

*   **Problem**: **Strict Relationship Filtering**.
    *   *Critique*: The code drops relationships where the source/target concept wasn't *also* extracted in the `concepts` list (`valid_relationships` list comprehension). LLMs often hallucinate a valid relationship to a concept they forgot to explicitly list in the `concepts` array.
    *   *Suggestion*: If a relationship mentions a "new" concept, implicitly add that concept to the list with a lower confidence score (e.g., 0.5) rather than discarding the relationship.

*   **Problem**: **Naive Normalization**.
    *   *Critique*: `_normalize_concept` only handles casing and whitespace. It fails to unify "Instrumental Variable" (singular) and "Instrumental Variables" (plural), or "IV" vs "I.V.".
    *   *Suggestion*: Integrate a lemmatizer (Spacy) or a dedicated "Concept Normalization" step using an LLM or vector similarity (already hinted at in `Deduplicator`).

### 2.3. Search Methodology
**File**: `packages/storage/src/research_kb_storage/search.py`

*   **Strength**: The `search_hybrid_v2` logic is robust. The 4-way signal combination with automatic re-normalization (if a signal is missing) is a best-in-class pattern.
*   **Weakness**: **Hardcoded Heuristics**.
    *   *Critique*: The weights (`fts=0.3`, `vector=0.6`, `graph=0.1`) are magic numbers.
    *   *Suggestion*: Implement **Reciprocal Rank Fusion (RRF)**. RRF is parameter-free (`1 / (k + rank)`) and often outperforms weighted sums when signals have different distributions (e.g., FTS scores vs Cosine Similarity).

### 2.4. Evaluation Methodology
**Files**: `scripts/eval_retrieval.py` vs `tests/quality/test_retrieval_quality.py`

*   **Problem**: **Split Personality**.
    *   `scripts/eval_retrieval.py` measures **Known Item Search** (finding a specific paper).
    *   `tests/quality/test_retrieval_quality.py` measures **Term Matching** (finding words in content).
    *   *Critique*: These measure different things. Term matching is a poor proxy for semantic relevance.
    *   *Suggestion*: Consolidate on **Synthetic Testsets**. Use an LLM to generate (Question, Answer, Chunk_ID) triples from the corpus. Evaluate "Hit Rate" against this ground truth. This is more scalable than manually writing YAML files.

---

## 3. Code Quality & Architecture

### 3.1. "Scripts as Tests" Anti-Pattern
The `tests/quality` directory contains tests that act more like scripts (`test_seed_concept_recall_threshold`). They have hardcoded lists of queries and print "reports" to stdout.
*   *Suggestion*: Move these to `pytest` fixtures and use standard assertions. Use `--junitxml` to generate reports rather than `print` statements.

### 3.2. Dependency Management
*   **Good**: `BGE_REVISION` is pinned in `chunker.py`. This ensures reproducibility of embeddings.
*   **Good**: `pydantic` usage for data contracts is excellent.

### 3.3. Project Structure ("Place in Universe")
*   **Context**: Located in `~/Claude/research-kb`, this project sits alongside `lever_of_archimedes`.
*   **Purpose**: It acts as the **"Hippocampus"** (Long-Term Memory). The `lever_of_archimedes` appears to be the "Prefrontal Cortex" (Executive Function).
*   **Integration**: The `daemon` and `mcp-server` packages suggest this is designed to be a "headless" service consumed by agents (like Claude Code), which is a forward-thinking design.

---

## 4. Validation & Skepticism

To rigorously test the claims made in the documentation, I conducted targeted experiments:

### 4.1. Metadata Alignment Bug (Confirmed)
I wrote a reproduction script (`reproduce_chunking_bug.py`) to test the fragility of `chunk_with_sections` in `chunker.py`.
*   **Hypothesis**: The use of `full_text.find(chunk.content[:50])` fails when PDF extraction introduces invisible characters (non-breaking spaces, soft hyphens).
*   **Result**: **CONFIRMED**. The script failed to locate chunks in the presence of `\u00A0` (NBSP) or `\u00AD` (soft hyphen), causing `chunk.metadata["section"]` to silently default to `None`.
*   **Impact**: A significant portion of the corpus likely lacks section metadata, degrading "context-aware" features.

### 4.2. Search Weights (Arbitrary)
*   **Hypothesis**: The search weights (`fts=0.3`, `vector=0.7`) were empirically derived via ablation studies.
*   **Result**: **DISPROVEN**. A global search for `*tune*` or `*ablation*` yielded no results. The weights appear to be "magic numbers" chosen based on intuition rather than data. This makes the "Graph-Boosted" claims scientifically shaky until properly benchmarked.

### 4.3. Seed Concept Quality (High)
*   **Hypothesis**: "Seed concepts" might be trivial.
*   **Result**: **VALIDATED (Good)**. `fixtures/concepts/seed_concepts.yaml` contains 48 high-quality, domain-specific concepts (e.g., "Frisch-Waugh-Lovell theorem", "local average treatment effect") with rich schema definitions (synonyms, relationships). This is a strong foundation, even if the extraction logic is flawed.

---

## 5. Key Recommendations

1.  **Switch to RRF**: Replace the weighted sum formula in `search.py` with Reciprocal Rank Fusion to remove magic numbers.
2.  **Fuzzy Metadata**: Replace `text.find()` in `chunker.py` with fuzzy matching to prevent metadata loss.
3.  **Implicit Concept Creation**: Allow relationships to create "shadow concepts" during extraction to improve graph recall.
4.  **Synthetic Eval**: Create a script `generate_eval_dataset.py` that uses an LLM to generate 100 QA pairs from the corpus, and use *that* as the ground truth for `eval_retrieval.py`.

---

## 6. Strategic Roadmap: Real-Time Domain Synthesis

Based on the audit and your goal of a "Proactive Real-Time Assistant" that synthesizes domains, the current architecture requires specific upgrades:

### 6.1. The Performance Bottleneck (Postgres vs. Proactive)
*   **The Problem**: You want the system to *proactively* suggest context while you work. This requires running multi-hop graph queries (to find non-obvious connections) in the background constantly. Postgres Recursive CTEs are too slow (hundreds of ms to seconds for 3+ hops) for this loop.
*   **Recommendation**: **Accelerate Phase 5**. You need a graph-native engine *now*.
    *   **Lightweight Option**: **KuzuDB** (Embedded, like DuckDB for graphs). Fits your Python stack perfectly, faster than Neo4j for this scale.
    *   **Production Option**: **Neo4j**. Overkill unless you plan to scale beyond 10M edges.
    *   **Immediate Action**: Load your 41k concepts/37k edges into **NetworkX** (in-memory) for the `daemon`. It will be instant (<10ms) for 4-hop queries.

### 6.2. The "Golden Dataset" Recipe
To stop guessing at "Magic Numbers" for search weights, execute this recipe to create a ground truth dataset:

1.  **Source**: Select 5 key chapters from *Mostly Harmless Econometrics* (your "Bible").
2.  **Generation**: Use `claude-3-opus` or `gpt-4` with this prompt:
    > "Read this text. Generate 10 complex questions that require connecting two distinct concepts to answer. For each question, identify the exact 2-3 paragraphs in the text that contain the answer."
3.  **Result**: A `golden_dataset.json` with 50 pairs of `(Question, [Target_Chunk_IDs])`.
4.  **Tuning**: Write a script that runs these 50 questions through your search engine with different weights (e.g., `graph=0.1` vs `graph=0.5`) and maximizes the "Hit Rate" on the Target Chunks.

### 6.3. The "Synthesis" Feature (Path-Augmented Generation)
To achieve "Domain Synthesis," you must implement this logic flow (currently missing):
1.  **Find the Bridge**: `path = find_shortest_path(Concept_A, Concept_B)`
2.  **Hydrate Context**: Fetch the definition and best text chunk for every node in that path.
3.  **Synthesize**: Feed the *entire path + definitions* to the LLM with the prompt: *"Explain how [Concept A] relates to [Concept B] using this conceptual chain..."*

---

## 7. Citations & References
*   **On RRF**: *Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. SIGIR.*
*   **On Semantic Chunking**: *Kamradt, G. (2024). 5 Levels of Text Splitting.* (General industry consensus on moving beyond fixed-size chunks).
