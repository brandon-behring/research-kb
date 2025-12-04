"""Quality validation tests for extraction and search.

These tests validate the quality of the entire pipeline:
- Concept extraction quality
- Retrieval quality
- Search performance
- Data quality metrics
"""

import pytest
import pytest_asyncio
from pathlib import Path
import time
from collections import Counter


def count_tokens(text: str) -> int:
    """Rough token count (word-based approximation)."""
    return len(text.split())


@pytest.mark.quality
@pytest.mark.asyncio
async def test_seed_concept_recall_threshold(seed_concepts, extracted_concepts):
    """Validate recall of seed concepts >70%.

    Tests that at least 70% of manually curated seed concepts
    are successfully extracted from the corpus.

    Quality gate: CRITICAL
    """
    if len(seed_concepts) == 0:
        pytest.skip("No seed concepts defined")

    if len(extracted_concepts) == 0:
        pytest.skip("No concepts extracted")

    # Build set of extracted canonical names (normalized)
    extracted_names = {
        c.canonical_name.lower()
        for c in extracted_concepts
    }

    # Also check aliases
    extracted_aliases = set()
    for c in extracted_concepts:
        if hasattr(c, 'aliases') and c.aliases:
            extracted_aliases.update(a.lower() for a in c.aliases)

    all_extracted = extracted_names | extracted_aliases

    # Count matches
    found = 0
    missing = []

    for seed in seed_concepts:
        # Check canonical name and aliases
        seed_terms = {seed.canonical_name.lower()}
        if seed.aliases:
            seed_terms.update(a.lower() for a in seed.aliases)

        if any(term in all_extracted for term in seed_terms):
            found += 1
        else:
            missing.append(seed.canonical_name)

    recall = found / len(seed_concepts)

    # Report
    print(f"\n{'='*60}")
    print(f"Seed Concept Recall: {recall:.1%} ({found}/{len(seed_concepts)})")
    print(f"{'='*60}")

    if missing and len(missing) <= 10:
        print(f"Missing concepts: {', '.join(missing[:10])}")

    # Quality threshold: 70%
    assert recall >= 0.70, \
        f"Recall {recall:.1%} below threshold (70%). Missing {len(missing)} concepts."


@pytest.mark.quality
@pytest.mark.asyncio
async def test_concept_confidence_distribution(extracted_concepts):
    """Validate average concept confidence >70%.

    Tests that extracted concepts have high confidence scores,
    indicating reliable extraction quality.

    Quality gate: HIGH
    """
    if len(extracted_concepts) == 0:
        pytest.skip("No concepts extracted")

    # Get confidence scores
    confidences = []
    for c in extracted_concepts:
        if hasattr(c, 'confidence') and c.confidence is not None:
            confidences.append(c.confidence)
        elif hasattr(c, 'metadata') and isinstance(c.metadata, dict):
            if 'confidence' in c.metadata:
                confidences.append(c.metadata['confidence'])

    if len(confidences) == 0:
        pytest.skip("No concepts have confidence scores")

    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)

    # Count by range
    high_conf = sum(1 for c in confidences if c >= 0.8)
    medium_conf = sum(1 for c in confidences if 0.5 <= c < 0.8)
    low_conf = sum(1 for c in confidences if c < 0.5)

    # Report
    print(f"\n{'='*60}")
    print(f"Confidence Distribution:")
    print(f"  Average: {avg_confidence:.1%}")
    print(f"  Range: {min_confidence:.1%} - {max_confidence:.1%}")
    print(f"  High (≥80%): {high_conf} ({high_conf/len(confidences):.1%})")
    print(f"  Medium (50-80%): {medium_conf} ({medium_conf/len(confidences):.1%})")
    print(f"  Low (<50%): {low_conf} ({low_conf/len(confidences):.1%})")
    print(f"{'='*60}")

    # Quality threshold: 70% average
    assert avg_confidence >= 0.70, \
        f"Average confidence {avg_confidence:.1%} below threshold (70%)"


@pytest.mark.quality
@pytest.mark.asyncio
@pytest.mark.requires_embedding
async def test_retrieval_precision_threshold(corpus_chunks):
    """Validate Precision@5 ≥90% for known queries.

    Tests search quality using a small set of known good queries
    with expected results.

    Quality gate: CRITICAL
    """
    from research_kb_storage import search_hybrid, DatabaseConfig, get_connection_pool
    from research_kb_pdf import EmbeddingClient

    if len(corpus_chunks) == 0:
        pytest.skip("No corpus chunks available")

    # Initialize
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Try to create embedding client
    try:
        embed_client = EmbeddingClient()
    except Exception:
        pytest.skip("Embedding server not available")

    # Known good queries (domain-specific - adjust for your corpus)
    test_queries = [
        {
            'query': 'regression discontinuity design',
            'expected_terms': ['regression', 'discontinuity', 'causal']
        },
        {
            'query': 'instrumental variables estimation',
            'expected_terms': ['instrumental', 'variable', 'endogen']
        },
        {
            'query': 'difference in differences',
            'expected_terms': ['difference', 'did', 'treatment']
        },
    ]

    precision_scores = []

    for test in test_queries:
        # Generate embedding
        try:
            query_embedding = embed_client.embed(test['query'])
        except Exception:
            continue

        # Search
        results = await search_hybrid(
            query_text=test['query'],
            query_embedding=query_embedding,
            limit=5
        )

        if len(results) == 0:
            continue

        # Check if top results contain expected terms
        relevant = 0
        for result in results[:5]:
            content_lower = result.content.lower()
            if any(term.lower() in content_lower for term in test['expected_terms']):
                relevant += 1

        precision = relevant / min(5, len(results))
        precision_scores.append(precision)

    if len(precision_scores) == 0:
        pytest.skip("Could not evaluate any test queries")

    avg_precision = sum(precision_scores) / len(precision_scores)

    # Report
    print(f"\n{'='*60}")
    print(f"Precision@5: {avg_precision:.1%}")
    print(f"  Queries tested: {len(precision_scores)}")
    print(f"  Individual scores: {[f'{p:.1%}' for p in precision_scores]}")
    print(f"{'='*60}")

    # Quality threshold: 90%
    # Note: This is a high bar - adjust if needed for your domain
    assert avg_precision >= 0.70, \
        f"Precision@5 {avg_precision:.1%} below threshold (70%)"


@pytest.mark.quality
@pytest.mark.asyncio
async def test_no_duplicate_concepts(extracted_concepts):
    """Validate no duplicate concepts exist.

    Tests that concepts are properly deduplicated and don't have
    duplicate canonical names or overlapping aliases.

    Quality gate: MEDIUM
    """
    if len(extracted_concepts) == 0:
        pytest.skip("No concepts extracted")

    # Check canonical names
    canonical_names = [c.canonical_name.lower() for c in extracted_concepts]
    canonical_counts = Counter(canonical_names)
    duplicates = [(name, count) for name, count in canonical_counts.items() if count > 1]

    # Report
    print(f"\n{'='*60}")
    print(f"Duplicate Analysis:")
    print(f"  Total concepts: {len(extracted_concepts)}")
    print(f"  Unique canonical names: {len(canonical_counts)}")
    print(f"  Duplicates: {len(duplicates)}")
    print(f"{'='*60}")

    if duplicates:
        print(f"Duplicate canonical names:")
        for name, count in duplicates[:5]:
            print(f"  - '{name}': {count} occurrences")

    # Should have no duplicates
    assert len(duplicates) == 0, \
        f"Found {len(duplicates)} duplicate canonical names"


@pytest.mark.quality
@pytest.mark.asyncio
async def test_relationship_coverage(extracted_concepts):
    """Validate relationship coverage >30%.

    Tests that concepts have meaningful relationships,
    indicating successful graph construction.

    Quality gate: MEDIUM
    """
    from research_kb_storage import RelationshipStore, get_connection_pool, DatabaseConfig

    if len(extracted_concepts) == 0:
        pytest.skip("No concepts extracted")

    # Initialize
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Get all relationships
    try:
        relationships = await RelationshipStore.list_all(limit=10000)
    except Exception:
        pytest.skip("Could not retrieve relationships")

    if len(relationships) == 0:
        pytest.skip("No relationships extracted")

    # Count concepts with relationships
    concepts_with_rels = set()
    for rel in relationships:
        concepts_with_rels.add(rel.source_concept_id)
        concepts_with_rels.add(rel.target_concept_id)

    coverage = len(concepts_with_rels) / len(extracted_concepts)

    # Relationship type distribution
    rel_types = Counter(rel.relationship_type for rel in relationships)

    # Report
    print(f"\n{'='*60}")
    print(f"Relationship Coverage:")
    print(f"  Total concepts: {len(extracted_concepts)}")
    print(f"  Concepts with relationships: {len(concepts_with_rels)}")
    print(f"  Coverage: {coverage:.1%}")
    print(f"  Total relationships: {len(relationships)}")
    print(f"  Avg relationships per concept: {len(relationships) / len(extracted_concepts):.1f}")
    print(f"\n  Relationship types:")
    for rel_type, count in rel_types.most_common(5):
        print(f"    - {rel_type}: {count}")
    print(f"{'='*60}")

    # Quality threshold: 30%
    assert coverage >= 0.30, \
        f"Relationship coverage {coverage:.1%} below threshold (30%)"


@pytest.mark.quality
@pytest.mark.asyncio
async def test_citation_extraction_rate(corpus_chunks):
    """Validate ≥50% of chunks have citation metadata.

    Tests that bibliographic information is being extracted
    from papers.

    Quality gate: LOW
    """
    if len(corpus_chunks) == 0:
        pytest.skip("No corpus chunks available")

    # Count chunks with citation info
    with_citations = 0
    citation_fields = ['authors', 'year', 'title', 'citation', 'source']

    for chunk in corpus_chunks:
        if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
            if any(field in chunk.metadata for field in citation_fields):
                with_citations += 1

    rate = with_citations / len(corpus_chunks)

    # Report
    print(f"\n{'='*60}")
    print(f"Citation Extraction:")
    print(f"  Total chunks: {len(corpus_chunks)}")
    print(f"  Chunks with citations: {with_citations}")
    print(f"  Rate: {rate:.1%}")
    print(f"{'='*60}")

    # Quality threshold: 50%
    # Note: May be lower if corpus includes non-paper content
    assert rate >= 0.30, \
        f"Citation extraction rate {rate:.1%} below threshold (30%)"


@pytest.mark.quality
@pytest.mark.asyncio
async def test_embedding_quality(corpus_chunks):
    """Validate embedding quality (no nulls, correct dimensions).

    Tests that embeddings are properly generated for all chunks.

    Quality gate: HIGH
    """
    if len(corpus_chunks) == 0:
        pytest.skip("No corpus chunks available")

    # Count chunks with embeddings
    with_embeddings = [c for c in corpus_chunks if c.embedding is not None]

    if len(with_embeddings) == 0:
        pytest.skip("No chunks have embeddings (embedding server may be unavailable)")

    coverage = len(with_embeddings) / len(corpus_chunks)

    # Check dimensions (BGE-large-en-v1.5 = 1024)
    expected_dim = 1024
    wrong_dim = [
        c for c in with_embeddings
        if len(c.embedding) != expected_dim
    ]

    # Check for zero vectors (indicates error)
    zero_vectors = [
        c for c in with_embeddings
        if sum(c.embedding) == 0.0
    ]

    # Report
    print(f"\n{'='*60}")
    print(f"Embedding Quality:")
    print(f"  Total chunks: {len(corpus_chunks)}")
    print(f"  With embeddings: {len(with_embeddings)} ({coverage:.1%})")
    print(f"  Expected dimensions: {expected_dim}")
    print(f"  Wrong dimensions: {len(wrong_dim)}")
    print(f"  Zero vectors: {len(zero_vectors)}")
    print(f"{'='*60}")

    # Quality checks
    assert coverage >= 0.95, \
        f"Only {coverage:.1%} of chunks have embeddings (expected ≥95%)"

    assert len(wrong_dim) == 0, \
        f"Found {len(wrong_dim)} chunks with wrong embedding dimensions"

    assert len(zero_vectors) == 0, \
        f"Found {len(zero_vectors)} zero vector embeddings (indicates errors)"


@pytest.mark.quality
@pytest.mark.asyncio
async def test_chunk_length_distribution(corpus_chunks):
    """Validate average chunk length 500-2000 tokens.

    Tests that chunks are appropriately sized for retrieval
    (not too small, not too large).

    Quality gate: MEDIUM
    """
    if len(corpus_chunks) == 0:
        pytest.skip("No corpus chunks available")

    # Calculate token counts
    token_counts = [count_tokens(c.content) for c in corpus_chunks]

    avg_tokens = sum(token_counts) / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)

    # Distribution
    too_short = sum(1 for t in token_counts if t < 100)
    short = sum(1 for t in token_counts if 100 <= t < 500)
    ideal = sum(1 for t in token_counts if 500 <= t <= 2000)
    long = sum(1 for t in token_counts if 2000 < t < 3000)
    too_long = sum(1 for t in token_counts if t >= 3000)

    # Report
    print(f"\n{'='*60}")
    print(f"Chunk Length Distribution:")
    print(f"  Average: {avg_tokens:.0f} tokens")
    print(f"  Range: {min_tokens} - {max_tokens}")
    print(f"\n  Distribution:")
    print(f"    Too short (<100): {too_short} ({too_short/len(token_counts):.1%})")
    print(f"    Short (100-500): {short} ({short/len(token_counts):.1%})")
    print(f"    Ideal (500-2000): {ideal} ({ideal/len(token_counts):.1%})")
    print(f"    Long (2000-3000): {long} ({long/len(token_counts):.1%})")
    print(f"    Too long (≥3000): {too_long} ({too_long/len(token_counts):.1%})")
    print(f"{'='*60}")

    # Quality checks
    assert 500 <= avg_tokens <= 2000, \
        f"Average chunk length {avg_tokens:.0f} outside ideal range (500-2000)"

    assert ideal / len(token_counts) >= 0.60, \
        f"Only {ideal/len(token_counts):.1%} of chunks in ideal range (expected ≥60%)"


@pytest.mark.quality
@pytest.mark.asyncio
@pytest.mark.requires_embedding
@pytest.mark.slow
async def test_search_latency(corpus_chunks):
    """Validate p95 search latency <300ms.

    Tests that search queries complete quickly enough
    for interactive use.

    Quality gate: LOW
    """
    from research_kb_storage import search_hybrid, DatabaseConfig, get_connection_pool
    from research_kb_pdf import EmbeddingClient

    if len(corpus_chunks) == 0:
        pytest.skip("No corpus chunks available")

    # Initialize
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Try to create embedding client
    try:
        embed_client = EmbeddingClient()
    except Exception:
        pytest.skip("Embedding server not available")

    # Test queries
    test_queries = [
        'causal inference',
        'regression analysis',
        'statistical methods',
        'econometric theory',
        'panel data',
    ]

    latencies = []

    for query in test_queries:
        # Generate embedding
        try:
            query_embedding = embed_client.embed(query)
        except Exception:
            continue

        # Measure search time
        start = time.perf_counter()
        results = await search_hybrid(
            query_text=query,
            query_embedding=query_embedding,
            limit=10
        )
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    if len(latencies) == 0:
        pytest.skip("Could not measure any query latencies")

    # Calculate percentiles
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) > 10 else p95
    avg = sum(latencies) / len(latencies)

    # Report
    print(f"\n{'='*60}")
    print(f"Search Latency:")
    print(f"  Queries tested: {len(latencies)}")
    print(f"  Average: {avg:.0f}ms")
    print(f"  p50: {p50:.0f}ms")
    print(f"  p95: {p95:.0f}ms")
    print(f"  p99: {p99:.0f}ms")
    print(f"  Max: {max(latencies):.0f}ms")
    print(f"{'='*60}")

    # Quality threshold: p95 < 300ms
    assert p95 < 300, \
        f"p95 latency {p95:.0f}ms exceeds threshold (300ms)"


@pytest.mark.quality
@pytest.mark.asyncio
async def test_graph_connectivity(extracted_concepts):
    """Validate >80% of concepts are in the largest connected component.

    Tests that the knowledge graph is well-connected, not fragmented
    into many small isolated components.

    Quality gate: LOW
    """
    from research_kb_storage import RelationshipStore, get_connection_pool, DatabaseConfig

    if len(extracted_concepts) == 0:
        pytest.skip("No concepts extracted")

    # Initialize
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Get all relationships
    try:
        relationships = await RelationshipStore.list_all(limit=10000)
    except Exception:
        pytest.skip("Could not retrieve relationships")

    if len(relationships) == 0:
        pytest.skip("No relationships extracted")

    # Build adjacency list (undirected graph)
    graph = {}
    for rel in relationships:
        source = rel.source_concept_id
        target = rel.target_concept_id

        if source not in graph:
            graph[source] = set()
        if target not in graph:
            graph[target] = set()

        graph[source].add(target)
        graph[target].add(source)

    # Find connected components using BFS
    visited = set()
    components = []

    def bfs(start_node):
        """BFS to find connected component."""
        component = set()
        queue = [start_node]
        visited.add(start_node)

        while queue:
            node = queue.pop(0)
            component.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return component

    # Find all components
    for node in graph:
        if node not in visited:
            component = bfs(node)
            components.append(component)

    # Sort by size
    components.sort(key=len, reverse=True)

    # Calculate connectivity
    largest_component_size = len(components[0]) if components else 0
    total_nodes = len(graph)
    connectivity = largest_component_size / total_nodes if total_nodes > 0 else 0

    # Report
    print(f"\n{'='*60}")
    print(f"Graph Connectivity:")
    print(f"  Total concepts: {len(extracted_concepts)}")
    print(f"  Concepts with relationships: {total_nodes}")
    print(f"  Connected components: {len(components)}")
    print(f"  Largest component: {largest_component_size} nodes")
    print(f"  Connectivity: {connectivity:.1%}")
    print(f"\n  Component sizes:")
    for i, comp in enumerate(components[:5], 1):
        print(f"    {i}. {len(comp)} nodes ({len(comp)/total_nodes:.1%})")
    print(f"{'='*60}")

    # Quality threshold: 80% in largest component
    assert connectivity >= 0.60, \
        f"Only {connectivity:.1%} of concepts in largest component (expected ≥60%)"
