"""Query Concept Extraction - Extract concepts from user queries.

This module provides functionality to extract concept IDs from user query text
for graph-boosted search. It reuses the ConceptExtractor but operates in a
lightweight mode optimized for short query strings.
"""

import re
from uuid import UUID

from research_kb_common import get_logger

logger = get_logger(__name__)


async def extract_query_concepts(
    query_text: str,
    min_confidence: float = 0.6,
    max_concepts: int = 5,
) -> list[UUID]:
    """Extract concept IDs from user query text.

    This function analyzes query text to identify mentions of known concepts
    in the knowledge graph. It's designed for short user queries (not full documents).

    Strategy:
    1. Simple text matching against concept canonical names and aliases
    2. Case-insensitive matching
    3. Returns only concepts that already exist in the database

    Args:
        query_text: User query text (typically 1-10 words)
        min_confidence: Minimum confidence threshold (not currently used, reserved for future)
        max_concepts: Maximum number of concepts to extract

    Returns:
        List of concept UUIDs found in query (empty list if none found)

    Raises:
        Never raises - returns empty list on any error (graceful degradation)

    Example:
        >>> concept_ids = await extract_query_concepts(
        ...     "instrumental variables for endogeneity"
        ... )
        >>> # Returns [<UUID for IV>, <UUID for endogeneity>]
    """
    if not query_text or not query_text.strip():
        return []

    try:
        # Normalize query text
        query_lower = query_text.lower().strip()

        # Search database directly for matching concepts
        # Uses SQL for efficiency instead of loading all concepts
        from research_kb_storage.connection import get_connection_pool

        pool = await get_connection_pool()
        matched_concept_ids: list[UUID] = []

        async with pool.acquire() as conn:
            # Strategy: Search for concepts where canonical_name appears in query
            # For efficiency, we search by extracting query words and matching
            query_words = re.findall(r"\b\w+\b", query_lower)

            # Build search patterns from query words
            # For multi-word queries, also search for the full phrase
            search_patterns = []

            # Full phrase (longest first for priority)
            search_patterns.append(query_lower)

            # Bigrams (pairs of adjacent words)
            for i in range(len(query_words) - 1):
                bigram = f"{query_words[i]} {query_words[i+1]}"
                search_patterns.append(bigram)

            # Individual words (for short concept names like "IV")
            for word in query_words:
                if len(word) >= 2:  # Skip single-char words
                    search_patterns.append(word)

            # Search for exact matches on canonical_name
            for pattern in search_patterns:
                if len(matched_concept_ids) >= max_concepts:
                    break

                # Query for concepts matching this pattern
                rows = await conn.fetch(
                    """
                    SELECT id, name, canonical_name
                    FROM concepts
                    WHERE canonical_name != ''
                      AND LOWER(canonical_name) = $1
                    LIMIT $2
                    """,
                    pattern,
                    max_concepts - len(matched_concept_ids),
                )

                for row in rows:
                    if row["id"] not in matched_concept_ids:
                        matched_concept_ids.append(row["id"])
                        logger.debug(
                            "query_concept_matched",
                            concept_id=row["id"],
                            concept_name=row["name"],
                            match_type="canonical_name",
                        )

            # Also search for substring matches (longer names only)
            if len(matched_concept_ids) < max_concepts:
                rows = await conn.fetch(
                    """
                    SELECT id, name, canonical_name
                    FROM concepts
                    WHERE canonical_name != ''
                      AND LENGTH(canonical_name) > 3
                      AND POSITION(LOWER(canonical_name) IN $1) > 0
                    ORDER BY LENGTH(canonical_name) DESC
                    LIMIT $2
                    """,
                    query_lower,
                    max_concepts - len(matched_concept_ids),
                )

                for row in rows:
                    if row["id"] not in matched_concept_ids:
                        matched_concept_ids.append(row["id"])
                        logger.debug(
                            "query_concept_matched",
                            concept_id=row["id"],
                            concept_name=row["name"],
                            match_type="canonical_name_substring",
                        )

        logger.info(
            "query_concepts_extracted",
            query_text=query_text[:100],  # Truncate for logging
            matched_count=len(matched_concept_ids),
        )

        return matched_concept_ids[:max_concepts]

    except Exception as e:
        # Graceful degradation - log error but return empty list
        logger.warning(
            "query_concept_extraction_failed",
            query_text=query_text[:100],
            error=str(e),
        )
        return []


async def extract_query_concepts_by_similarity(
    query_embedding: list[float],
    min_similarity: float = 0.85,
    max_concepts: int = 5,
) -> list[UUID]:
    """Extract concepts from query using embedding similarity.

    Alternative strategy using semantic similarity between query embedding
    and concept embeddings. More robust than text matching but requires
    concept embeddings to be populated.

    Args:
        query_embedding: Query embedding vector (1024-dim, BGE-large-en-v1.5)
        min_similarity: Minimum cosine similarity threshold (0-1)
        max_concepts: Maximum number of concepts to extract

    Returns:
        List of concept UUIDs with high similarity to query

    Raises:
        Never raises - returns empty list on any error (graceful degradation)

    Example:
        >>> concept_ids = await extract_query_concepts_by_similarity(
        ...     query_embedding=[0.1] * 1024,
        ...     min_similarity=0.85
        ... )
    """
    if not query_embedding or len(query_embedding) != 1024:
        logger.warning(
            "invalid_query_embedding",
            embedding_dim=len(query_embedding) if query_embedding else 0,
        )
        return []

    try:
        from research_kb_storage.connection import get_connection_pool
        from pgvector.asyncpg import register_vector

        pool = await get_connection_pool()

        async with pool.acquire() as conn:
            await register_vector(conn)

            # Query concepts with embedding similarity
            rows = await conn.fetch(
                """
                SELECT id, name, canonical_name,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM concepts
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> $1::vector) >= $2
                ORDER BY similarity DESC
                LIMIT $3
                """,
                query_embedding,
                min_similarity,
                max_concepts,
            )

            concept_ids = [row["id"] for row in rows]

            logger.info(
                "query_concepts_extracted_by_similarity",
                matched_count=len(concept_ids),
                min_similarity=min_similarity,
            )

            return concept_ids

    except Exception as e:
        # Graceful degradation
        logger.warning(
            "query_concept_similarity_extraction_failed",
            error=str(e),
        )
        return []


async def extract_query_concepts_unified(
    query_text: str,
    query_embedding: list[float] | None = None,
    use_text_match: bool = True,
    use_semantic: bool = True,
    min_semantic_similarity: float = 0.80,
    max_concepts: int = 10,
) -> list[UUID]:
    """Extract concepts from query using combined text and semantic matching.

    Unified strategy that combines text-based and embedding-based concept
    extraction, returning deduplicated results prioritized by match strength.

    Args:
        query_text: User query text
        query_embedding: Pre-computed query embedding (1024-dim), or None
        use_text_match: Whether to use text-based matching
        use_semantic: Whether to use semantic similarity matching
        min_semantic_similarity: Minimum cosine similarity for semantic matches
        max_concepts: Maximum concepts to return

    Returns:
        List of deduplicated concept UUIDs from both strategies

    Example:
        >>> from research_kb_pdf import embed_text
        >>> embedding = embed_text("instrumental variables")
        >>> concepts = await extract_query_concepts_unified(
        ...     "how does IV work?",
        ...     query_embedding=embedding,
        ... )
    """
    if not query_text or not query_text.strip():
        return []

    results: set[UUID] = set()

    # Strategy 1: Text-based matching (exact/fuzzy on canonical names)
    if use_text_match:
        try:
            text_matches = await extract_query_concepts(
                query_text,
                max_concepts=max_concepts,
            )
            results.update(text_matches)
            logger.debug(
                "text_match_results",
                count=len(text_matches),
            )
        except Exception as e:
            logger.warning("text_match_failed", error=str(e))

    # Strategy 2: Semantic similarity matching (if embedding provided)
    if use_semantic and query_embedding:
        try:
            semantic_matches = await extract_query_concepts_by_similarity(
                query_embedding,
                min_similarity=min_semantic_similarity,
                max_concepts=max_concepts,
            )
            results.update(semantic_matches)
            logger.debug(
                "semantic_match_results",
                count=len(semantic_matches),
            )
        except Exception as e:
            logger.warning("semantic_match_failed", error=str(e))

    final_results = list(results)[:max_concepts]

    logger.info(
        "query_concepts_unified",
        query_text=query_text[:100],
        use_text=use_text_match,
        use_semantic=use_semantic and query_embedding is not None,
        total_concepts=len(final_results),
    )

    return final_results
