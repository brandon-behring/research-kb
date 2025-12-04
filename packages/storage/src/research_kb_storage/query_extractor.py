"""Query Concept Extraction - Extract concepts from user queries.

This module provides functionality to extract concept IDs from user query text
for graph-boosted search. It reuses the ConceptExtractor but operates in a
lightweight mode optimized for short query strings.
"""

from uuid import UUID

from research_kb_common import get_logger
from research_kb_storage.concept_store import ConceptStore

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

        # Get all concepts from database
        # TODO: Optimize with LIKE query or trigram similarity in future
        concepts = await ConceptStore.list_all(limit=1000)

        matched_concept_ids = []

        for concept in concepts:
            # Check canonical name match
            if concept.canonical_name in query_lower:
                matched_concept_ids.append(concept.id)
                logger.debug(
                    "query_concept_matched",
                    concept_id=concept.id,
                    concept_name=concept.name,
                    match_type="canonical_name",
                )
                continue

            # Check alias matches
            for alias in concept.aliases:
                alias_lower = alias.lower()
                # Require word boundaries for short aliases to avoid false positives
                if len(alias) <= 3:
                    # Short alias - require word boundaries
                    import re

                    pattern = rf"\b{re.escape(alias_lower)}\b"
                    if re.search(pattern, query_lower):
                        matched_concept_ids.append(concept.id)
                        logger.debug(
                            "query_concept_matched",
                            concept_id=concept.id,
                            concept_name=concept.name,
                            match_type="alias",
                            alias=alias,
                        )
                        break
                else:
                    # Long alias - simple substring match
                    if alias_lower in query_lower:
                        matched_concept_ids.append(concept.id)
                        logger.debug(
                            "query_concept_matched",
                            concept_id=concept.id,
                            concept_name=concept.name,
                            match_type="alias",
                            alias=alias,
                        )
                        break

            # Stop if we've reached max_concepts
            if len(matched_concept_ids) >= max_concepts:
                break

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
