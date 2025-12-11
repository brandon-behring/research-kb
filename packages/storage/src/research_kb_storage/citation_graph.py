"""Citation Graph - Build and query source-to-source citation links.

Provides:
- Match extracted citations to corpus sources
- Build source_citations edges
- Compute PageRank-style authority scores
- Query citation relationships with type awareness

Phase 3: Citation graph integration for search enhancement.
"""

import json
from typing import Optional
from uuid import UUID

from research_kb_common import get_logger
from research_kb_contracts import Citation, SourceType

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


# ============================================================================
# Citation Matching
# ============================================================================


async def match_citation_to_source(citation: Citation) -> Optional[UUID]:
    """Match a citation to a source in our corpus.

    Matching priority:
    1. DOI exact match (highest confidence)
    2. arXiv ID exact match
    3. Fuzzy match on title + year + first author (threshold 0.85)

    Args:
        citation: Citation to match

    Returns:
        Source UUID if matched, None if external to corpus
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

        # Priority 1: DOI exact match
        if citation.doi:
            row = await conn.fetchrow(
                """
                SELECT id FROM sources
                WHERE metadata->>'doi' = $1
                LIMIT 1
                """,
                citation.doi,
            )
            if row:
                return row["id"]

        # Priority 2: arXiv ID exact match
        if citation.arxiv_id:
            row = await conn.fetchrow(
                """
                SELECT id FROM sources
                WHERE metadata->>'arxiv_id' = $1
                LIMIT 1
                """,
                citation.arxiv_id,
            )
            if row:
                return row["id"]

        # Priority 3: Fuzzy match on title + year + first author
        if citation.title:
            # Normalize title for matching
            normalized_title = citation.title.lower().strip()

            # Use trigram similarity if available, fallback to ILIKE
            row = await conn.fetchrow(
                """
                SELECT id, title,
                       similarity(LOWER(title), $1) AS title_sim
                FROM sources
                WHERE LOWER(title) % $1  -- Trigram similarity operator
                  AND ($2::int IS NULL OR year = $2 OR year IS NULL)
                ORDER BY similarity(LOWER(title), $1) DESC
                LIMIT 1
                """,
                normalized_title,
                citation.year,
            )

            if row and row["title_sim"] >= 0.85:
                logger.debug(
                    "fuzzy_match_found",
                    citation_title=citation.title,
                    source_title=row["title"],
                    similarity=row["title_sim"],
                )
                return row["id"]

    return None


async def match_citation_to_source_simple(citation: Citation) -> Optional[UUID]:
    """Simple citation matching without trigram extension.

    Falls back to exact title match + year.
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

        # Priority 1: DOI exact match
        if citation.doi:
            row = await conn.fetchrow(
                "SELECT id FROM sources WHERE metadata->>'doi' = $1 LIMIT 1",
                citation.doi,
            )
            if row:
                return row["id"]

        # Priority 2: arXiv ID exact match
        if citation.arxiv_id:
            row = await conn.fetchrow(
                "SELECT id FROM sources WHERE metadata->>'arxiv_id' = $1 LIMIT 1",
                citation.arxiv_id,
            )
            if row:
                return row["id"]

        # Priority 3: Exact title + year match
        if citation.title:
            normalized_title = citation.title.lower().strip()

            row = await conn.fetchrow(
                """
                SELECT id FROM sources
                WHERE LOWER(title) = $1
                  AND ($2::int IS NULL OR year = $2)
                LIMIT 1
                """,
                normalized_title,
                citation.year,
            )
            if row:
                return row["id"]

            # Try partial match (title contains)
            row = await conn.fetchrow(
                """
                SELECT id FROM sources
                WHERE LOWER(title) LIKE '%' || $1 || '%'
                   OR $1 LIKE '%' || LOWER(title) || '%'
                LIMIT 1
                """,
                normalized_title,
            )
            if row:
                return row["id"]

    return None


# ============================================================================
# Graph Building
# ============================================================================


async def build_citation_graph() -> dict:
    """Build source_citations edges from extracted citations.

    For each citation in the citations table:
    1. Match to a source in our corpus (or NULL for external)
    2. Create edge in source_citations table
    3. Track statistics by source type

    Returns:
        Statistics dict with keys:
        - total_processed: int
        - matched: int (citations matched to corpus sources)
        - unmatched: int (external citations)
        - by_type: dict mapping "PAPER→PAPER" etc. to counts
        - errors: int
    """
    pool = await get_connection_pool()
    stats = {
        "total_processed": 0,
        "matched": 0,
        "unmatched": 0,
        "by_type": {},
        "errors": 0,
    }

    async with pool.acquire() as conn:
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

        # Get all citations with their source info
        citations = await conn.fetch(
            """
            SELECT c.*, s.source_type AS citing_source_type
            FROM citations c
            JOIN sources s ON c.source_id = s.id
            ORDER BY c.created_at
            """
        )

        logger.info("building_citation_graph", total_citations=len(citations))

        for cit_row in citations:
            stats["total_processed"] += 1

            try:
                # Build Citation object for matching
                citation = Citation(
                    id=cit_row["id"],
                    source_id=cit_row["source_id"],
                    authors=cit_row["authors"] or [],
                    title=cit_row["title"],
                    year=cit_row["year"],
                    venue=cit_row["venue"],
                    doi=cit_row["doi"],
                    arxiv_id=cit_row["arxiv_id"],
                    raw_string=cit_row["raw_string"],
                )

                # Try to match to corpus source
                cited_source_id = await match_citation_to_source_simple(citation)

                # Check if edge already exists
                existing = await conn.fetchval(
                    """
                    SELECT id FROM source_citations
                    WHERE citing_source_id = $1 AND citation_id = $2
                    """,
                    cit_row["source_id"],
                    cit_row["id"],
                )

                if existing:
                    continue  # Skip duplicate

                # Insert edge
                await conn.execute(
                    """
                    INSERT INTO source_citations (citing_source_id, cited_source_id, citation_id)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (citing_source_id, citation_id) DO NOTHING
                    """,
                    cit_row["source_id"],
                    cited_source_id,
                    cit_row["id"],
                )

                if cited_source_id:
                    stats["matched"] += 1

                    # Get cited source type for statistics
                    cited_type = await conn.fetchval(
                        "SELECT source_type FROM sources WHERE id = $1",
                        cited_source_id,
                    )

                    citing_type = cit_row["citing_source_type"]
                    key = f"{citing_type}→{cited_type}"
                    stats["by_type"][key] = stats["by_type"].get(key, 0) + 1
                else:
                    stats["unmatched"] += 1

            except Exception as e:
                stats["errors"] += 1
                logger.warning(
                    "citation_graph_edge_failed",
                    citation_id=str(cit_row["id"]),
                    error=str(e),
                )

            if stats["total_processed"] % 500 == 0:
                logger.info(
                    "citation_graph_progress",
                    processed=stats["total_processed"],
                    matched=stats["matched"],
                )

    logger.info(
        "citation_graph_built",
        total=stats["total_processed"],
        matched=stats["matched"],
        unmatched=stats["unmatched"],
        errors=stats["errors"],
    )

    return stats


# ============================================================================
# PageRank Authority Computation
# ============================================================================


async def compute_pagerank_authority(
    iterations: int = 20,
    damping: float = 0.85,
) -> dict:
    """Compute PageRank-style citation authority scores.

    Algorithm:
    1. Initialize all sources with equal score (1/N)
    2. Iterate: score = (1-d)/N + d * sum(score[citing] / out_degree[citing])
    3. Persist final scores to sources.citation_authority

    Args:
        iterations: Number of PageRank iterations (default: 20)
        damping: Damping factor (default: 0.85)

    Returns:
        Statistics dict with min, max, mean scores
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        # Get all source IDs
        source_rows = await conn.fetch("SELECT id FROM sources")
        source_ids = [row["id"] for row in source_rows]
        n = len(source_ids)

        if n == 0:
            return {"error": "No sources"}

        # Initialize scores
        scores = {sid: 1.0 / n for sid in source_ids}

        logger.info("computing_pagerank", sources=n, iterations=iterations)

        # Get citation graph edges
        edges = await conn.fetch(
            """
            SELECT citing_source_id, cited_source_id
            FROM source_citations
            WHERE cited_source_id IS NOT NULL
            """
        )

        # Build adjacency lists
        incoming = {sid: [] for sid in source_ids}  # Who cites me
        outgoing = {sid: 0 for sid in source_ids}   # How many I cite

        for edge in edges:
            citing = edge["citing_source_id"]
            cited = edge["cited_source_id"]
            if citing in outgoing and cited in incoming:
                incoming[cited].append(citing)
                outgoing[citing] += 1

        # PageRank iterations
        for i in range(iterations):
            new_scores = {}

            for sid in source_ids:
                # Sum contributions from citing sources
                incoming_score = 0.0
                for citing_id in incoming[sid]:
                    if outgoing[citing_id] > 0:
                        incoming_score += scores[citing_id] / outgoing[citing_id]

                new_scores[sid] = (1 - damping) / n + damping * incoming_score

            scores = new_scores

            if (i + 1) % 5 == 0:
                logger.debug("pagerank_iteration", iteration=i + 1)

        # Normalize to 0-1 range
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {sid: s / max_score for sid, s in scores.items()}

        # Persist scores
        async with conn.transaction():
            for sid, score in scores.items():
                await conn.execute(
                    "UPDATE sources SET citation_authority = $1 WHERE id = $2",
                    score,
                    sid,
                )

        score_values = list(scores.values())
        stats = {
            "sources": n,
            "min_score": min(score_values),
            "max_score": max(score_values),
            "mean_score": sum(score_values) / n,
        }

        logger.info("pagerank_computed", **stats)

        return stats


# ============================================================================
# Citation Graph Queries
# ============================================================================


async def get_citing_sources(
    source_id: UUID,
    source_type: Optional[SourceType] = None,
    limit: int = 100,
) -> list[dict]:
    """Find sources that cite this source.

    Args:
        source_id: Source being cited
        source_type: Optional filter (PAPER, TEXTBOOK)
        limit: Maximum results

    Returns:
        List of dicts with source info + citation count
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

        query = """
            SELECT s.id, s.source_type, s.title, s.authors, s.year,
                   s.citation_authority,
                   COUNT(sc.id) AS citation_count
            FROM sources s
            JOIN source_citations sc ON sc.citing_source_id = s.id
            WHERE sc.cited_source_id = $1
        """

        params = [source_id]

        if source_type:
            query += " AND s.source_type = $2"
            params.append(source_type.value)

        query += """
            GROUP BY s.id
            ORDER BY COUNT(sc.id) DESC, s.citation_authority DESC
            LIMIT ${}
        """.format(len(params) + 1)

        params.append(limit)

        rows = await conn.fetch(query, *params)

        return [
            {
                "id": row["id"],
                "source_type": row["source_type"],
                "title": row["title"],
                "authors": row["authors"],
                "year": row["year"],
                "citation_authority": row["citation_authority"],
                "citation_count": row["citation_count"],
            }
            for row in rows
        ]


async def get_cited_sources(
    source_id: UUID,
    source_type: Optional[SourceType] = None,
    limit: int = 100,
) -> list[dict]:
    """Find sources cited by this source.

    Args:
        source_id: Source doing the citing
        source_type: Optional filter (PAPER, TEXTBOOK)
        limit: Maximum results

    Returns:
        List of dicts with source info
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

        query = """
            SELECT DISTINCT s.id, s.source_type, s.title, s.authors, s.year,
                   s.citation_authority
            FROM sources s
            JOIN source_citations sc ON sc.cited_source_id = s.id
            WHERE sc.citing_source_id = $1
        """

        params = [source_id]

        if source_type:
            query += " AND s.source_type = $2"
            params.append(source_type.value)

        query += """
            ORDER BY s.citation_authority DESC
            LIMIT ${}
        """.format(len(params) + 1)

        params.append(limit)

        rows = await conn.fetch(query, *params)

        return [
            {
                "id": row["id"],
                "source_type": row["source_type"],
                "title": row["title"],
                "authors": row["authors"],
                "year": row["year"],
                "citation_authority": row["citation_authority"],
            }
            for row in rows
        ]


async def get_citation_stats(source_id: UUID) -> dict:
    """Get citation statistics for a source.

    Returns:
        Dict with:
        - cited_by_count: Total sources citing this
        - cited_by_papers: Papers citing this
        - cited_by_textbooks: Textbooks citing this
        - cites_count: Total sources this cites
        - cites_papers: Papers this cites
        - cites_textbooks: Textbooks this cites
        - citation_authority: PageRank score
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM source_citation_stats
            WHERE source_id = $1
            """,
            source_id,
        )

        if not row:
            return {
                "cited_by_count": 0,
                "cited_by_papers": 0,
                "cited_by_textbooks": 0,
                "cites_count": 0,
                "cites_papers": 0,
                "cites_textbooks": 0,
                "citation_authority": 0.0,
            }

        return {
            "cited_by_count": row["cited_by_count"],
            "cited_by_papers": row["cited_by_papers"],
            "cited_by_textbooks": row["cited_by_textbooks"],
            "cites_count": row["cites_count"],
            "cites_papers": row["cites_papers"],
            "cites_textbooks": row["cites_textbooks"],
            "citation_authority": row["citation_authority"] or 0.0,
        }


async def get_corpus_citation_summary() -> dict:
    """Get corpus-wide citation statistics.

    Returns:
        Dict with total citations, internal/external breakdown, type combinations
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM corpus_citation_summary")

        if not row:
            return {}

        return {
            "total_citations": row["total_citations"],
            "total_edges": row["total_edges"],
            "internal_edges": row["internal_edges"],
            "external_edges": row["external_edges"],
            "paper_to_paper": row["paper_to_paper"],
            "paper_to_textbook": row["paper_to_textbook"],
            "textbook_to_paper": row["textbook_to_paper"],
            "textbook_to_textbook": row["textbook_to_textbook"],
        }


async def get_most_cited_sources(
    source_type: Optional[SourceType] = None,
    limit: int = 10,
) -> list[dict]:
    """Get most cited sources in corpus.

    Args:
        source_type: Optional filter
        limit: Maximum results

    Returns:
        List of sources with citation counts
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

        query = """
            SELECT s.id, s.source_type, s.title, s.authors, s.year,
                   s.citation_authority,
                   COUNT(sc.id) AS cited_by_count
            FROM sources s
            LEFT JOIN source_citations sc ON sc.cited_source_id = s.id
        """

        params = []

        if source_type:
            query += " WHERE s.source_type = $1"
            params.append(source_type.value)

        query += """
            GROUP BY s.id
            HAVING COUNT(sc.id) > 0
            ORDER BY COUNT(sc.id) DESC, s.citation_authority DESC
            LIMIT ${}
        """.format(len(params) + 1)

        params.append(limit)

        rows = await conn.fetch(query, *params)

        return [
            {
                "id": row["id"],
                "source_type": row["source_type"],
                "title": row["title"],
                "authors": row["authors"],
                "year": row["year"],
                "citation_authority": row["citation_authority"],
                "cited_by_count": row["cited_by_count"],
            }
            for row in rows
        ]
