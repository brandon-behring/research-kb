"""Search Interface Page.

Full hybrid search with FTS + vector + citation authority boosting.
Results include chunk previews and source information.
"""

import os
import streamlit as st
import asyncio
import asyncpg
from typing import Optional


def _get_db_password() -> str:
    """Get database password from environment."""
    return os.environ.get("POSTGRES_PASSWORD", "postgres")


def run_async(coro):
    """Run async function in Streamlit context."""
    return asyncio.run(coro)


async def search_chunks(
    query_text: str,
    limit: int = 20,
    fts_weight: float = 0.3,
    vector_weight: float = 0.7,
    citation_weight: float = 0.1,
    source_type_filter: Optional[str] = None,
):
    """Search for relevant chunks using hybrid search.

    Args:
        query_text: User's search query
        limit: Maximum results to return
        fts_weight: Weight for full-text search score
        vector_weight: Weight for vector similarity score
        citation_weight: Weight for citation authority
        source_type_filter: Optional filter by source type

    Returns:
        List of search results with scores
    """
    from research_kb_pdf_tools import EmbeddingClient

    # Generate embedding for query (uses BGE query instruction prefix)
    embedding_client = EmbeddingClient()
    query_embedding = embedding_client.embed_query(query_text)

    # Create fresh connection (avoid global pool cache issues with event loops)
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password=_get_db_password(),
    )

    try:
        # Build query based on filters
        if source_type_filter and source_type_filter != "All":
            results = await conn.fetch("""
                WITH fts_search AS (
                    SELECT c.id, c.content, c.source_id, c.chunk_index,
                           ts_rank_cd(c.fts_vector, plainto_tsquery('english', $1)) as fts_score
                    FROM chunks c
                    JOIN sources s ON c.source_id = s.id
                    WHERE c.fts_vector @@ plainto_tsquery('english', $1)
                      AND s.source_type = $5
                ),
                vector_search AS (
                    SELECT c.id, c.content, c.source_id, c.chunk_index,
                           1 - (c.embedding <=> $2::vector) as vector_score
                    FROM chunks c
                    JOIN sources s ON c.source_id = s.id
                    WHERE s.source_type = $5
                    ORDER BY c.embedding <=> $2::vector
                    LIMIT 100
                ),
                combined AS (
                    SELECT
                        COALESCE(f.id, v.id) as id,
                        COALESCE(f.content, v.content) as content,
                        COALESCE(f.source_id, v.source_id) as source_id,
                        COALESCE(f.chunk_index, v.chunk_index) as chunk_index,
                        COALESCE(f.fts_score, 0) as fts_score,
                        COALESCE(v.vector_score, 0) as vector_score
                    FROM fts_search f
                    FULL OUTER JOIN vector_search v ON f.id = v.id
                )
                SELECT
                    c.id, c.content, c.source_id, c.chunk_index,
                    c.fts_score, c.vector_score,
                    COALESCE(s.citation_authority, 0) as citation_score,
                    (c.fts_score * $3 + c.vector_score * $4 +
                     COALESCE(s.citation_authority, 0) * $6) as combined_score,
                    s.title as source_title,
                    s.source_type,
                    s.authors,
                    s.year
                FROM combined c
                JOIN sources s ON c.source_id = s.id
                ORDER BY combined_score DESC
                LIMIT $7
            """, query_text, query_embedding, fts_weight, vector_weight,
                source_type_filter.lower(), citation_weight, limit)
        else:
            results = await conn.fetch("""
                WITH fts_search AS (
                    SELECT c.id, c.content, c.source_id, c.chunk_index,
                           ts_rank_cd(c.fts_vector, plainto_tsquery('english', $1)) as fts_score
                    FROM chunks c
                    WHERE c.fts_vector @@ plainto_tsquery('english', $1)
                ),
                vector_search AS (
                    SELECT c.id, c.content, c.source_id, c.chunk_index,
                           1 - (c.embedding <=> $2::vector) as vector_score
                    FROM chunks c
                    ORDER BY c.embedding <=> $2::vector
                    LIMIT 100
                ),
                combined AS (
                    SELECT
                        COALESCE(f.id, v.id) as id,
                        COALESCE(f.content, v.content) as content,
                        COALESCE(f.source_id, v.source_id) as source_id,
                        COALESCE(f.chunk_index, v.chunk_index) as chunk_index,
                        COALESCE(f.fts_score, 0) as fts_score,
                        COALESCE(v.vector_score, 0) as vector_score
                    FROM fts_search f
                    FULL OUTER JOIN vector_search v ON f.id = v.id
                )
                SELECT
                    c.id, c.content, c.source_id, c.chunk_index,
                    c.fts_score, c.vector_score,
                    COALESCE(s.citation_authority, 0) as citation_score,
                    (c.fts_score * $3 + c.vector_score * $4 +
                     COALESCE(s.citation_authority, 0) * $5) as combined_score,
                    s.title as source_title,
                    s.source_type,
                    s.authors,
                    s.year
                FROM combined c
                JOIN sources s ON c.source_id = s.id
                ORDER BY combined_score DESC
                LIMIT $6
            """, query_text, query_embedding, fts_weight, vector_weight,
                citation_weight, limit)
    finally:
        await conn.close()

    return results


def search_page():
    """Render the search interface page."""
    st.header("🔍 Hybrid Search")
    st.markdown(
        "Search the knowledge base using full-text + semantic similarity. "
        "Results are ranked by combined FTS, vector, and citation authority scores."
    )

    # Search input
    query = st.text_input(
        "Search Query",
        placeholder="e.g., instrumental variables, difference-in-differences, causal forests",
        help="Enter keywords or a natural language question",
    )

    # Advanced options
    with st.expander("Search Options"):
        col1, col2 = st.columns(2)

        with col1:
            source_type = st.selectbox(
                "Source Type",
                ["All", "Paper", "Textbook"],
                index=0,
            )
            limit = st.slider(
                "Max Results",
                min_value=5,
                max_value=50,
                value=20,
            )

        with col2:
            fts_weight = st.slider(
                "FTS Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                help="Weight for keyword/full-text match",
            )
            vector_weight = st.slider(
                "Vector Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Weight for semantic similarity",
            )
            citation_weight = st.slider(
                "Citation Weight",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                help="Weight for source authority (PageRank)",
            )

    # Execute search
    if query:
        with st.spinner("Searching..."):
            results = run_async(search_chunks(
                query_text=query,
                limit=limit,
                fts_weight=fts_weight,
                vector_weight=vector_weight,
                citation_weight=citation_weight,
                source_type_filter=source_type,
            ))

        if not results:
            st.warning("No results found. Try different keywords.")
            return

        st.success(f"Found **{len(results)}** results")

        # Display results
        for i, result in enumerate(results):
            with st.container():
                # Header with source info
                source_type_icon = "📄" if result["source_type"] == "paper" else "📚"
                authors = result["authors"] or []
                author_str = ", ".join(authors[:2])
                if len(authors) > 2:
                    author_str += " et al."

                st.markdown(
                    f"### {i + 1}. {source_type_icon} {result['source_title'][:80]}"
                )
                st.caption(
                    f"*{author_str}* ({result['year'] or 'N/A'}) | "
                    f"Chunk {result['chunk_index']} | "
                    f"Score: {result['combined_score']:.4f}"
                )

                # Content preview
                content = result["content"] or ""
                preview = content[:500]
                if len(content) > 500:
                    preview += "..."

                st.text_area(
                    "Content",
                    value=preview,
                    height=120,
                    disabled=True,
                    key=f"content_{i}",
                    label_visibility="collapsed",
                )

                # Score breakdown
                col1, col2, col3 = st.columns(3)
                col1.metric("FTS", f"{result['fts_score']:.4f}")
                col2.metric("Vector", f"{result['vector_score']:.4f}")
                col3.metric("Citation", f"{result['citation_score']:.4f}")

                st.divider()
    else:
        st.info("Enter a search query above to find relevant content.")

        # Example queries
        st.markdown("**Example queries:**")
        example_queries = [
            "instrumental variables",
            "difference-in-differences parallel trends",
            "double machine learning",
            "causal forests heterogeneous treatment effects",
            "propensity score matching",
        ]
        for eq in example_queries:
            if st.button(eq, key=f"example_{eq}"):
                st.session_state["search_query"] = eq
                st.rerun()
