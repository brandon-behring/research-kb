"""Search Interface Page.

Full hybrid search with FTS + vector + citation authority boosting.
Results include chunk previews and source information.

Uses the Research-KB API for search, which provides:
- Hybrid search (FTS + vector + graph + citation signals)
- Context-aware weighting
- Cross-encoder reranking
- Query expansion
"""

import streamlit as st
import asyncio
import httpx
from typing import Optional

from research_kb_dashboard.api_client import ResearchKBClient


def run_async(coro):
    """Run async function in Streamlit context."""
    return asyncio.run(coro)


async def search_chunks(
    query_text: str,
    limit: int = 20,
    context_type: str = "balanced",
    source_type_filter: Optional[str] = None,
):
    """Search for relevant chunks using hybrid search via API.

    Args:
        query_text: User's search query
        limit: Maximum results to return
        context_type: "building", "auditing", or "balanced"
        source_type_filter: Optional filter by source type

    Returns:
        List of search results in API format
    """
    client = ResearchKBClient()
    try:
        # Map source filter for API
        api_source_filter = None
        if source_type_filter and source_type_filter != "All":
            api_source_filter = source_type_filter.upper()

        response = await client.search(
            query=query_text,
            limit=limit,
            context_type=context_type,
            source_filter=api_source_filter,
            use_graph=True,
            use_rerank=True,
            use_expand=True,
        )
        return response.get("results", [])
    finally:
        await client.close()


def search_page():
    """Render the search interface page."""
    st.header("🔍 Hybrid Search")
    st.markdown(
        "Search the knowledge base using full-text + semantic similarity + knowledge graph. "
        "Results are ranked by combined FTS, vector, graph, and citation authority scores."
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
            context_type = st.selectbox(
                "Search Context",
                ["balanced", "building", "auditing"],
                index=0,
                help="building: favor semantic breadth (80% vector), "
                     "auditing: favor precision (50/50), "
                     "balanced: default (70% vector)",
            )

    # Execute search
    if query:
        try:
            with st.spinner("Searching..."):
                results = run_async(search_chunks(
                    query_text=query,
                    limit=limit,
                    context_type=context_type,
                    source_type_filter=source_type,
                ))

            if not results:
                st.warning("No results found. Try different keywords.")
                return

            st.success(f"Found **{len(results)}** results")

            # Display results
            for i, result in enumerate(results):
                with st.container():
                    # Extract nested source/chunk data from API response
                    source = result.get("source", {})
                    chunk = result.get("chunk", {})
                    scores = result.get("scores", {})

                    # Header with source info
                    source_type_val = source.get("source_type", "paper")
                    source_type_icon = "📄" if source_type_val == "paper" else "📚"
                    authors = source.get("authors") or []
                    author_str = ", ".join(authors[:2])
                    if len(authors) > 2:
                        author_str += " et al."

                    title = source.get("title", "Untitled")[:80]
                    st.markdown(f"### {i + 1}. {source_type_icon} {title}")
                    st.caption(
                        f"*{author_str}* ({source.get('year') or 'N/A'}) | "
                        f"Score: {result.get('combined_score', 0):.4f}"
                    )

                    # Content preview
                    content = chunk.get("content", "")
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
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("FTS", f"{scores.get('fts', 0):.4f}")
                    col2.metric("Vector", f"{scores.get('vector', 0):.4f}")
                    col3.metric("Graph", f"{scores.get('graph', 0):.4f}")
                    col4.metric("Citation", f"{scores.get('citation', 0):.4f}")

                    st.divider()

        except httpx.ConnectError:
            st.error("Cannot connect to API server. Ensure the API is running.")
            st.caption("Start with: uvicorn research_kb_api.main:app --host 0.0.0.0 --port 8000")
        except Exception as e:
            st.error(f"Search error: {e}")
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
