"""Research-KB Knowledge Explorer Dashboard.

Main Streamlit application entry point. Run with:
    streamlit run packages/dashboard/src/research_kb_dashboard/app.py

Requires the Research-KB API server to be running at RESEARCH_KB_API_URL
(default: http://localhost:8000).
"""

import streamlit as st
import asyncio
import httpx

from research_kb_dashboard.api_client import ResearchKBClient

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Research-KB Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def run_async(coro):
    """Run async function in Streamlit context."""
    return asyncio.run(coro)


async def get_stats():
    """Get database statistics via API."""
    client = ResearchKBClient()
    try:
        stats = await client.get_stats()
        # Map API response fields to expected format
        # API returns: sources, chunks, concepts, relationships, citations, chunk_concepts
        return {
            "sources": stats.get("sources", 0),
            "chunks": stats.get("chunks", 0),
            "citations": stats.get("citations", 0),
            "concepts": stats.get("concepts", 0),
            "edges": stats.get("relationships", 0),  # relationships = internal edges
        }
    finally:
        await client.close()


def main():
    """Main dashboard entry point."""
    st.title("Research-KB Knowledge Explorer")
    st.markdown("*Explore research literature through search, citations, and knowledge graphs*")

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select View",
            [
                "Search",
                "Citation Network",
                "Concept Graph",
                "Assumption Audit",
                "Statistics",
                "Extraction Queue",
            ],
            index=0,
        )

        st.divider()
        st.header("Corpus")

        # Load stats
        try:
            stats = run_async(get_stats())
            st.metric("Sources", stats["sources"])
            st.metric("Chunks", f"{stats['chunks']:,}")
            st.metric("Concepts", f"{stats['concepts']:,}")
            st.metric("Relationships", f"{stats['edges']:,}")
            st.metric("Citations", f"{stats['citations']:,}")

        except httpx.ConnectError:
            st.error("Cannot connect to API server. Ensure the API is running.")
            st.caption("Start with: uvicorn research_kb_api.main:create_app --factory --port 8000")
        except Exception as e:
            st.error(f"Could not load stats: {e}")

    # Main content area
    if page == "Search":
        from research_kb_dashboard.pages.search import search_page

        search_page()
    elif page == "Citation Network":
        from research_kb_dashboard.pages.citations import citation_network_page

        citation_network_page()
    elif page == "Concept Graph":
        from research_kb_dashboard.pages.concepts import concept_graph_page

        concept_graph_page()
    elif page == "Assumption Audit":
        from research_kb_dashboard.pages.assumptions import assumptions_page

        assumptions_page()
    elif page == "Statistics":
        from research_kb_dashboard.pages.statistics import statistics_page

        statistics_page()
    elif page == "Extraction Queue":
        from research_kb_dashboard.pages.queue import queue_page

        queue_page()


if __name__ == "__main__":
    main()
