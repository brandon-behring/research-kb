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
    st.title("📚 Research-KB Knowledge Explorer")
    st.markdown("*Explore causal inference literature through citations and search*")

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select View",
            [
                "📊 Citation Network",
                "🔍 Search",
                "🧠 Concept Graph",
                "📋 Extraction Queue",
            ],
            index=0,
        )

        st.divider()
        st.header("Statistics")

        # Load stats
        try:
            stats = run_async(get_stats())
            st.metric("Sources", stats["sources"])
            st.metric("Chunks", f"{stats['chunks']:,}")
            st.metric("Citations", f"{stats['citations']:,}")
            st.metric("Internal Edges", stats["edges"])
            st.metric("Concepts", stats["concepts"])

        except httpx.ConnectError:
            st.error("Cannot connect to API server. Ensure the API is running.")
            st.caption("Start with: uvicorn research_kb_api.main:app --host 0.0.0.0 --port 8000")
        except Exception as e:
            st.error(f"Could not load stats: {e}")

    # Main content area
    if page == "📊 Citation Network":
        render_citation_network()
    elif page == "🔍 Search":
        render_search()
    elif page == "🧠 Concept Graph":
        render_concept_graph()
    elif page == "📋 Extraction Queue":
        render_queue()


def render_citation_network():
    """Render the citation network visualization."""
    from research_kb_dashboard.pages.citations import citation_network_page

    citation_network_page()


def render_queue():
    """Render the extraction queue status page."""
    from research_kb_dashboard.pages.queue import queue_page

    queue_page()


def render_search():
    """Render the search interface."""
    from research_kb_dashboard.pages.search import search_page

    search_page()


async def get_concept_count():
    """Get concept count for progress display via API."""
    client = ResearchKBClient()
    try:
        stats = await client.get_stats()
        return stats.get("concepts", 0)
    finally:
        await client.close()


def render_concept_graph():
    """Render the concept graph explorer."""
    st.header("🧠 Concept Graph Explorer")
    st.info(
        "**Concept graph is rebuilding.** "
        "The knowledge graph was reset and is being re-extracted. "
        "This view will be available once concepts > 1000."
    )

    # Show current count
    try:
        count = run_async(get_concept_count())
        st.metric("Current Concepts", count)
        st.progress(min(count / 1000, 1.0))
        st.caption("Need 1,000 concepts to enable this view")

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
