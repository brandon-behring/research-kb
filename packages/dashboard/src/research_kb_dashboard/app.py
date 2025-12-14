"""Research-KB Knowledge Explorer Dashboard.

Main Streamlit application entry point. Run with:
    streamlit run packages/dashboard/src/research_kb_dashboard/app.py
"""

import streamlit as st
import asyncio
import asyncpg

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
    """Get database statistics."""
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password="postgres",
    )
    try:
        sources = await conn.fetchval("SELECT COUNT(*) FROM sources")
        chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        citations = await conn.fetchval("SELECT COUNT(*) FROM citations")
        concepts = await conn.fetchval("SELECT COUNT(*) FROM concepts")
        edges = await conn.fetchval(
            "SELECT COUNT(*) FROM source_citations WHERE cited_source_id IS NOT NULL"
        )
    finally:
        await conn.close()

    return {
        "sources": sources,
        "chunks": chunks,
        "citations": citations,
        "concepts": concepts,
        "edges": edges,
    }


def main():
    """Main dashboard entry point."""
    st.title("📚 Research-KB Knowledge Explorer")
    st.markdown("*Explore causal inference literature through citations and search*")

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select View",
            ["📊 Citation Network", "🔍 Search", "🧠 Concept Graph", "📋 Extraction Queue"],
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
    """Get concept count for progress display."""
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password="postgres",
    )
    try:
        return await conn.fetchval("SELECT COUNT(*) FROM concepts")
    finally:
        await conn.close()


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
