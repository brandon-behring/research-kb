"""Citation Network Visualization Page.

Interactive PyVis graph showing paper/textbook citation relationships.
Nodes sized by PageRank authority, colored by source type.
"""

import os
import streamlit as st
import asyncio
import asyncpg
from typing import Optional


def _get_db_password() -> str:
    """Get database password from environment."""
    return os.environ.get("POSTGRES_PASSWORD", "postgres")

from research_kb_dashboard.components.graph import (
    create_network,
    render_network,
    get_node_color,
    get_node_size,
    truncate_title,
)


def run_async(coro):
    """Run async function in Streamlit context."""
    return asyncio.run(coro)


async def load_citation_data(source_type_filter: Optional[str] = None):
    """Load sources and citation edges from database.

    Args:
        source_type_filter: Optional filter by source type

    Returns:
        Tuple of (sources list, edges list)
    """
    # Create fresh connection (avoid global pool cache issues with event loops)
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password=_get_db_password(),
    )

    try:
        # Load sources
        if source_type_filter and source_type_filter != "All":
            sources = await conn.fetch("""
                SELECT id, source_type, title, authors, year,
                       COALESCE(citation_authority, 0.1) as authority
                FROM sources
                WHERE source_type = $1
                ORDER BY citation_authority DESC NULLS LAST
            """, source_type_filter.lower())
        else:
            sources = await conn.fetch("""
                SELECT id, source_type, title, authors, year,
                       COALESCE(citation_authority, 0.1) as authority
                FROM sources
                ORDER BY citation_authority DESC NULLS LAST
            """)

        # Get source IDs for edge filtering
        source_ids = [str(s["id"]) for s in sources]

        # Load edges (only internal citations between corpus sources)
        edges = await conn.fetch("""
            SELECT citing_source_id, cited_source_id
            FROM source_citations
            WHERE cited_source_id IS NOT NULL
        """)

        # Filter edges to only include sources in our filtered set
        if source_type_filter and source_type_filter != "All":
            edges = [
                e for e in edges
                if str(e["citing_source_id"]) in source_ids
                and str(e["cited_source_id"]) in source_ids
            ]
    finally:
        await conn.close()

    return sources, edges


def citation_network_page():
    """Render the citation network visualization page."""
    st.header("📊 Citation Network")
    st.markdown(
        "Interactive visualization of citation relationships between sources. "
        "Node size = PageRank authority. "
        "Hover for details, scroll to zoom, drag to pan."
    )

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        source_type = st.selectbox(
            "Filter by Type",
            ["All", "Paper", "Textbook"],
            index=0,
        )

    with col2:
        min_authority = st.slider(
            "Min Authority Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Filter out sources below this PageRank authority score",
        )

    with col3:
        show_isolated = st.checkbox(
            "Show Isolated Nodes",
            value=True,
            help="Show sources with no citations",
        )

    # Load data
    with st.spinner("Loading citation network..."):
        sources, edges = run_async(load_citation_data(source_type))

    # Filter by authority
    if min_authority > 0:
        sources = [s for s in sources if s["authority"] >= min_authority]

    # Build node set for filtering isolated
    citing_ids = {str(e["citing_source_id"]) for e in edges}
    cited_ids = {str(e["cited_source_id"]) for e in edges}
    connected_ids = citing_ids | cited_ids

    if not show_isolated:
        sources = [s for s in sources if str(s["id"]) in connected_ids]

    # Stats
    st.info(f"Showing **{len(sources)}** sources and **{len(edges)}** citation edges")

    if len(sources) == 0:
        st.warning("No sources match the current filters.")
        return

    # Build network
    net = create_network(height="650px", directed=True)

    # Add nodes
    source_id_set = {str(s["id"]) for s in sources}

    for source in sources:
        source_id = str(source["id"])
        title = source["title"] or "Untitled"
        source_type_val = source["source_type"] or "unknown"
        authority = float(source["authority"])
        year = source["year"] or "N/A"
        authors = source["authors"] or []
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."

        # Tooltip (plain text - HTML gets escaped by vis.js)
        tooltip = f"{title}\n\n{author_str}\nYear: {year}\nType: {source_type_val}\nAuthority: {authority:.3f}"

        net.add_node(
            source_id,
            label=truncate_title(title, 30),
            title=tooltip,
            size=get_node_size(authority),
            color=get_node_color(source_type_val),
            shape="dot",
        )

    # Add edges (only between nodes in our set)
    for edge in edges:
        citing_id = str(edge["citing_source_id"])
        cited_id = str(edge["cited_source_id"])

        if citing_id in source_id_set and cited_id in source_id_set:
            net.add_edge(citing_id, cited_id)

    # Render
    render_network(net, key="citation_network")

    # Legend
    with st.expander("Legend"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Node Colors:**")
            st.markdown("🔵 Paper")
            st.markdown("🟢 Textbook")
            st.markdown("🟠 Code Repository")
        with col2:
            st.markdown("**Node Size:**")
            st.markdown("Larger = Higher PageRank authority")
            st.markdown("(more citations from authoritative sources)")

    # Top sources table
    with st.expander("Top Sources by Authority"):
        import pandas as pd

        top_sources = sorted(sources, key=lambda x: x["authority"], reverse=True)[:20]
        df = pd.DataFrame([
            {
                "Title": s["title"][:60] + "..." if len(s["title"] or "") > 60 else s["title"],
                "Type": s["source_type"],
                "Year": s["year"],
                "Authority": f"{s['authority']:.4f}",
            }
            for s in top_sources
        ])
        st.dataframe(df, use_container_width=True)
