"""Citation Network Visualization Page.

Interactive PyVis graph showing paper/textbook citation relationships.
Nodes sized by PageRank authority, colored by source type.

Uses the Research-KB API for data access.
"""

import streamlit as st
import asyncio
import httpx
from typing import Optional

from research_kb_dashboard.api_client import ResearchKBClient

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
    """Load sources and citation edges via API.

    Args:
        source_type_filter: Optional filter by source type

    Returns:
        Tuple of (sources list, edges list)
    """
    client = ResearchKBClient()
    try:
        # Map source filter for API
        api_source_filter = None
        if source_type_filter and source_type_filter != "All":
            api_source_filter = source_type_filter.upper()

        # Load sources via API
        response = await client.list_sources(
            limit=1000,  # Get all for visualization
            source_type=api_source_filter,
        )

        # Convert API response to expected format
        sources = []
        for s in response.get("sources", []):
            sources.append({
                "id": s.get("id"),
                "source_type": s.get("source_type", "paper"),
                "title": s.get("title", "Untitled"),
                "authors": s.get("authors", []),
                "year": s.get("year"),
                # Use metadata for authority if available, default to 0.1
                "authority": s.get("metadata", {}).get("citation_authority", 0.1)
                             if s.get("metadata") else 0.1,
            })

        # Get source IDs for edge filtering
        source_ids = {s["id"] for s in sources}

        # Load edges by getting citations for each source
        # NOTE: This is less efficient than a direct query, but works with the API
        # For large datasets, consider adding a dedicated /citations endpoint
        edges = []
        for source in sources:
            try:
                citations = await client.get_source_citations(source["id"])
                # Add edges for cited sources (this source cites them)
                for cited in citations.get("cited_sources", []):
                    if cited.get("id") in source_ids:
                        edges.append({
                            "citing_source_id": source["id"],
                            "cited_source_id": cited.get("id"),
                        })
            except Exception:
                # Skip sources with citation errors
                continue

        return sources, edges
    finally:
        await client.close()


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
    try:
        with st.spinner("Loading citation network..."):
            sources, edges = run_async(load_citation_data(source_type))
    except httpx.ConnectError:
        st.error("Cannot connect to API server. Ensure the API is running.")
        st.caption("Start with: uvicorn research_kb_api.main:app --host 0.0.0.0 --port 8000")
        return
    except Exception as e:
        st.error(f"Failed to load citation data: {e}")
        return

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
