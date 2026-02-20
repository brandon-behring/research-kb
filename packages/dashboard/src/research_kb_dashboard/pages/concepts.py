"""Concept Graph Explorer page.

Interactive visualization of the knowledge graph with concept search,
domain filtering, and N-hop neighborhood exploration.
"""

import asyncio

import streamlit as st
import httpx

from research_kb_dashboard.api_client import ResearchKBClient
from research_kb_dashboard.components.graph import (
    create_network,
    render_network,
    truncate_title,
)

# Concept type colors
CONCEPT_TYPE_COLORS = {
    "METHOD": "#4299e1",  # blue
    "ASSUMPTION": "#ed8936",  # orange
    "PROBLEM": "#e53e3e",  # red
    "DEFINITION": "#48bb78",  # green
    "THEOREM": "#9f7aea",  # purple
    "CONCEPT": "#a0aec0",  # gray
    "PRINCIPLE": "#38b2ac",  # teal
    "TECHNIQUE": "#667eea",  # indigo
    "MODEL": "#d69e2e",  # yellow
}

# Relationship type edge styles
RELATIONSHIP_COLORS = {
    "REQUIRES": "#e53e3e",
    "USES": "#4299e1",
    "ADDRESSES": "#48bb78",
    "GENERALIZES": "#9f7aea",
    "SPECIALIZES": "#9f7aea",
    "ALTERNATIVE_TO": "#ed8936",
    "EXTENDS": "#667eea",
}


def run_async(coro):
    """Run async function in Streamlit context."""
    return asyncio.run(coro)


async def search_concepts(query: str, concept_type: str | None, limit: int):
    """Search concepts via API."""
    client = ResearchKBClient()
    try:
        return await client.list_concepts(
            query=query,
            concept_type=concept_type,
            limit=limit,
        )
    finally:
        await client.close()


async def get_neighborhood(concept_name: str, hops: int, limit: int):
    """Get concept neighborhood via API."""
    client = ResearchKBClient()
    try:
        return await client.get_graph_neighborhood(
            concept_name=concept_name,
            hops=hops,
            limit=limit,
        )
    finally:
        await client.close()


async def get_path(concept_a: str, concept_b: str):
    """Get shortest path between concepts."""
    client = ResearchKBClient()
    try:
        return await client.get_graph_path(concept_a, concept_b)
    finally:
        await client.close()


def concept_graph_page():
    """Render the concept graph explorer page."""
    st.header("Concept Graph Explorer")
    st.markdown(
        "Explore the knowledge graph: search concepts, visualize neighborhoods, find paths."
    )

    # Controls
    tab1, tab2, tab3 = st.tabs(["Search Concepts", "Neighborhood Explorer", "Path Finder"])

    # --- Tab 1: Concept Search ---
    with tab1:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            query = st.text_input("Search concepts", placeholder="e.g., instrumental variables")
        with col2:
            concept_type = st.selectbox(
                "Type filter",
                [
                    "All",
                    "METHOD",
                    "ASSUMPTION",
                    "PROBLEM",
                    "DEFINITION",
                    "THEOREM",
                    "CONCEPT",
                    "PRINCIPLE",
                    "TECHNIQUE",
                    "MODEL",
                ],
            )
        with col3:
            limit = st.number_input("Max results", min_value=10, max_value=200, value=50)

        if query:
            try:
                type_filter = None if concept_type == "All" else concept_type
                results = run_async(search_concepts(query, type_filter, limit))
                concepts = results.get("concepts", [])

                if not concepts:
                    st.info("No concepts found. Try a broader query.")
                else:
                    st.success(f"Found {len(concepts)} concepts")

                    for concept in concepts:
                        ctype = concept.get("concept_type", "CONCEPT")
                        color = CONCEPT_TYPE_COLORS.get(ctype, "#a0aec0")
                        name = concept.get("name", "")

                        st.markdown(
                            f"**:{color}[{ctype}]** {name}"
                            f" &mdash; {concept.get('description', '')[:200]}"
                        )

            except httpx.ConnectError:
                st.error("Cannot connect to API server.")
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Tab 2: Neighborhood Visualization ---
    with tab2:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            center = st.text_input(
                "Center concept",
                placeholder="e.g., double machine learning",
                key="neighborhood_center",
            )
        with col2:
            hops = st.selectbox("Hops", [1, 2, 3], index=1)
        with col3:
            max_nodes = st.number_input(
                "Max nodes",
                min_value=20,
                max_value=200,
                value=100,
                key="max_nodes",
            )

        if center:
            try:
                with st.spinner("Loading neighborhood..."):
                    data = run_async(get_neighborhood(center, hops, max_nodes))

                nodes = data.get("nodes", [])
                edges = data.get("edges", [])

                if not nodes:
                    st.warning("No neighborhood found. Check concept name exists.")
                else:
                    st.success(f"{len(nodes)} nodes, {len(edges)} edges")

                    # Build PyVis graph
                    net = create_network(height="600px", directed=True)

                    for node in nodes:
                        ctype = node.get("concept_type", "CONCEPT")
                        color = CONCEPT_TYPE_COLORS.get(ctype, "#a0aec0")
                        label = truncate_title(node.get("name", ""), 30)
                        is_center = node.get("name", "").lower() == center.lower()
                        size = 30 if is_center else 15

                        net.add_node(
                            node.get("id", node.get("name", "")),
                            label=label,
                            color=color,
                            size=size,
                            title=f"{ctype}: {node.get('name', '')}\n{node.get('description', '')[:200]}",
                        )

                    for edge in edges:
                        rel_type = edge.get("relationship_type", "RELATED")
                        color = RELATIONSHIP_COLORS.get(rel_type, "#a0aec0")

                        net.add_edge(
                            edge.get("source", ""),
                            edge.get("target", ""),
                            title=rel_type,
                            color=color,
                            width=2,
                        )

                    render_network(net, key="concept_neighborhood")

                    # Legend
                    st.markdown(
                        "**Legend:** "
                        + " | ".join(
                            f"<span style='color:{color}'>{ctype}</span>"
                            for ctype, color in CONCEPT_TYPE_COLORS.items()
                        ),
                        unsafe_allow_html=True,
                    )

            except httpx.ConnectError:
                st.error("Cannot connect to API server.")
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Tab 3: Path Finder ---
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            concept_a = st.text_input("From concept", placeholder="e.g., IV", key="path_from")
        with col2:
            concept_b = st.text_input(
                "To concept", placeholder="e.g., unconfoundedness", key="path_to"
            )

        if concept_a and concept_b:
            try:
                with st.spinner("Finding path..."):
                    data = run_async(get_path(concept_a, concept_b))

                path = data.get("path", [])
                if not path:
                    st.warning("No path found between these concepts.")
                else:
                    st.success(f"Path length: {data.get('path_length', len(path))}")

                    # Show path as a chain
                    path_str = " -> ".join(
                        step.get("name", str(step)) if isinstance(step, dict) else str(step)
                        for step in path
                    )
                    st.code(path_str)

            except httpx.ConnectError:
                st.error("Cannot connect to API server.")
            except Exception as e:
                st.error(f"Error: {e}")
