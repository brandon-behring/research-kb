"""Corpus Statistics page.

Visualizes corpus composition: domain distribution, concept types,
citation authority, and source timeline.
"""

import asyncio

import plotly.express as px
import streamlit as st
import httpx

from research_kb_dashboard.api_client import ResearchKBClient


def run_async(coro):
    """Run async function in Streamlit context."""
    return asyncio.run(coro)


async def load_stats():
    """Load comprehensive stats from API."""
    client = ResearchKBClient()
    try:
        stats = await client.get_stats()
        sources_data = await client.list_sources(limit=1000)
        return stats, sources_data
    finally:
        await client.close()


def statistics_page():
    """Render the corpus statistics page."""
    st.header("Corpus Statistics")

    try:
        with st.spinner("Loading statistics..."):
            stats, sources_data = run_async(load_stats())
    except httpx.ConnectError:
        st.error("Cannot connect to API server.")
        st.caption("Start with: uvicorn research_kb_api.main:create_app --factory --port 8000")
        return
    except Exception as e:
        st.error(f"Error loading stats: {e}")
        return

    # Top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Sources", f"{stats.get('sources', 0):,}")
    col2.metric("Chunks", f"{stats.get('chunks', 0):,}")
    col3.metric("Concepts", f"{stats.get('concepts', 0):,}")
    col4.metric("Relationships", f"{stats.get('relationships', 0):,}")
    col5.metric("Citations", f"{stats.get('citations', 0):,}")

    st.divider()

    sources = sources_data.get("sources", [])
    if not sources:
        st.info("No sources in database. Ingest some papers first.")
        return

    # --- Domain Distribution ---
    st.subheader("Domain Distribution")

    domain_counts = {}
    for source in sources:
        domain = (source.get("metadata") or {}).get("domain", "untagged")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    domain_sorted = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    domains = [d[0] for d in domain_sorted]
    counts = [d[1] for d in domain_sorted]

    fig_domain = px.bar(
        x=counts,
        y=domains,
        orientation="h",
        labels={"x": "Number of Sources", "y": "Domain"},
        color=counts,
        color_continuous_scale="Viridis",
    )
    fig_domain.update_layout(
        height=max(300, len(domains) * 30),
        showlegend=False,
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_domain, use_container_width=True)

    # --- Source Type Distribution ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Source Types")
        type_counts = {}
        for source in sources:
            stype = source.get("source_type", "unknown")
            type_counts[stype] = type_counts.get(stype, 0) + 1

        fig_types = px.pie(
            names=list(type_counts.keys()),
            values=list(type_counts.values()),
            hole=0.4,
        )
        fig_types.update_layout(height=300)
        st.plotly_chart(fig_types, use_container_width=True)

    with col2:
        st.subheader("Publication Timeline")
        years = [s.get("year") for s in sources if s.get("year")]
        if years:
            year_counts = {}
            for y in years:
                year_counts[y] = year_counts.get(y, 0) + 1

            fig_years = px.bar(
                x=sorted(year_counts.keys()),
                y=[year_counts[y] for y in sorted(year_counts.keys())],
                labels={"x": "Year", "y": "Sources"},
            )
            fig_years.update_layout(height=300)
            st.plotly_chart(fig_years, use_container_width=True)
        else:
            st.info("No year data available.")

    # --- Top Sources by Citation Authority ---
    st.subheader("Top Sources by Citation Count")

    # Sort by citation count if available
    sources_with_citations = [
        s
        for s in sources
        if (s.get("metadata") or {}).get("citation_count") or s.get("citation_count")
    ]

    if sources_with_citations:
        top_cited = sorted(
            sources_with_citations,
            key=lambda s: (s.get("metadata") or {}).get("citation_count", 0)
            or s.get("citation_count", 0),
            reverse=True,
        )[:20]

        for i, source in enumerate(top_cited, 1):
            citations = (source.get("metadata") or {}).get("citation_count", 0) or source.get(
                "citation_count", 0
            )
            title = source.get("title", "Unknown")[:80]
            year = source.get("year", "")
            st.markdown(f"**{i}.** {title} ({year}) -- {citations} citations")
    else:
        st.info(
            "Citation counts not available. Run `research-kb enrich citations` "
            "to add Semantic Scholar metadata."
        )

    # --- Knowledge Graph Stats ---
    if stats.get("concepts", 0) > 0:
        st.divider()
        st.subheader("Knowledge Graph")

        col1, col2, col3 = st.columns(3)
        col1.metric("Concepts", f"{stats.get('concepts', 0):,}")
        col2.metric("Relationships", f"{stats.get('relationships', 0):,}")
        ratio = (
            stats.get("relationships", 0) / stats.get("concepts", 1)
            if stats.get("concepts", 0) > 0
            else 0
        )
        col3.metric("Avg Relationships/Concept", f"{ratio:.1f}")
