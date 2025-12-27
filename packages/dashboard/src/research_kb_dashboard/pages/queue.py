"""Extraction Queue Status Page.

Shows extraction progress by source, sorted small-to-large for quick wins.
Provides commands to run extraction on specific sources.

Uses the Research-KB API for data access.

NOTE: Per-source extraction status currently requires direct database access
because the API doesn't expose chunk_concepts per source. This is a known
limitation that could be addressed with a /extraction-status API endpoint.
"""

import streamlit as st
import asyncio
import httpx
import pandas as pd

from research_kb_dashboard.api_client import ResearchKBClient


def run_async(coro):
    """Run async function in Streamlit context."""
    return asyncio.run(coro)


async def get_extraction_status():
    """Get extraction status via API.

    Returns:
        Tuple of (overall_stats dict, list of per-source stats)

    NOTE: The API provides overall stats but not per-source extraction status.
    Per-source stats are derived from source metadata which may not include
    extraction progress details. For full per-source extraction tracking,
    a dedicated /extraction-status endpoint would be needed.
    """
    client = ResearchKBClient()
    try:
        # Get overall stats from API
        stats = await client.get_stats()

        # Calculate overall extraction progress
        # The API returns chunk_concepts count which represents processed chunks
        total_chunks = stats.get("chunks", 0)
        total_concepts = stats.get("concepts", 0)
        chunk_concept_links = stats.get("chunk_concepts", 0)

        # Estimate processed chunks from chunk_concepts links
        # NOTE: This is an approximation - ideally the API would return
        # a distinct count of chunk_ids in chunk_concepts
        # For now, assume ~3 concepts per chunk on average
        processed_chunks = min(chunk_concept_links // 3, total_chunks) if chunk_concept_links > 0 else 0

        overall = {
            "total_chunks": total_chunks,
            "processed_chunks": processed_chunks,
            "remaining_chunks": total_chunks - processed_chunks,
            "percent_complete": (processed_chunks / total_chunks * 100) if total_chunks > 0 else 0,
            "total_concepts": total_concepts,
            "chunk_concept_links": chunk_concept_links,
        }

        # Get sources for per-source display
        # NOTE: Without a dedicated extraction status endpoint, we can only
        # show basic source info. Chunk counts require the SourceWithChunks endpoint
        # which would be too slow to call for all sources.
        sources_response = await client.list_sources(limit=1000)

        per_source = []
        for s in sources_response.get("sources", []):
            # We don't have per-source chunk counts without additional API calls
            # Show placeholder data that indicates limitation
            per_source.append({
                "id": s.get("id"),
                "title": s.get("title", "Untitled"),
                "source_type": s.get("source_type", "paper"),
                "total_chunks": None,  # Not available from list endpoint
                "processed_chunks": None,  # Not available from API
            })

        return overall, per_source
    finally:
        await client.close()


def format_time(minutes: float) -> str:
    """Format minutes as human-readable time string."""
    if minutes < 60:
        return f"{int(minutes)} min"
    elif minutes < 1440:  # Less than 24 hours
        hours = minutes / 60
        return f"{hours:.1f} hours"
    else:
        days = minutes / 1440
        return f"{days:.1f} days"


def queue_page():
    """Render the extraction queue status page."""
    st.header("📋 Extraction Queue")
    st.markdown(
        "Track concept extraction progress. "
        "Shows overall progress and commands to run extraction."
    )

    # Load data
    try:
        with st.spinner("Loading extraction status..."):
            overall, per_source = run_async(get_extraction_status())
    except httpx.ConnectError:
        st.error("Cannot connect to API server. Ensure the API is running.")
        st.caption("Start with: uvicorn research_kb_api.main:app --host 0.0.0.0 --port 8000")
        return
    except Exception as e:
        st.error(f"Failed to load extraction status: {e}")
        return

    # Overall progress section
    st.subheader("Overall Progress")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunks", f"{overall['total_chunks']:,}")
    col2.metric("Processed (est.)", f"{overall['processed_chunks']:,}")
    col3.metric("Remaining (est.)", f"{overall['remaining_chunks']:,}")
    col4.metric("Complete (est.)", f"{overall['percent_complete']:.1f}%")

    # Progress bar
    st.progress(min(overall['percent_complete'] / 100, 1.0))

    # ETA calculation (assuming ~1.5 chunks/min throughput)
    throughput = 1.5  # chunks per minute
    eta_minutes = overall['remaining_chunks'] / throughput if throughput > 0 else 0
    st.caption(
        f"📊 At {throughput} chunks/min: **{format_time(eta_minutes)}** remaining "
        f"| Concepts: {overall['total_concepts']:,} | Links: {overall['chunk_concept_links']:,}"
    )

    st.divider()

    # Per-source status notice
    st.info(
        "**Note:** Per-source extraction status requires a dedicated API endpoint. "
        "The current API provides overall stats but not per-source chunk/extraction counts. "
        "Use the CLI for detailed per-source status: `research-kb extraction-status`"
    )

    # Source list (without chunk counts)
    st.subheader("Sources")

    # Filter controls
    col1, col2 = st.columns(2)

    with col1:
        source_type_filter = st.selectbox(
            "Filter by Type",
            ["All", "paper", "textbook"],
            index=0,
        )

    # Build filtered dataframe
    data = []
    for source in per_source:
        # Apply filter
        if source_type_filter != "All" and source['source_type'] != source_type_filter:
            continue

        data.append({
            "id": str(source['id']),
            "title": source['title'][:60] if source['title'] else "Untitled",
            "type": source['source_type'],
        })

    if data:
        df = pd.DataFrame(data)

        # Display table (limited info without chunk counts)
        st.dataframe(
            df[["title", "type"]].rename(columns={
                "title": "Source",
                "type": "Type",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Run Extraction")

        # Show general extraction commands
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Process specific source:**")
            if len(data) > 0:
                st.code(
                    f"python scripts/extract_concepts.py --source-id {data[0]['id']}",
                    language="bash"
                )
        with col2:
            st.markdown("**Resume all (auto-selects unprocessed):**")
            st.code("python scripts/extract_concepts.py --resume", language="bash")

    else:
        st.info("No sources match the current filters.")

    # Footer with stats
    st.divider()
    st.caption(
        f"Throughput assumption: {throughput} chunks/min "
        "(based on Ollama llama3.1:8b on RTX 2070 SUPER)"
    )
