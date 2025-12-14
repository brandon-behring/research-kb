"""Extraction Queue Status Page.

Shows extraction progress by source, sorted small-to-large for quick wins.
Provides commands to run extraction on specific sources.
"""

import streamlit as st
import asyncio
import asyncpg
import pandas as pd
from datetime import timedelta


def run_async(coro):
    """Run async function in Streamlit context."""
    return asyncio.run(coro)


async def get_extraction_status():
    """Get extraction status for all sources.

    Returns:
        Tuple of (overall_stats dict, list of per-source stats)
    """
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password="postgres",
    )

    try:
        # Overall stats
        total_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        total_concepts = await conn.fetchval("SELECT COUNT(*) FROM concepts")
        chunk_concept_links = await conn.fetchval("SELECT COUNT(*) FROM chunk_concepts")

        processed_chunks = await conn.fetchval("""
            SELECT COUNT(DISTINCT chunk_id) FROM chunk_concepts
        """)

        overall = {
            "total_chunks": total_chunks,
            "processed_chunks": processed_chunks,
            "remaining_chunks": total_chunks - processed_chunks,
            "percent_complete": (processed_chunks / total_chunks * 100) if total_chunks > 0 else 0,
            "total_concepts": total_concepts,
            "chunk_concept_links": chunk_concept_links,
        }

        # Per-source stats (sorted by remaining chunks ascending = small first)
        per_source = await conn.fetch("""
            SELECT
                s.id,
                s.title,
                s.source_type,
                COUNT(DISTINCT c.id) as total_chunks,
                COUNT(DISTINCT cc.chunk_id) as processed_chunks
            FROM sources s
            LEFT JOIN chunks c ON s.id = c.source_id
            LEFT JOIN chunk_concepts cc ON c.id = cc.chunk_id
            GROUP BY s.id, s.title, s.source_type
            HAVING COUNT(DISTINCT c.id) > 0
            ORDER BY (COUNT(DISTINCT c.id) - COUNT(DISTINCT cc.chunk_id)) ASC,
                     COUNT(DISTINCT c.id) ASC
        """)

    finally:
        await conn.close()

    return overall, per_source


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
        "Track concept extraction progress by source. "
        "Sources sorted by remaining chunks (smallest first for quick wins)."
    )

    # Load data
    with st.spinner("Loading extraction status..."):
        overall, per_source = run_async(get_extraction_status())

    # Overall progress section
    st.subheader("Overall Progress")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunks", f"{overall['total_chunks']:,}")
    col2.metric("Processed", f"{overall['processed_chunks']:,}")
    col3.metric("Remaining", f"{overall['remaining_chunks']:,}")
    col4.metric("Complete", f"{overall['percent_complete']:.1f}%")

    # Progress bar
    st.progress(overall['percent_complete'] / 100)

    # ETA calculation (assuming ~1.5 chunks/min throughput)
    throughput = 1.5  # chunks per minute
    eta_minutes = overall['remaining_chunks'] / throughput if throughput > 0 else 0
    st.caption(
        f"📊 At {throughput} chunks/min: **{format_time(eta_minutes)}** remaining "
        f"| Concepts: {overall['total_concepts']:,} | Links: {overall['chunk_concept_links']:,}"
    )

    st.divider()

    # Quick wins section
    quick_wins = [
        r for r in per_source
        if (r['total_chunks'] - r['processed_chunks']) > 0
        and (r['total_chunks'] - r['processed_chunks']) <= 50
    ]

    if quick_wins:
        with st.expander(f"⚡ Quick Wins ({len(quick_wins)} sources with ≤50 chunks remaining)", expanded=True):
            for source in quick_wins[:10]:
                remaining = source['total_chunks'] - source['processed_chunks']
                title = source['title'][:60] if source['title'] else "Untitled"
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{title}** ({remaining} remaining)")
                with col2:
                    cmd = f"python scripts/extract_concepts.py --source-id {source['id']}"
                    st.code(cmd, language="bash")

    st.divider()

    # Filter controls
    col1, col2, col3 = st.columns(3)

    with col1:
        source_type_filter = st.selectbox(
            "Filter by Type",
            ["All", "paper", "textbook"],
            index=0,
        )

    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Incomplete", "Complete", "Not Started"],
            index=1,  # Default to incomplete
        )

    with col3:
        sort_order = st.selectbox(
            "Sort Order",
            ["Remaining (asc)", "Remaining (desc)", "Total (asc)", "Total (desc)"],
            index=0,
        )

    # Build filtered dataframe
    data = []
    for source in per_source:
        remaining = source['total_chunks'] - source['processed_chunks']
        progress_pct = (source['processed_chunks'] / source['total_chunks'] * 100) if source['total_chunks'] > 0 else 0

        # Apply filters
        if source_type_filter != "All" and source['source_type'] != source_type_filter:
            continue

        if status_filter == "Incomplete" and remaining == 0:
            continue
        elif status_filter == "Complete" and remaining > 0:
            continue
        elif status_filter == "Not Started" and source['processed_chunks'] > 0:
            continue

        data.append({
            "id": str(source['id']),
            "title": source['title'][:50] if source['title'] else "Untitled",
            "type": source['source_type'],
            "total": source['total_chunks'],
            "done": source['processed_chunks'],
            "remaining": remaining,
            "progress": progress_pct,
        })

    # Apply sort
    if data:
        df = pd.DataFrame(data)

        if sort_order == "Remaining (asc)":
            df = df.sort_values("remaining", ascending=True)
        elif sort_order == "Remaining (desc)":
            df = df.sort_values("remaining", ascending=False)
        elif sort_order == "Total (asc)":
            df = df.sort_values("total", ascending=True)
        elif sort_order == "Total (desc)":
            df = df.sort_values("total", ascending=False)

        # Display table
        st.subheader(f"Sources ({len(df)} shown)")

        # Show as interactive table
        st.dataframe(
            df[["title", "type", "total", "done", "remaining", "progress"]].rename(columns={
                "title": "Source",
                "type": "Type",
                "total": "Total",
                "done": "Done",
                "remaining": "Remaining",
                "progress": "Progress %",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Selected source details
        st.subheader("Run Extraction")

        # Show command for first incomplete source
        incomplete = df[df["remaining"] > 0]
        if not incomplete.empty:
            first_incomplete = incomplete.iloc[0]
            st.markdown(f"**Next up:** {first_incomplete['title']} ({first_incomplete['remaining']} chunks)")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Process this source:**")
                st.code(
                    f"python scripts/extract_concepts.py --source-id {first_incomplete['id']}",
                    language="bash"
                )
            with col2:
                st.markdown("**Resume all (auto-selects unprocessed):**")
                st.code("python scripts/extract_concepts.py --resume", language="bash")
        else:
            st.success("🎉 All sources have been processed!")

    else:
        st.info("No sources match the current filters.")

    # Footer with stats
    st.divider()
    st.caption(
        f"Throughput assumption: {throughput} chunks/min "
        "(based on Ollama llama3.1:8b on RTX 2070 SUPER)"
    )
