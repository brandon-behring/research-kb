"""Assumption Auditing page.

Interactive UI for the North Star feature: explore what assumptions
a causal inference method requires and where they're discussed in the corpus.
"""

import asyncio
import json

import streamlit as st
import httpx

from research_kb_dashboard.api_client import ResearchKBClient


def run_async(coro):
    """Run async function in Streamlit context."""
    return asyncio.run(coro)


async def audit_method(method_name: str):
    """Call the assumption audit endpoint.

    Falls back to CLI if API endpoint not available.
    """
    client = ResearchKBClient()
    try:
        # Try the API endpoint first
        http_client = await client._get_client()
        response = await http_client.get(
            "/audit-assumptions",
            params={"method": method_name},
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # API endpoint not available, try CLI fallback
            return await _cli_fallback(method_name)
        raise
    finally:
        await client.close()


async def _cli_fallback(method_name: str):
    """Fall back to CLI for assumption auditing."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_kb_cli",
            "audit-assumptions",
            method_name,
            "--format",
            "json",
            "--no-ollama",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 0:
        return json.loads(result.stdout)
    return {"error": result.stderr, "method": method_name, "assumptions": []}


async def list_cached_methods():
    """Get methods with cached assumptions."""
    client = ResearchKBClient()
    try:
        http_client = await client._get_client()
        response = await http_client.get(
            "/concepts", params={"concept_type": "METHOD", "limit": 100}
        )
        response.raise_for_status()
        return response.json().get("concepts", [])
    except Exception:
        return []
    finally:
        await client.close()


# Common causal inference methods for quick selection
COMMON_METHODS = [
    "instrumental variables",
    "double machine learning",
    "difference-in-differences",
    "regression discontinuity",
    "synthetic control",
    "propensity score matching",
    "causal forests",
    "inverse probability weighting",
    "mediation analysis",
    "Bayesian structural time series",
]


def assumptions_page():
    """Render the assumption auditing page."""
    st.header("Assumption Auditing")
    st.markdown(
        "Explore the assumptions required by causal inference methods. "
        "Understand *what* you must believe for a method's conclusions to hold."
    )

    # Method selection
    col1, col2 = st.columns([3, 1])
    with col1:
        method_input = st.text_input(
            "Method name",
            placeholder="e.g., instrumental variables",
        )
    with col2:
        st.markdown("**Quick select:**")

    # Quick select buttons
    selected_method = method_input
    cols = st.columns(5)
    for i, method in enumerate(COMMON_METHODS):
        with cols[i % 5]:
            if st.button(method, key=f"method_{i}", use_container_width=True):
                selected_method = method

    if not selected_method:
        st.info("Enter a method name or click a quick-select button above.")
        return

    # Run audit
    try:
        with st.spinner(f"Auditing assumptions for '{selected_method}'..."):
            result = run_async(audit_method(selected_method))
    except httpx.ConnectError:
        st.error("Cannot connect to API server.")
        st.caption("Trying CLI fallback...")
        try:
            result = run_async(_cli_fallback(selected_method))
        except Exception as e:
            st.error(f"CLI fallback also failed: {e}")
            return
    except Exception as e:
        st.error(f"Error: {e}")
        return

    if "error" in result and result["error"]:
        st.warning(f"Partial result: {result['error']}")

    # Display results
    assumptions = result.get("assumptions", [])
    method_name = result.get("method", selected_method)

    st.subheader(f"Assumptions for: {method_name}")

    if not assumptions:
        st.info(
            "No cached assumptions found. Run concept extraction with "
            "`python scripts/extract_concepts.py` to populate the knowledge graph."
        )
        return

    st.success(f"Found {len(assumptions)} assumptions")

    for i, assumption in enumerate(assumptions, 1):
        name = assumption.get("name", assumption.get("assumption", f"Assumption {i}"))
        description = assumption.get("description", assumption.get("definition", ""))
        importance = assumption.get("importance", "")
        testable = assumption.get("testable", None)
        sources = assumption.get("sources", assumption.get("source_chunks", []))

        # Expander for each assumption
        icon = "!" if importance == "critical" else "?"
        with st.expander(f"**{i}. {name}**", expanded=(i <= 3)):
            if description:
                st.markdown(description)

            cols = st.columns(3)
            if importance:
                cols[0].markdown(f"**Importance:** {importance}")
            if testable is not None:
                cols[1].markdown(f"**Testable:** {'Yes' if testable else 'No'}")
            if sources:
                cols[2].markdown(f"**Sources:** {len(sources)} references")

            # Show source references
            if sources:
                st.markdown("**Referenced in:**")
                for src in sources[:5]:  # Limit display
                    if isinstance(src, dict):
                        title = src.get("source_title", src.get("title", "Unknown"))
                        chunk_text = src.get("content", src.get("chunk_text", ""))[:200]
                        st.markdown(f"- *{title}*: {chunk_text}...")
                    else:
                        st.markdown(f"- {src}")

    # Export option
    st.divider()
    st.download_button(
        "Download as JSON",
        data=json.dumps(result, indent=2),
        file_name=f"assumptions_{method_name.replace(' ', '_')}.json",
        mime="application/json",
    )
