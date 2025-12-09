"""Graph visualization helpers using PyVis.

Provides reusable components for rendering interactive network graphs
in Streamlit with PyVis.
"""

import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
from pathlib import Path
import tempfile


def create_network(
    height: str = "600px",
    width: str = "100%",
    bgcolor: str = "#ffffff",
    font_color: str = "#000000",
    directed: bool = True,
) -> Network:
    """Create a configured PyVis Network instance.

    Args:
        height: Graph height (CSS units)
        width: Graph width (CSS units)
        bgcolor: Background color
        font_color: Label font color
        directed: Whether edges are directed (arrows)

    Returns:
        Configured Network instance
    """
    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        directed=directed,
        notebook=False,
        cdn_resources="remote",
    )

    # Configure physics for better layout
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "stabilization": {
                "enabled": true,
                "iterations": 100
            },
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09
            }
        },
        "interaction": {
            "navigationButtons": true,
            "keyboard": true,
            "hover": true,
            "tooltipDelay": 100
        },
        "nodes": {
            "font": {
                "size": 12
            }
        },
        "edges": {
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "smooth": {
                "type": "continuous"
            }
        }
    }
    """)

    return net


def render_network(net: Network, key: str = "graph") -> None:
    """Render a PyVis network in Streamlit.

    Args:
        net: Configured Network instance with nodes/edges added
        key: Unique key for the Streamlit component
    """
    # Generate HTML to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False
    ) as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name

    # Read and display
    with open(tmp_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(html_content, height=650, scrolling=True)

    # Cleanup
    Path(tmp_path).unlink(missing_ok=True)


def get_node_color(source_type: str) -> str:
    """Get node color based on source type.

    Args:
        source_type: Source type (paper, textbook, etc.)

    Returns:
        Hex color code
    """
    colors = {
        "paper": "#4299e1",      # Blue
        "textbook": "#48bb78",   # Green
        "code_repo": "#ed8936",  # Orange
        "unknown": "#a0aec0",    # Gray
    }
    return colors.get(source_type.lower(), colors["unknown"])


def get_node_size(authority: float, min_size: int = 10, max_size: int = 50) -> int:
    """Calculate node size based on authority score.

    Args:
        authority: PageRank authority score (0-1)
        min_size: Minimum node size
        max_size: Maximum node size

    Returns:
        Node size in pixels
    """
    # Scale authority (0-1) to size range
    return int(min_size + (authority * (max_size - min_size)))


def truncate_title(title: str, max_length: int = 40) -> str:
    """Truncate title for display as node label.

    Args:
        title: Full title
        max_length: Maximum characters

    Returns:
        Truncated title with ellipsis if needed
    """
    if len(title) <= max_length:
        return title
    return title[:max_length - 3] + "..."
