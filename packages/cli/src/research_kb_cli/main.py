"""Research KB CLI - Main entry point.

Provides the ``research-kb`` command-line interface.
Sub-commands are grouped by domain: search, graph, citations, sources,
discover, and enrich.

Usage:
    research-kb search query "backdoor criterion" --limit 5 --format markdown
    research-kb graph concepts "IV"
    research-kb citations stats
    research-kb sources list

See docs/INDEX.md for architecture and phase overview.
"""

import typer

from research_kb_cli.commands.citations import app as citations_app
from research_kb_cli.commands.graph import app as graph_app
from research_kb_cli.commands.search import app as search_app
from research_kb_cli.commands.sources import app as sources_app
from research_kb_cli.discover import app as discover_app
from research_kb_cli.enrich import app as enrich_app

# ---------------------------------------------------------------------------
# Root Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="research-kb",
    help="Search and explore the research knowledge base with semantic search and graph-boosted retrieval.",
    add_completion=False,
)

# Register sub-apps
app.add_typer(search_app, name="search")
app.add_typer(graph_app, name="graph")
app.add_typer(citations_app, name="citations")
app.add_typer(sources_app, name="sources")
app.add_typer(discover_app, name="discover")
app.add_typer(enrich_app, name="enrich")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
