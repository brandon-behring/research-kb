"""FastMCP server for research-kb knowledge base.

Exposes the research-kb semantic search system via MCP protocol
for integration with Claude Code.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Configure logging IMMEDIATELY to ensure stderr usage
from research_kb_common import get_logger, configure_logging
configure_logging()

from fastmcp import FastMCP
from research_kb_storage import get_connection_pool, close_connection_pool, DatabaseConfig

logger = get_logger(__name__)


@dataclass
class AppContext:
    """Application context available to all tools."""

    pool: Pool


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage database connection pool lifecycle.

    Creates the connection pool on startup and closes it on shutdown.
    The pool is made available to tools via the context.
    """
    logger.info("mcp_server_starting", server_name=server.name)

    # Initialize database connection pool
    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    try:
        logger.info("mcp_server_ready", pool_size=pool.get_size())
        yield AppContext(pool=pool)
    finally:
        logger.info("mcp_server_shutting_down")
        await close_connection_pool()
        logger.info("mcp_server_stopped")


# Create the MCP server instance
mcp = FastMCP(
    "research-kb",
    lifespan=app_lifespan,
)


# Import and register all tools
# This happens at module load time after mcp is created
from research_kb_mcp.tools import (  # noqa: E402
    register_search_tools,
    register_source_tools,
    register_concept_tools,
    register_graph_tools,
    register_health_tools,
    register_citation_tools,
    register_assumption_tools,
)

register_search_tools(mcp)
register_source_tools(mcp)
register_concept_tools(mcp)
register_graph_tools(mcp)
register_health_tools(mcp)
register_citation_tools(mcp)
register_assumption_tools(mcp)


def main():
    """Entry point for research-kb-mcp command."""
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("mcp_server_interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error("mcp_server_error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
