"""MCP server for research-kb knowledge base.

Exposes the research-kb semantic search system to Claude Code via
Model Context Protocol (stdio transport).
"""

from research_kb_mcp.server import mcp, main

__all__ = ["mcp", "main"]
__version__ = "1.0.0"
