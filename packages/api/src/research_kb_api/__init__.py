"""Research-KB REST API.

FastAPI-based REST API for semantic search over causal inference literature.
"""

from research_kb_api.main import app, create_app

__all__ = ["app", "create_app"]
