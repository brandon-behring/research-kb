"""API route handlers."""

from research_kb_api.routes.health import router as health_router
from research_kb_api.routes.search import router as search_router
from research_kb_api.routes.sources import router as sources_router
from research_kb_api.routes.concepts import router as concepts_router
from research_kb_api.routes.graph import router as graph_router
from research_kb_api.routes.domains import router as domains_router

__all__ = [
    "health_router",
    "search_router",
    "sources_router",
    "concepts_router",
    "graph_router",
    "domains_router",
]
