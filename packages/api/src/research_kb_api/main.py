"""FastAPI application for research-kb.

Main entry point for the REST API server.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from research_kb_common import get_logger
from research_kb_storage import DatabaseConfig, get_connection_pool

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup/shutdown.

    - Startup: Initialize database connection pool
    - Shutdown: Close pool (handled automatically by asyncpg)
    """
    logger.info("api_starting")

    # Initialize database connection pool
    config = DatabaseConfig()
    pool = await get_connection_pool(config)
    app.state.pool = pool

    logger.info("api_started", pool_size=pool.get_size())

    yield

    # Cleanup
    logger.info("api_stopping")
    await pool.close()
    logger.info("api_stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Research-KB API",
        description="Semantic search API for causal inference literature",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include routers
    from research_kb_api.routes.health import router as health_router
    from research_kb_api.routes.search import router as search_router
    from research_kb_api.routes.sources import router as sources_router
    from research_kb_api.routes.concepts import router as concepts_router
    from research_kb_api.routes.graph import router as graph_router

    app.include_router(health_router, tags=["Health"])
    app.include_router(search_router, prefix="/search", tags=["Search"])
    app.include_router(sources_router, prefix="/sources", tags=["Sources"])
    app.include_router(concepts_router, prefix="/concepts", tags=["Concepts"])
    app.include_router(graph_router, prefix="/graph", tags=["Graph"])

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "research_kb_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
