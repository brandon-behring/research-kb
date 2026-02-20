"""Health check endpoints.

Provides health monitoring for the API and its dependencies.

Kubernetes-style health checks:
- /health/live - Liveness probe (is the process alive?)
- /health/ready - Readiness probe (can it serve traffic?)
- /health - Combined check with actual dependency validation
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from research_kb_api import schemas
from research_kb_api import service

if TYPE_CHECKING:
    pass

router = APIRouter()


@router.get("/health/live")
async def liveness() -> dict:
    """Liveness probe - is the process alive?

    Returns 200 if the process is running. Used by orchestrators
    to detect if the container needs to be restarted.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness() -> dict:
    """Readiness probe - can the service handle traffic?

    Checks that all required dependencies are available.
    Returns 200 only when the service can handle requests.
    """
    components = {}
    start = time.time()

    # Check database connectivity
    try:
        stats = await service.get_stats()
        components["database"] = {"status": "ready", "sources": stats.get("sources", 0)}
    except Exception as e:
        components["database"] = {"status": "not_ready", "error": str(e)[:100]}

    # Check embedding model
    try:
        client = service.get_embedding_client()
        if client is not None:
            components["embedding"] = {"status": "ready"}
        else:
            components["embedding"] = {
                "status": "not_ready",
                "error": "client not initialized",
            }
    except Exception as e:
        components["embedding"] = {"status": "not_ready", "error": str(e)[:100]}

    # Overall status
    all_ready = all(c.get("status") == "ready" for c in components.values())
    elapsed_ms = (time.time() - start) * 1000

    return {
        "status": "ready" if all_ready else "not_ready",
        "components": components,
        "check_duration_ms": round(elapsed_ms, 2),
    }


@router.get("/health", response_model=schemas.HealthCheck)
async def health_check() -> schemas.HealthCheck:
    """Primary health check with actual dependency validation.

    Pings the database to verify connectivity.
    """
    db_status = "connected"
    embedding_status = "ready"
    overall_status = "healthy"

    # Actually check database
    try:
        await service.get_stats()
    except Exception:
        db_status = "disconnected"
        overall_status = "degraded"

    # Check embedding client availability
    try:
        client = service.get_embedding_client()
        if client is None:
            embedding_status = "initializing"
    except Exception:
        embedding_status = "unavailable"
        overall_status = "degraded"

    return schemas.HealthCheck(
        status=overall_status,
        version="1.0.0",
        database=db_status,
        embedding_model=embedding_status,
    )


@router.get("/health/detailed", response_model=schemas.HealthDetail)
async def health_detailed(request: Request) -> schemas.HealthDetail:
    """Detailed health check with component status and statistics.

    Checks database connectivity and returns corpus statistics.
    """
    components = {}
    stats = None

    # Check database
    try:
        stats_data = await service.get_stats()
        components["database"] = "healthy"
        stats = schemas.DatabaseStats(**stats_data)
    except Exception as e:
        components["database"] = f"unhealthy: {e}"

    # Check embedding model
    try:
        _ = service.get_embedding_client()
        components["embedding_model"] = "healthy"
    except Exception as e:
        components["embedding_model"] = f"unhealthy: {e}"

    # Overall status
    all_healthy = all(v == "healthy" for v in components.values())
    status = "healthy" if all_healthy else "degraded"

    return schemas.HealthDetail(
        status=status,
        version="1.0.0",
        components=components,
        stats=stats,
    )


@router.get("/stats", response_model=schemas.DatabaseStats)
async def get_stats() -> schemas.DatabaseStats:
    """Get database statistics.

    Returns counts for sources, chunks, concepts, relationships, and citations.
    """
    stats_data = await service.get_stats()
    return schemas.DatabaseStats(**stats_data)


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    """
    from research_kb_api.metrics import metrics_endpoint
    from starlette.requests import Request

    # Create a minimal request object (metrics_endpoint doesn't need it)
    return await metrics_endpoint(Request(scope={"type": "http"}))
