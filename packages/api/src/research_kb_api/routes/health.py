"""Health check endpoints.

Provides health monitoring for the API and its dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from research_kb_api import schemas
from research_kb_api import service

if TYPE_CHECKING:
    pass

router = APIRouter()


@router.get("/health", response_model=schemas.HealthCheck)
async def health_check() -> schemas.HealthCheck:
    """Basic health check.

    Returns minimal status for load balancer probes.
    """
    return schemas.HealthCheck(
        status="healthy",
        version="1.0.0",
        database="connected",
        embedding_model="ready",
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
