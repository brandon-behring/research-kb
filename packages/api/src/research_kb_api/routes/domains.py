"""Domain endpoints for the research-kb API.

Provides access to knowledge domain information and statistics.
Domains organize the corpus by topic area (e.g., causal_inference, time_series).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from research_kb_api import schemas
from research_kb_storage import DomainStore
from research_kb_common import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/domains", tags=["Domains"])


@router.get("", response_model=schemas.DomainListResponse)
async def list_domains() -> schemas.DomainListResponse:
    """List all knowledge domains with statistics.

    Returns:
        List of domains with source, chunk, and concept counts.

    Example domains:
        - causal_inference: Econometrics, treatment effects, IV, DiD, DML
        - time_series: Forecasting, ARIMA, VAR, GARCH, state-space
    """
    try:
        stats = await DomainStore.get_all_stats()
        domains = [
            schemas.DomainStats(
                domain_id=s["domain_id"],
                name=s["name"],
                source_count=s["source_count"],
                chunk_count=s["chunk_count"],
                concept_count=s["concept_count"],
            )
            for s in stats
        ]
        return schemas.DomainListResponse(domains=domains, total=len(domains))
    except Exception as e:
        logger.error("list_domains_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list domains: {e}")


@router.get("/{domain_id}", response_model=schemas.DomainStats)
async def get_domain(domain_id: str) -> schemas.DomainStats:
    """Get statistics for a specific domain.

    Args:
        domain_id: Domain identifier (e.g., 'causal_inference', 'time_series')

    Returns:
        Domain statistics including source, chunk, and concept counts.

    Raises:
        404: Domain not found
    """
    try:
        stats = await DomainStore.get_stats(domain_id)
        if stats is None:
            raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")
        return schemas.DomainStats(
            domain_id=stats["domain_id"],
            name=stats["name"],
            source_count=stats["source_count"],
            chunk_count=stats["chunk_count"],
            concept_count=stats["concept_count"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_domain_failed", domain_id=domain_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get domain: {e}")
