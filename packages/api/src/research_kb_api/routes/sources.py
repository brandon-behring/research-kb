"""Source endpoints.

Provides access to source documents (papers, textbooks) and their chunks.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from research_kb_api import schemas
from research_kb_api import service

router = APIRouter()


@router.get("", response_model=schemas.SourceListResponse)
async def list_sources(
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
) -> schemas.SourceListResponse:
    """List sources with pagination.

    Parameters
    ----------
    limit : int
        Maximum number of sources to return (1-1000)
    offset : int
        Offset for pagination
    source_type : str, optional
        Filter by source type (PAPER, TEXTBOOK, etc.)

    Returns
    -------
    SourceListResponse
        Paginated list of sources with metadata.
    """
    sources = await service.get_sources(
        limit=limit,
        offset=offset,
        source_type=source_type,
    )

    total = await service.count_sources(source_type=source_type)

    return schemas.SourceListResponse(
        sources=[
            schemas.SourceDetail(
                id=str(s.id),
                title=s.title,
                authors=s.authors or [],
                year=s.year,
                source_type=s.source_type.value if s.source_type else None,
                file_path=s.file_path,
                abstract=s.metadata.get("abstract") if s.metadata else None,
                metadata=s.metadata,
                created_at=s.created_at.isoformat() if s.created_at else None,
            )
            for s in sources
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{source_id}", response_model=schemas.SourceWithChunks)
async def get_source(source_id: str) -> schemas.SourceWithChunks:
    """Get source details with chunks.

    Parameters
    ----------
    source_id : str
        UUID of the source

    Returns
    -------
    SourceWithChunks
        Source details with all associated text chunks.

    Raises
    ------
    HTTPException
        404 if source not found
    """
    source = await service.get_source_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")

    chunks = await service.get_source_chunks(source_id)

    return schemas.SourceWithChunks(
        source=schemas.SourceDetail(
            id=str(source.id),
            title=source.title,
            authors=source.authors or [],
            year=source.year,
            source_type=source.source_type.value if source.source_type else None,
            file_path=source.file_path,
            abstract=source.metadata.get("abstract") if source.metadata else None,
            metadata=source.metadata,
            created_at=source.created_at.isoformat() if source.created_at else None,
        ),
        chunks=[
            schemas.ChunkDetail(
                id=str(c.id),
                content=c.content,
                page_start=c.page_start,
                page_end=c.page_end,
                metadata=c.metadata,
            )
            for c in chunks
        ],
        chunk_count=len(chunks),
    )


@router.get("/{source_id}/citations", response_model=schemas.SourceCitations)
async def get_source_citations(source_id: str) -> schemas.SourceCitations:
    """Get citation information for a source.

    Parameters
    ----------
    source_id : str
        UUID of the source

    Returns
    -------
    SourceCitations
        Lists of sources that cite this source and sources cited by this source.

    Raises
    ------
    HTTPException
        404 if source not found
    """
    source = await service.get_source_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")

    citations = await service.get_citations_for_source(source_id)

    return schemas.SourceCitations(
        source_id=source_id,
        citing_sources=[
            schemas.CitationSummary(
                id=c["id"],
                title=c["title"],
                year=c.get("year"),
            )
            for c in citations.get("citing_sources", [])
        ],
        cited_sources=[
            schemas.CitationSummary(
                id=c["id"],
                title=c["title"],
                year=c.get("year"),
            )
            for c in citations.get("cited_sources", [])
        ],
        citation_count=len(citations.get("citing_sources", [])),
        reference_count=len(citations.get("cited_sources", [])),
    )
