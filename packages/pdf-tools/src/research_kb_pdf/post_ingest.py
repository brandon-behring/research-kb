"""Post-ingestion hooks for PDF ingestion pipelines.

Issue #5: after a batch of sources lands in the database, downstream
structures (citation graph, PageRank authority) need to be updated so
queries like ``research_kb_citation_network`` return results immediately.

This module provides a single entry point, ``run_post_ingest_hooks``,
that ingestion scripts call at their success-exit point. It scopes the
citation graph build to the just-ingested source IDs so the hook stays
cheap even on large corpora.

Usage
-----
>>> from research_kb_pdf.post_ingest import run_post_ingest_hooks
>>> await run_post_ingest_hooks(source_ids=[src.id for src in new_sources])

Set ``build_citations=False`` to skip the hook for very large batch runs
that prefer to defer the build to a single manual invocation.
"""

from __future__ import annotations

from uuid import UUID

from research_kb_common import get_logger

logger = get_logger(__name__)


async def run_post_ingest_hooks(
    source_ids: list[UUID],
    *,
    build_citations: bool = True,
    compute_pagerank: bool = False,
) -> dict:
    """Run post-ingestion maintenance for a batch of newly ingested sources.

    Parameters
    ----------
    source_ids : list[UUID]
        IDs of sources that were just ingested successfully.
    build_citations : bool, default True
        Build ``source_citations`` edges for the new sources. Cheap because
        the query is scoped by source_id.
    compute_pagerank : bool, default False
        Recompute global PageRank citation authority scores. Left off by
        default because the computation is global and more expensive; call
        explicitly on large batches or on a schedule.

    Returns
    -------
    dict
        Summary with keys ``citations`` (build_citation_graph stats) and
        ``pagerank`` (compute_pagerank_authority stats) for each hook that
        ran; missing keys indicate hooks that were skipped.

    Raises
    ------
    ValueError
        If ``source_ids`` is empty and a hook was requested.
    """
    if not source_ids:
        if build_citations or compute_pagerank:
            raise ValueError("run_post_ingest_hooks requires at least one source_id")
        return {}

    summary: dict = {"source_count": len(source_ids)}

    if build_citations:
        # Lazy import to avoid a hard dependency on research_kb_storage when
        # consumers only need the module for typing. Ingestion scripts that
        # invoke this hook always already have storage installed.
        from research_kb_storage import build_citation_graph

        logger.info(
            "post_ingest_citations_build_starting",
            source_count=len(source_ids),
        )
        citation_stats = await build_citation_graph(source_ids=source_ids)
        summary["citations"] = citation_stats
        logger.info(
            "post_ingest_citations_build_complete",
            matched=citation_stats.get("matched", 0),
            unmatched=citation_stats.get("unmatched", 0),
        )

    if compute_pagerank:
        from research_kb_storage import compute_pagerank_authority

        logger.info("post_ingest_pagerank_starting")
        pagerank_stats = await compute_pagerank_authority()
        summary["pagerank"] = pagerank_stats
        logger.info("post_ingest_pagerank_complete")

    return summary


__all__ = ["run_post_ingest_hooks"]
