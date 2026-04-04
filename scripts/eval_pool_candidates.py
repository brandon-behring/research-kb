#!/usr/bin/env python3
"""TREC-style pooling tool for building evaluation ground truth.

Runs each query against multiple search configurations, pools top-K results
from each, deduplicates by chunk_id, and outputs a YAML file ready for
annotation.

Configs:
  - fts:       FTS-only (search_hybrid with fts_weight=1.0, vector_weight=0.0)
  - vector:    Vector-only (search_hybrid with fts_weight=0.0, vector_weight=1.0)
  - hybrid:    Hybrid 30/70 (search_hybrid with fts_weight=0.3, vector_weight=0.7)
  - citations: Hybrid + citations (search_hybrid_v2 with citation_weight=0.15)

Usage:
    python scripts/eval_pool_candidates.py --queries fixtures/eval/retrieval_test_cases.yaml
    python scripts/eval_pool_candidates.py --queries fixtures/eval/v2_queries.yaml --top-k 20
    python scripts/eval_pool_candidates.py --queries fixtures/eval/v2_queries.yaml --configs fts,vector
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_pdf import EmbeddingClient
from research_kb_storage import (
    DatabaseConfig,
    SearchQuery,
    get_connection_pool,
    search_hybrid,
    search_hybrid_v2,
)

ALL_CONFIGS = ["fts", "vector", "hybrid", "citations"]


def build_search_query(
    text: str,
    embedding: Optional[list[float]],
    config_name: str,
    limit: int,
    domain_id: Optional[str] = None,
) -> SearchQuery:
    """Build a SearchQuery for a specific pooling configuration.

    Parameters
    ----------
    text : str
        Query text.
    embedding : list[float] or None
        1024-dim BGE embedding. Required for vector/hybrid/citations configs.
    config_name : str
        One of: fts, vector, hybrid, citations.
    limit : int
        Max results to return.
    domain_id : str or None
        Optional domain filter.

    Returns
    -------
    SearchQuery
        Configured query object.
    """
    if config_name == "fts":
        return SearchQuery(
            text=text,
            embedding=None,
            fts_weight=1.0,
            vector_weight=0.0,
            limit=limit,
            domain_id=domain_id,
        )
    elif config_name == "vector":
        if embedding is None:
            raise ValueError("Vector config requires an embedding")
        return SearchQuery(
            text=None,
            embedding=embedding,
            fts_weight=0.0,
            vector_weight=1.0,
            limit=limit,
            domain_id=domain_id,
        )
    elif config_name == "hybrid":
        if embedding is None:
            raise ValueError("Hybrid config requires an embedding")
        return SearchQuery(
            text=text,
            embedding=embedding,
            fts_weight=0.3,
            vector_weight=0.7,
            limit=limit,
            domain_id=domain_id,
        )
    elif config_name == "citations":
        if embedding is None:
            raise ValueError("Citations config requires an embedding")
        return SearchQuery(
            text=text,
            embedding=embedding,
            fts_weight=0.3,
            vector_weight=0.7,
            limit=limit,
            use_citations=True,
            citation_weight=0.15,
            domain_id=domain_id,
        )
    else:
        raise ValueError(f"Unknown config: {config_name}. Expected one of {ALL_CONFIGS}")


async def pool_query(
    query_text: str,
    domain: Optional[str],
    embed_client: EmbeddingClient,
    top_k: int,
    configs: list[str],
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a query against each config, pool and deduplicate results.

    Parameters
    ----------
    query_text : str
        The search query.
    domain : str or None
        Domain hint (not used as filter, stored as metadata).
    embed_client : EmbeddingClient
        Client for computing query embeddings.
    top_k : int
        Number of results per config.
    configs : list[str]
        Search configurations to use.
    verbose : bool
        Print per-config result counts.

    Returns
    -------
    dict
        Deduplicated candidate pool with metadata.
    """
    # Compute embedding once, reuse across configs
    embedding = None
    needs_embedding = any(c in configs for c in ["vector", "hybrid", "citations"])
    if needs_embedding:
        embedding = embed_client.embed_query(query_text)

    candidates: dict[str, dict[str, Any]] = {}

    for config_name in configs:
        try:
            query = build_search_query(query_text, embedding, config_name, top_k, domain_id=None)

            if config_name == "citations":
                results = await search_hybrid_v2(query)
            else:
                results = await search_hybrid(query)

            config_count = 0
            for r in results:
                cid = str(r.chunk.id)
                if cid not in candidates:
                    candidates[cid] = {
                        "chunk_id": cid,
                        "source_id": str(r.source.id),
                        "source_title": r.source.title,
                        "authors": list(r.source.authors[:3]),
                        "page_start": r.chunk.page_start,
                        "page_end": r.chunk.page_end,
                        "text_snippet": r.chunk.content[:200] if r.chunk.content else "",
                        "found_in_configs": [config_name],
                        # Annotation fields (filled later by prelabel + annotate)
                        "relevance": None,
                        "evidence_text": None,
                        "suggested_grade": None,
                        "similarity_score": None,
                    }
                    config_count += 1
                else:
                    if config_name not in candidates[cid]["found_in_configs"]:
                        candidates[cid]["found_in_configs"].append(config_name)

            if verbose:
                print(f"    {config_name}: {len(results)} results, {config_count} new unique")

        except Exception as e:
            print(f"    WARNING: {config_name} failed for '{query_text[:40]}...': {e}", file=sys.stderr)

    return candidates


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Load queries from YAML file.

    Supports two formats:
    1. v1 test cases (retrieval_test_cases.yaml): has 'test_cases' key
    2. v2 query list: has 'queries' key with query_id, query_text, domain, difficulty

    Parameters
    ----------
    path : Path
        Path to YAML file.

    Returns
    -------
    list[dict]
        List of query dicts with: query_id, query_text, domain, difficulty.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    if "test_cases" in data:
        # v1 format: convert to v2 query list
        queries = []
        for i, tc in enumerate(data["test_cases"]):
            domain = tc.get("domain", "unknown")
            queries.append({
                "query_id": f"q_{domain}_{i + 1:03d}",
                "query_text": tc["query"],
                "domain": domain,
                "difficulty": tc.get("difficulty", "medium"),
            })
        return queries
    elif "queries" in data:
        # v2 format: already structured
        return data["queries"]
    else:
        raise ValueError(f"Unrecognized YAML format in {path}. Expected 'test_cases' or 'queries' key.")


async def run_pooling(
    queries: list[dict[str, Any]],
    embed_client: EmbeddingClient,
    top_k: int,
    configs: list[str],
    verbose: bool = False,
) -> dict[str, Any]:
    """Run TREC pooling across all queries and configs.

    Parameters
    ----------
    queries : list[dict]
        Query list with query_id, query_text, domain, difficulty.
    embed_client : EmbeddingClient
        Embedding client.
    top_k : int
        Results per config.
    configs : list[str]
        Configs to pool from.
    verbose : bool
        Verbose output.

    Returns
    -------
    dict
        Full pooling output with metadata and per-query candidates.
    """
    pooled_queries = []
    total_candidates = 0

    for i, q in enumerate(queries):
        query_text = q["query_text"]
        domain = q.get("domain")
        query_id = q.get("query_id", f"q_{i + 1:03d}")
        difficulty = q.get("difficulty", "medium")

        print(f"  [{i + 1}/{len(queries)}] {query_text[:60]}", end="")
        if verbose:
            print()

        candidates = await pool_query(
            query_text=query_text,
            domain=domain,
            embed_client=embed_client,
            top_k=top_k,
            configs=configs,
            verbose=verbose,
        )

        # Convert to list sorted by number of configs found in (descending)
        candidate_list = sorted(
            candidates.values(),
            key=lambda c: len(c["found_in_configs"]),
            reverse=True,
        )

        pooled_queries.append({
            "query_id": query_id,
            "query_text": query_text,
            "domain": domain,
            "difficulty": difficulty,
            "candidate_count": len(candidate_list),
            "candidates": candidate_list,
        })

        total_candidates += len(candidate_list)
        if not verbose:
            print(f" -> {len(candidate_list)} candidates")

    output = {
        "metadata": {
            "created": datetime.now(timezone.utc).isoformat(),
            "configs": configs,
            "top_k": top_k,
            "total_queries": len(queries),
            "total_candidates": total_candidates,
            "avg_candidates_per_query": round(total_candidates / len(queries), 1) if queries else 0,
        },
        "queries": pooled_queries,
    }

    return output


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="TREC-style pooling: run queries against multiple configs, pool results"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Path to YAML file with queries (v1 test_cases or v2 query list)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("fixtures/eval/v2_pool_candidates.yaml"),
        help="Output YAML path (default: fixtures/eval/v2_pool_candidates.yaml)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Results per config (default: 20)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(ALL_CONFIGS),
        help=f"Comma-separated configs (default: {','.join(ALL_CONFIGS)})",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Only pool first N queries (for smoke testing)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-config breakdown",
    )
    args = parser.parse_args()

    # Parse configs
    configs = [c.strip() for c in args.configs.split(",")]
    for c in configs:
        if c not in ALL_CONFIGS:
            print(f"ERROR: Unknown config '{c}'. Must be one of: {ALL_CONFIGS}", file=sys.stderr)
            sys.exit(1)

    # Load queries
    queries = load_queries(args.queries)
    if args.limit_queries:
        queries = queries[: args.limit_queries]

    print(f"Pooling {len(queries)} queries x {len(configs)} configs (top-{args.top_k} each)")
    print(f"Configs: {configs}")
    print()

    # Initialize DB + embedding client
    config = DatabaseConfig()
    await get_connection_pool(config)
    embed_client = EmbeddingClient()

    # Run pooling
    output = await run_pooling(
        queries=queries,
        embed_client=embed_client,
        top_k=args.top_k,
        configs=configs,
        verbose=args.verbose,
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print()
    print(f"Pooling complete:")
    print(f"  Queries: {output['metadata']['total_queries']}")
    print(f"  Total candidates: {output['metadata']['total_candidates']}")
    print(f"  Avg per query: {output['metadata']['avg_candidates_per_query']}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
