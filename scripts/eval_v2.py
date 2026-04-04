#!/usr/bin/env python3
"""Standards-compliant IR evaluation using ranx.

Replaces eval_retrieval.py (v1) with proper NDCG, MRR, MAP, Recall metrics
computed against TREC-pooled, human-annotated ground truth.

Key differences from v1:
  - Uses ranx for all metric computation (standard IR methodology)
  - Graded relevance (0-3) instead of binary
  - Multi-document ground truth per query (TREC pooling)
  - Statistical significance via Fisher's randomization test
  - Fixed limit=100 (not variable per-query)

Metrics:
  Primary:   NDCG@10 (CI gate metric)
  Secondary: MRR@10, MAP@10, Recall@10, Recall@20, Hit@10, Precision@10

Usage:
    # Dry-run with auto-labels (before annotation)
    python scripts/eval_v2.py --use-suggested --per-domain --verbose

    # Real eval with human annotations
    python scripts/eval_v2.py --per-domain --per-difficulty --output evaluation_runs/v2_baseline.json

    # Citation ablation (Phase 7)
    python scripts/eval_v2.py --config hybrid --output evaluation_runs/v2_citations_off.json

    # Compare two runs
    python scripts/eval_v2.py --compare evaluation_runs/v2_baseline.json evaluation_runs/v2_citations_off.json
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from ranx import Qrels, Run, compare, evaluate

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_pdf import EmbeddingClient
from research_kb_storage import (
    SearchQuery,
    get_connection_pool,
    search_hybrid,
    search_hybrid_v2,
)

# ranx metric names → eval_compare.py JSON keys (Decision 29, 35)
RANX_METRICS = [
    "ndcg@5",
    "ndcg@10",
    "mrr@10",
    "map@10",
    "recall@10",
    "recall@20",
    "hit_rate@10",
    "precision@10",
]

METRIC_NAME_MAP = {
    "ndcg@5": "ndcg_5",
    "ndcg@10": "ndcg_10",
    "mrr@10": "mrr",
    "map@10": "map_10",
    "recall@10": "recall_10",
    "recall@20": "recall_20",
    "hit_rate@10": "hit_rate_at_k",
    "precision@10": "precision_10",
}

SEARCH_CONFIGS = {
    "citations": {
        "fts_weight": 0.3,
        "vector_weight": 0.7,
        "use_citations": True,
        "citation_weight": 0.15,
    },
    "hybrid": {
        "fts_weight": 0.3,
        "vector_weight": 0.7,
    },
    "fts": {
        "fts_weight": 1.0,
        "vector_weight": 0.0,
    },
    "vector": {
        "fts_weight": 0.0,
        "vector_weight": 1.0,
    },
}


def convert_metric_names(ranx_metrics: dict[str, float]) -> dict[str, float]:
    """Convert ranx metric names to eval_compare.py JSON keys.

    Parameters
    ----------
    ranx_metrics : dict
        Metrics keyed by ranx names (e.g., "ndcg@10").

    Returns
    -------
    dict
        Metrics keyed by eval_compare.py names (e.g., "ndcg_10").
    """
    return {METRIC_NAME_MAP.get(k, k): float(v) for k, v in ranx_metrics.items() if k in METRIC_NAME_MAP}


def load_ground_truth(
    path: Path,
    use_suggested: bool = False,
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    """Load ground truth from annotated YAML file.

    Parameters
    ----------
    path : Path
        Path to the annotated/prelabeled YAML file.
    use_suggested : bool
        If True, fall back to suggested_grade when relevance is null.

    Returns
    -------
    qrels_dict : dict
        {query_id: {chunk_id: grade}} for ranx Qrels (grade >= 1 only, Decision 36).
    queries : list
        Full query metadata for domain/difficulty grouping.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    queries = data["queries"]
    qrels_dict: dict[str, dict[str, int]] = {}

    for query in queries:
        qid = query["query_id"]
        qrels_dict[qid] = {}
        for candidate in query["candidates"]:
            grade = candidate.get("relevance")
            if grade is None and use_suggested:
                grade = candidate.get("suggested_grade")
            if grade is not None and grade > 0:
                qrels_dict[qid][candidate["chunk_id"]] = int(grade)

    # Warn about queries with no relevant docs
    empty_queries = [qid for qid, docs in qrels_dict.items() if not docs]
    if empty_queries:
        print(f"WARNING: {len(empty_queries)} queries have zero relevant docs in qrels:")
        for qid in empty_queries[:5]:
            print(f"  {qid}")
        if len(empty_queries) > 5:
            print(f"  ... and {len(empty_queries) - 5} more")

    return qrels_dict, queries


def build_search_query(
    text: str,
    embedding: Optional[list[float]],
    config_name: str,
    limit: int = 100,
) -> SearchQuery:
    """Build SearchQuery for a given config.

    Parameters
    ----------
    text : str
        Query text.
    embedding : list[float] or None
        1024-dim BGE embedding.
    config_name : str
        One of: citations, hybrid, fts, vector.
    limit : int
        Max results (default: 100, per Phase 1.2).

    Returns
    -------
    SearchQuery
    """
    config = SEARCH_CONFIGS.get(config_name)
    if config is None:
        raise ValueError(f"Unknown config: {config_name}. Expected one of {list(SEARCH_CONFIGS)}")

    kwargs: dict[str, Any] = {"limit": limit}
    kwargs["fts_weight"] = config["fts_weight"]
    kwargs["vector_weight"] = config["vector_weight"]

    if config_name == "fts":
        kwargs["text"] = text
        kwargs["embedding"] = None
    elif config_name == "vector":
        kwargs["text"] = None
        kwargs["embedding"] = embedding
    else:
        kwargs["text"] = text
        kwargs["embedding"] = embedding

    if config.get("use_citations"):
        kwargs["use_citations"] = True
        kwargs["citation_weight"] = config["citation_weight"]

    return SearchQuery(**kwargs)


async def run_evaluation(
    queries: list[dict[str, Any]],
    config_name: str,
    embed_client: EmbeddingClient,
    verbose: bool = False,
) -> dict[str, dict[str, float]]:
    """Run search for all queries and build a ranx Run dict.

    Parameters
    ----------
    queries : list
        Query metadata from ground truth YAML.
    config_name : str
        Search configuration name.
    embed_client : EmbeddingClient
        For computing query embeddings.
    verbose : bool
        Print per-query progress.

    Returns
    -------
    run_dict : dict
        {query_id: {chunk_id: score}} for ranx Run.
    """
    run_dict: dict[str, dict[str, float]] = {}
    use_v2 = config_name == "citations"

    for i, query in enumerate(queries):
        query_text = query["query_text"]
        qid = query["query_id"]

        embedding = embed_client.embed_query(query_text)
        search_query = build_search_query(query_text, embedding, config_name, limit=100)

        if use_v2:
            results = await search_hybrid_v2(search_query)
        else:
            results = await search_hybrid(search_query)

        run_dict[qid] = {}
        for r in results:
            chunk_id = str(r.chunk.id)
            run_dict[qid][chunk_id] = float(r.combined_score)

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(queries)}] {query_text[:50]}... ({len(results)} results)")

    return run_dict


def compute_source_hit_rate(
    queries: list[dict[str, Any]],
    run_dict: dict[str, dict[str, float]],
    use_suggested: bool = False,
    top_k: int = 10,
) -> float:
    """Derive source-level hit rate from chunk-level results.

    For each query, checks if any chunk from a relevant source (grade >= 2)
    appears in the top-K results. This is a post-processing derivation,
    not a ranx-native metric.

    Parameters
    ----------
    queries : list
        Query metadata with candidates.
    run_dict : dict
        {query_id: {chunk_id: score}} from search.
    use_suggested : bool
        Use suggested_grade when relevance is null.
    top_k : int
        Number of top results to check.

    Returns
    -------
    float
        Source hit rate (0.0 to 1.0).
    """
    hits = 0
    for query in queries:
        qid = query["query_id"]
        if qid not in run_dict:
            continue

        # Get relevant source IDs from ground truth (grade >= 2)
        relevant_sources = set()
        for c in query["candidates"]:
            grade = c.get("relevance")
            if grade is None and use_suggested:
                grade = c.get("suggested_grade")
            if grade is not None and grade >= 2:
                relevant_sources.add(c["source_id"])

        if not relevant_sources:
            continue

        # Get top-K chunk IDs from run, map to source IDs
        sorted_chunks = sorted(run_dict[qid].items(), key=lambda x: x[1], reverse=True)[:top_k]
        retrieved_chunk_ids = {cid for cid, _ in sorted_chunks}

        # Check if any retrieved chunk belongs to a relevant source
        for c in query["candidates"]:
            if c["chunk_id"] in retrieved_chunk_ids and c["source_id"] in relevant_sources:
                hits += 1
                break

    return hits / len(queries) if queries else 0.0


def compute_per_group_metrics(
    queries: list[dict[str, Any]],
    per_query_metrics: dict[str, Any],
    group_key: str,
) -> dict[str, dict[str, Any]]:
    """Compute metrics grouped by a query metadata field.

    Parameters
    ----------
    queries : list
        Query metadata.
    per_query_metrics : dict
        Per-query metric arrays from ranx (return_mean=False).
    group_key : str
        Field to group by (e.g., "domain", "difficulty").

    Returns
    -------
    dict
        {group_value: {"total": N, "mrr": ..., "ndcg_10": ...}}
    """
    # Build query_id → index mapping
    all_qids = [q["query_id"] for q in queries]

    # Group queries
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, query in enumerate(queries):
        groups[query.get(group_key, "unknown")].append(idx)

    result = {}
    for group_name, indices in sorted(groups.items()):
        group_metrics: dict[str, Any] = {"total": len(indices)}
        for metric_name in RANX_METRICS:
            values = per_query_metrics[metric_name]
            group_mean = float(np.mean([values[i] for i in indices]))
            json_key = METRIC_NAME_MAP.get(metric_name, metric_name)
            group_metrics[json_key] = round(group_mean, 4)
        result[group_name] = group_metrics

    return result


def build_output(
    aggregate: dict[str, float],
    queries: list[dict[str, Any]],
    run_dict: dict[str, dict[str, float]],
    per_query_metrics: dict[str, Any],
    config_name: str,
    ground_truth_path: str,
    ground_truth_field: str,
    per_domain: bool = False,
    per_difficulty: bool = False,
    use_suggested: bool = False,
) -> dict[str, Any]:
    """Build JSON output compatible with eval_compare.py.

    Parameters
    ----------
    aggregate : dict
        Aggregate ranx metrics.
    queries : list
        Query metadata.
    run_dict : dict
        Search results.
    per_query_metrics : dict
        Per-query ranx metrics.
    config_name : str
        Search configuration used.
    ground_truth_path : str
        Path to ground truth file.
    ground_truth_field : str
        Which field was used (relevance or suggested_grade).
    per_domain : bool
        Include per-domain breakdown.
    per_difficulty : bool
        Include per-difficulty breakdown.
    use_suggested : bool
        Whether suggested_grade was used.

    Returns
    -------
    dict
        JSON-serializable output.
    """
    converted = convert_metric_names(aggregate)
    source_hit_rate = compute_source_hit_rate(queries, run_dict, use_suggested)

    output: dict[str, Any] = {
        "version": 2,
        "methodology": "ranx_v2",
        "created": datetime.now(timezone.utc).isoformat(),
        "ground_truth": ground_truth_path,
        "ground_truth_field": ground_truth_field,
        "search_config": config_name,
        "total": len(queries),
        "passed": None,
        "failed": None,
        "source_hit_rate_at_k": round(source_hit_rate, 4),
    }
    output.update({k: round(v, 4) for k, v in converted.items()})

    if per_domain:
        output["per_domain"] = compute_per_group_metrics(queries, per_query_metrics, "domain")

    if per_difficulty:
        output["per_difficulty"] = compute_per_group_metrics(queries, per_query_metrics, "difficulty")

    return output


def run_comparison(
    qrels: Qrels,
    baseline_path: str,
    candidate_path: str,
) -> None:
    """Compare two saved runs with statistical significance.

    Loads two JSON files (eval_v2 output), reconstructs ranx Runs from
    stored run_dict, and runs Fisher's randomization test.

    Parameters
    ----------
    qrels : Qrels
        Ground truth.
    baseline_path : str
        Path to baseline JSON (must contain run_dict).
    candidate_path : str
        Path to candidate JSON (must contain run_dict).
    """
    with open(baseline_path) as f:
        baseline_data = json.load(f)
    with open(candidate_path) as f:
        candidate_data = json.load(f)

    if "run_dict" not in baseline_data or "run_dict" not in candidate_data:
        print("ERROR: JSON files must contain 'run_dict' for comparison.")
        print("Re-run eval_v2.py with --save-run to include run data.")
        sys.exit(1)

    run_baseline = Run(baseline_data["run_dict"], name=Path(baseline_path).stem)
    run_candidate = Run(candidate_data["run_dict"], name=Path(candidate_path).stem)

    report = compare(
        qrels=qrels,
        runs=[run_baseline, run_candidate],
        metrics=["ndcg@10", "mrr@10", "map@10", "recall@10"],
        max_p=0.05,
        stat_test="fisher",
    )
    print(report)


async def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="v2 IR evaluation using ranx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("fixtures/eval/v2_pool_prelabeled.yaml"),
        help="Annotated YAML ground truth file",
    )
    parser.add_argument(
        "--use-suggested",
        action="store_true",
        help="Use suggested_grade when relevance is null (for dry-run before annotation)",
    )
    parser.add_argument(
        "--config",
        choices=list(SEARCH_CONFIGS),
        default="citations",
        help="Search configuration (default: citations)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path",
    )
    parser.add_argument(
        "--per-domain",
        action="store_true",
        help="Include per-domain metric breakdown",
    )
    parser.add_argument(
        "--per-difficulty",
        action="store_true",
        help="Include per-difficulty metric breakdown",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query progress",
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        help="Exit code 1 if NDCG@10 below this threshold (for CI gating)",
    )
    parser.add_argument(
        "--save-run",
        action="store_true",
        help="Include run_dict in JSON output (needed for --compare)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="Compare two saved runs with statistical significance",
    )
    args = parser.parse_args()

    # --- Load ground truth ---
    print(f"Loading ground truth from {args.ground_truth}")
    qrels_dict, queries = load_ground_truth(args.ground_truth, args.use_suggested)
    gt_field = "suggested_grade" if args.use_suggested else "relevance"
    print(f"  {len(queries)} queries, {sum(len(d) for d in qrels_dict.values())} relevant judgments")
    print(f"  Ground truth field: {gt_field}")

    qrels = Qrels(qrels_dict)

    # --- Comparison mode ---
    if args.compare:
        run_comparison(qrels, args.compare[0], args.compare[1])
        return

    # --- Run search ---
    print(f"\nRunning search with config '{args.config}' (limit=100)...")
    embed_client = EmbeddingClient()
    run_dict = await run_evaluation(queries, args.config, embed_client, args.verbose)
    print(f"  Search complete: {len(run_dict)} queries")

    run = Run(run_dict)

    # --- Compute metrics ---
    print("\nComputing metrics...")
    aggregate = evaluate(qrels, run, RANX_METRICS)
    per_query_metrics = evaluate(qrels, run, RANX_METRICS, return_mean=False)

    # --- Build output ---
    output = build_output(
        aggregate=aggregate,
        queries=queries,
        run_dict=run_dict,
        per_query_metrics=per_query_metrics,
        config_name=args.config,
        ground_truth_path=str(args.ground_truth),
        ground_truth_field=gt_field,
        per_domain=args.per_domain,
        per_difficulty=args.per_difficulty,
        use_suggested=args.use_suggested,
    )

    if args.save_run:
        output["run_dict"] = run_dict

    # --- Print summary ---
    print("\n=== v2 Evaluation Results ===")
    print(f"  Config: {args.config}")
    print(f"  Queries: {output['total']}")
    print(f"  NDCG@10:   {output['ndcg_10']:.4f}  (primary)")
    print(f"  MRR@10:    {output['mrr']:.4f}")
    print(f"  MAP@10:    {output['map_10']:.4f}")
    print(f"  Recall@10: {output['recall_10']:.4f}")
    print(f"  Hit@10:    {output['hit_rate_at_k']:.4f}")
    print(f"  Source Hit@10: {output['source_hit_rate_at_k']:.4f}")

    if args.per_domain and "per_domain" in output:
        print(f"\n  Per-domain ({len(output['per_domain'])} domains):")
        for domain, metrics in sorted(output["per_domain"].items(), key=lambda x: x[1].get("ndcg_10", 0)):
            ndcg = metrics.get("ndcg_10", 0)
            mrr = metrics.get("mrr", 0)
            n = metrics["total"]
            print(f"    {domain:30s}  NDCG@10={ndcg:.3f}  MRR={mrr:.3f}  n={n}")

    if args.per_difficulty and "per_difficulty" in output:
        print(f"\n  Per-difficulty:")
        for diff, metrics in sorted(output["per_difficulty"].items()):
            ndcg = metrics.get("ndcg_10", 0)
            n = metrics["total"]
            print(f"    {diff:10s}  NDCG@10={ndcg:.3f}  n={n}")

    # --- Save output ---
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")

    # --- CI gate ---
    if args.fail_below is not None:
        ndcg10 = output["ndcg_10"]
        if ndcg10 < args.fail_below:
            print(f"\nFAIL: NDCG@10 {ndcg10:.4f} < threshold {args.fail_below}")
            sys.exit(1)
        else:
            print(f"\nPASS: NDCG@10 {ndcg10:.4f} >= threshold {args.fail_below}")


if __name__ == "__main__":
    asyncio.run(main())
