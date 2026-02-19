"""Discover candidate chunks for expanding the golden eval dataset.

Searches across underrepresented domains to find chunks that match
specific queries, producing verified (query, chunk_id, domain) triples.

Usage:
    python scripts/discover_golden_candidates.py
"""

import asyncio
import json
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for pkg in ("storage", "contracts", "common", "pdf-tools"):
    sys.path.insert(0, str(_root / "packages" / pkg / "src"))

from research_kb_pdf import EmbeddingClient
from research_kb_storage import DatabaseConfig, SearchQuery, get_connection_pool
from research_kb_storage.search import search_hybrid


# Candidate queries for underrepresented domains
CANDIDATE_QUERIES = [
    # Software engineering (currently 0 entries)
    {"query": "API design patterns", "domain": "software_engineering", "difficulty": "easy"},
    {"query": "REST API versioning", "domain": "software_engineering", "difficulty": "easy"},
    {"query": "microservice architecture", "domain": "software_engineering", "difficulty": "medium"},
    {"query": "continuous integration pipeline", "domain": "software_engineering", "difficulty": "easy"},
    {"query": "code review best practices", "domain": "software_engineering", "difficulty": "medium"},
    {"query": "dependency injection pattern", "domain": "software_engineering", "difficulty": "easy"},
    {"query": "test driven development", "domain": "software_engineering", "difficulty": "easy"},
    {"query": "feature flags deployment", "domain": "software_engineering", "difficulty": "medium"},
    # Deep learning (currently 0 entries)
    {"query": "attention mechanism in transformers", "domain": "deep_learning", "difficulty": "easy"},
    {"query": "backpropagation algorithm", "domain": "deep_learning", "difficulty": "easy"},
    {"query": "batch normalization", "domain": "deep_learning", "difficulty": "easy"},
    {"query": "convolutional neural network architecture", "domain": "deep_learning", "difficulty": "easy"},
    {"query": "dropout regularization", "domain": "deep_learning", "difficulty": "easy"},
    {"query": "gradient vanishing problem", "domain": "deep_learning", "difficulty": "medium"},
    {"query": "transfer learning fine tuning", "domain": "deep_learning", "difficulty": "medium"},
    {"query": "vision transformer ViT", "domain": "deep_learning", "difficulty": "medium"},
    # Econometrics (currently 0 entries)
    {"query": "difference in differences", "domain": "econometrics", "difficulty": "easy"},
    {"query": "regression discontinuity design", "domain": "econometrics", "difficulty": "easy"},
    {"query": "fixed effects panel data", "domain": "econometrics", "difficulty": "medium"},
    {"query": "heteroskedasticity robust standard errors", "domain": "econometrics", "difficulty": "easy"},
    {"query": "omitted variable bias", "domain": "econometrics", "difficulty": "easy"},
    {"query": "endogeneity problem", "domain": "econometrics", "difficulty": "medium"},
    {"query": "two stage least squares", "domain": "econometrics", "difficulty": "medium"},
    # Statistics (currently 0 entries)
    {"query": "Bayesian prior posterior", "domain": "statistics", "difficulty": "easy"},
    {"query": "maximum likelihood estimation", "domain": "statistics", "difficulty": "easy"},
    {"query": "central limit theorem", "domain": "statistics", "difficulty": "easy"},
    {"query": "p-value hypothesis testing", "domain": "statistics", "difficulty": "easy"},
    {"query": "conformal prediction coverage", "domain": "statistics", "difficulty": "medium"},
    {"query": "bootstrap resampling", "domain": "statistics", "difficulty": "easy"},
    # Machine learning (currently 0 entries)
    {"query": "random forest feature importance", "domain": "machine_learning", "difficulty": "easy"},
    {"query": "cross validation model selection", "domain": "machine_learning", "difficulty": "easy"},
    {"query": "gradient boosting XGBoost", "domain": "machine_learning", "difficulty": "easy"},
    {"query": "bias variance tradeoff", "domain": "machine_learning", "difficulty": "easy"},
    {"query": "regularization L1 L2", "domain": "machine_learning", "difficulty": "easy"},
    {"query": "causal forest heterogeneous effects", "domain": "machine_learning", "difficulty": "medium"},
    # RAG/LLM (add harder queries)
    {"query": "retrieval augmented generation architecture", "domain": "rag_llm", "difficulty": "medium"},
    {"query": "chunk overlap strategy for RAG", "domain": "rag_llm", "difficulty": "medium"},
    {"query": "hallucination detection in LLMs", "domain": "rag_llm", "difficulty": "medium"},
    {"query": "prompt engineering techniques", "domain": "rag_llm", "difficulty": "easy"},
    {"query": "embedding model comparison for retrieval", "domain": "rag_llm", "difficulty": "hard"},
    {"query": "what assumptions does double machine learning require", "domain": "causal_inference", "difficulty": "hard"},
    # Causal inference (add harder queries)
    {"query": "SUTVA stable unit treatment value", "domain": "causal_inference", "difficulty": "medium"},
    {"query": "propensity score matching limitations", "domain": "causal_inference", "difficulty": "medium"},
    {"query": "mediation analysis direct indirect effects", "domain": "causal_inference", "difficulty": "medium"},
    # Time series (add harder queries)
    {"query": "ARIMA model selection AIC BIC", "domain": "time_series", "difficulty": "medium"},
    {"query": "stationarity unit root test", "domain": "time_series", "difficulty": "medium"},
    {"query": "exponential smoothing state space", "domain": "time_series", "difficulty": "medium"},
    # Finance (currently 0 entries)
    {"query": "Black-Scholes option pricing", "domain": "finance", "difficulty": "easy"},
    {"query": "GARCH volatility model", "domain": "finance", "difficulty": "easy"},
    {"query": "risk neutral pricing", "domain": "finance", "difficulty": "medium"},
]


async def main():
    pool = await get_connection_pool(DatabaseConfig())
    embed_client = EmbeddingClient()

    results = []
    seen_queries = set()

    for candidate in CANDIDATE_QUERIES:
        query = candidate["query"]
        domain = candidate["domain"]
        difficulty = candidate["difficulty"]

        if query in seen_queries:
            continue
        seen_queries.add(query)

        try:
            embedding = embed_client.embed_query(query)
        except Exception as e:
            print(f"  SKIP (embed error): {query} → {e}")
            continue

        sq = SearchQuery(
            text=query,
            embedding=embedding,
            limit=10,
            fts_weight=0.3,
            vector_weight=0.7,
            domain_id=domain,
        )

        try:
            search_results = await search_hybrid(sq)
        except Exception as e:
            print(f"  SKIP (search error): {query} → {e}")
            continue

        if not search_results:
            print(f"  NO RESULTS: {query} [{domain}]")
            continue

        # Take top 2 results as target chunks — check they're from the right domain
        targets = []
        source_title = None
        for r in search_results[:5]:
            chunk_id = str(r.chunk_id)
            # Verify domain match
            row = await pool.fetchrow(
                "SELECT s.title, s.metadata->>'domain' as domain "
                "FROM chunks c JOIN sources s ON c.source_id = s.id "
                "WHERE c.id = $1",
                r.chunk_id,
            )
            if row and row["domain"] == domain:
                targets.append(chunk_id)
                if not source_title:
                    source_title = row["title"]
                if len(targets) >= 2:
                    break

        if not targets:
            # Try without domain filter — the query might match a different domain's source
            print(f"  NO DOMAIN MATCH: {query} [{domain}] (top result domain: {search_results[0].source_title[:40] if search_results else '?'})")
            continue

        entry = {
            "query": query,
            "target_chunk_ids": targets,
            "domain": domain,
            "source_title": source_title[:80] if source_title else "Unknown",
            "difficulty": difficulty,
        }
        results.append(entry)
        status = "HIT" if len(targets) == 2 else "PARTIAL"
        print(f"  [{status:7s}] {query:50s} | {domain:25s} | {source_title[:40] if source_title else '?'}")

    print(f"\n  Generated {len(results)} golden entries from {len(CANDIDATE_QUERIES)} candidates.")

    # Write output
    output_path = _root / "fixtures" / "eval" / "golden_candidates_2026-02-19.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Written to: {output_path}")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
