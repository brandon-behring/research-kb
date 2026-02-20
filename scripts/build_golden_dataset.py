"""Build expanded golden dataset using SQL + FTS to find target chunks.

Finds chunks matching specific queries within target domains by joining
sources (with domain metadata) to chunks (with FTS vectors).

Outputs verified entries to fixtures/eval/golden_candidates_2026-02-19.json.

Usage:
    python scripts/build_golden_dataset.py
"""

import asyncio
import json
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for pkg in ("storage", "contracts", "common"):
    sys.path.insert(0, str(_root / "packages" / pkg / "src"))

from research_kb_storage import DatabaseConfig, get_connection_pool

# New golden entries to create
# Format: (query, domain, fts_search_terms, difficulty)
# fts_search_terms: what to search for in FTS to find relevant chunks
CANDIDATES = [
    # Software engineering
    ("API design patterns", "software_engineering", "API design pattern", "easy"),
    ("REST API versioning", "software_engineering", "API versioning REST", "easy"),
    ("microservice architecture", "software_engineering", "microservice", "medium"),
    (
        "continuous integration",
        "software_engineering",
        "continuous integration",
        "easy",
    ),
    ("dependency injection", "software_engineering", "dependency injection", "easy"),
    (
        "test driven development",
        "software_engineering",
        "test driven development TDD",
        "easy",
    ),
    (
        "feature flags deployment strategy",
        "software_engineering",
        "feature flag",
        "medium",
    ),
    (
        "GitHub Actions workflow",
        "software_engineering",
        "GitHub Actions workflow",
        "easy",
    ),
    # Deep learning
    (
        "attention mechanism in transformers",
        "deep_learning",
        "attention mechanism transformer",
        "easy",
    ),
    ("backpropagation algorithm", "deep_learning", "backpropagation gradient", "easy"),
    ("batch normalization technique", "deep_learning", "batch normalization", "easy"),
    (
        "convolutional neural network",
        "deep_learning",
        "convolutional neural network",
        "easy",
    ),
    ("dropout regularization", "deep_learning", "dropout regularization", "easy"),
    ("vanishing gradient problem", "deep_learning", "vanishing gradient", "medium"),
    ("transfer learning", "deep_learning", "transfer learning fine-tuning", "medium"),
    ("vision transformer ViT", "deep_learning", "vision transformer ViT", "medium"),
    # Econometrics
    (
        "difference in differences estimator",
        "econometrics",
        "difference in differences",
        "easy",
    ),
    (
        "regression discontinuity design",
        "econometrics",
        "regression discontinuity",
        "easy",
    ),
    ("fixed effects panel data", "econometrics", "fixed effects panel", "medium"),
    (
        "heteroskedasticity robust errors",
        "econometrics",
        "heteroskedasticity robust",
        "easy",
    ),
    ("omitted variable bias", "econometrics", "omitted variable bias", "easy"),
    ("endogeneity problem", "econometrics", "endogeneity", "medium"),
    (
        "two stage least squares",
        "econometrics",
        "two-stage least squares 2SLS",
        "medium",
    ),
    # Statistics
    ("Bayesian prior and posterior", "statistics", "prior posterior Bayesian", "easy"),
    (
        "maximum likelihood estimation",
        "statistics",
        "maximum likelihood estimation MLE",
        "easy",
    ),
    ("central limit theorem", "statistics", "central limit theorem", "easy"),
    ("p-value hypothesis testing", "statistics", "p-value hypothesis test", "easy"),
    (
        "conformal prediction coverage",
        "statistics",
        "conformal prediction coverage",
        "medium",
    ),
    ("bootstrap resampling method", "statistics", "bootstrap resampling", "easy"),
    # Machine learning
    (
        "random forest feature importance",
        "machine_learning",
        "random forest feature importance",
        "easy",
    ),
    (
        "cross validation model selection",
        "machine_learning",
        "cross-validation model selection",
        "easy",
    ),
    ("gradient boosting", "machine_learning", "gradient boosting XGBoost", "easy"),
    ("bias variance tradeoff", "machine_learning", "bias variance tradeoff", "easy"),
    (
        "L1 L2 regularization",
        "machine_learning",
        "regularization L1 L2 lasso ridge",
        "easy",
    ),
    (
        "causal forest heterogeneous effects",
        "machine_learning",
        "causal forest heterogeneous treatment",
        "medium",
    ),
    # RAG/LLM (add harder queries — domain already well-represented)
    (
        "retrieval augmented generation architecture",
        "rag_llm",
        "retrieval augmented generation RAG",
        "medium",
    ),
    ("chunking strategy for RAG", "rag_llm", "chunking overlap strategy", "medium"),
    (
        "hallucination detection in LLMs",
        "rag_llm",
        "hallucination detection LLM",
        "medium",
    ),
    ("prompt engineering techniques", "rag_llm", "prompt engineering", "easy"),
    ("embedding model for retrieval", "rag_llm", "embedding model retrieval", "hard"),
    # Causal inference (add harder queries)
    (
        "what assumptions does DML require",
        "causal_inference",
        "double machine learning assumptions",
        "hard",
    ),
    (
        "SUTVA assumption",
        "causal_inference",
        "SUTVA stable unit treatment value",
        "medium",
    ),
    (
        "propensity score matching",
        "causal_inference",
        "propensity score matching",
        "medium",
    ),
    (
        "mediation analysis",
        "causal_inference",
        "mediation direct indirect effect",
        "medium",
    ),
    # Time series (add harder queries)
    ("ARIMA model selection", "time_series", "ARIMA model selection AIC BIC", "medium"),
    (
        "unit root test stationarity",
        "time_series",
        "unit root stationarity Dickey-Fuller",
        "medium",
    ),
    (
        "exponential smoothing",
        "time_series",
        "exponential smoothing state space",
        "medium",
    ),
    # Finance
    ("Black-Scholes option pricing", "finance", "Black-Scholes option pricing", "easy"),
    ("GARCH volatility model", "finance", "GARCH volatility", "easy"),
    ("risk neutral pricing", "finance", "risk neutral pricing", "medium"),
]


async def main():
    pool = await get_connection_pool(DatabaseConfig())
    results = []

    for query, domain, fts_terms, difficulty in CANDIDATES:
        # Find chunks using FTS + domain join on sources
        rows = await pool.fetch(
            """
            SELECT c.id, c.content, s.title,
                   ts_rank(c.fts_vector, plainto_tsquery('english', $1)) as rank
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE s.metadata->>'domain' = $2
              AND c.fts_vector @@ plainto_tsquery('english', $1)
            ORDER BY rank DESC
            LIMIT 5
            """,
            fts_terms,
            domain,
        )

        if not rows:
            # Try broader search without domain for debugging
            any_rows = await pool.fetch(
                """
                SELECT c.id, s.title, s.metadata->>'domain' as actual_domain,
                       ts_rank(c.fts_vector, plainto_tsquery('english', $1)) as rank
                FROM chunks c
                JOIN sources s ON c.source_id = s.id
                WHERE c.fts_vector @@ plainto_tsquery('english', $1)
                ORDER BY rank DESC
                LIMIT 3
                """,
                fts_terms,
            )
            if any_rows:
                actual = any_rows[0]["actual_domain"]
                print(
                    f"  DOMAIN MISS: {query:50s} | expected={domain}, found in {actual} "
                    f"({any_rows[0]['title'][:40]})"
                )
            else:
                print(f"  NO FTS HIT:  {query:50s} | no chunks match '{fts_terms}'")
            continue

        # Take top 2 chunks as targets
        target_ids = [str(rows[0]["id"])]
        source_title = rows[0]["title"]

        # Prefer second chunk from same source if available
        for r in rows[1:]:
            if str(r["id"]) != target_ids[0]:
                target_ids.append(str(r["id"]))
                break

        entry = {
            "query": query,
            "target_chunk_ids": target_ids,
            "domain": domain,
            "source_title": source_title[:80],
            "difficulty": difficulty,
        }
        results.append(entry)
        print(f"  [{len(target_ids)} chunks] {query:50s} | {domain:25s} | {source_title[:40]}")

    print(f"\n  Generated {len(results)} golden entries from {len(CANDIDATES)} candidates.")

    # Load existing golden dataset and merge
    existing_path = _root / "fixtures" / "eval" / "golden_dataset.json"
    with open(existing_path) as f:
        existing = json.load(f)

    existing_queries = {e["query"] for e in existing}
    new_entries = [e for e in results if e["query"] not in existing_queries]

    combined = existing + new_entries
    print(f"  Existing: {len(existing)}, New: {len(new_entries)}, Combined: {len(combined)}")

    # Write combined dataset
    with open(existing_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Written to: {existing_path}")

    # Also save just the new entries for reference
    new_path = _root / "fixtures" / "eval" / "golden_candidates_2026-02-19.json"
    with open(new_path, "w") as f:
        json.dump(new_entries, f, indent=2)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
