"""Discover candidate chunks for expanding the golden eval dataset.

Searches across domains to find chunks matching specific queries,
producing verified (query, chunk_id, domain, difficulty) triples.

Supports two search modes:
  - FTS (default): PostgreSQL full-text search, no external services needed
  - Hybrid: FTS + vector similarity (requires embed_server)

Usage:
    python scripts/discover_golden_candidates.py              # Write candidates + merge
    python scripts/discover_golden_candidates.py --dry-run    # Preview only
    python scripts/discover_golden_candidates.py --hybrid     # Use embedding search
"""

import argparse
import asyncio
import json
import sys
import textwrap
from pathlib import Path

_root = Path(__file__).parent.parent
for pkg in ("storage", "contracts", "common", "pdf-tools"):
    sys.path.insert(0, str(_root / "packages" / pkg / "src"))

from research_kb_storage import DatabaseConfig, get_connection_pool

# ---------------------------------------------------------------------------
# Candidate queries — 87 new queries for golden dataset expansion
# ---------------------------------------------------------------------------
# Format: {"query": str, "domain": str, "difficulty": str, "fts_terms": str}
#
# fts_terms: alternative search terms for FTS (handles cases where the query
# phrasing doesn't match PostgreSQL tsquery well).
#
# Excludes queries already in golden_dataset.json as of 2026-02-20.
# ---------------------------------------------------------------------------

CANDIDATE_QUERIES = [
    # ===== NEW DOMAINS (38 queries) =====
    # --- mathematics (8) ---
    {
        "query": "eigenvalue decomposition",
        "domain": "mathematics",
        "difficulty": "easy",
        "fts_terms": "eigenvalue decomposition",
    },
    {
        "query": "proof by induction",
        "domain": "mathematics",
        "difficulty": "easy",
        "fts_terms": "proof induction",
    },
    {
        "query": "linear algebra matrix factorization",
        "domain": "mathematics",
        "difficulty": "easy",
        "fts_terms": "matrix factorization",
    },
    {
        "query": "convex optimization gradient descent",
        "domain": "machine_learning",
        "difficulty": "easy",
        "fts_terms": "convex optimization gradient",
    },
    {
        "query": "measure theory probability",
        "domain": "mathematics",
        "difficulty": "medium",
        "fts_terms": "measure theory probability",
    },
    {
        "query": "real analysis convergence",
        "domain": "mathematics",
        "difficulty": "medium",
        "fts_terms": "convergence analysis",
    },
    {
        "query": "Fourier transform signal processing",
        "domain": "mathematics",
        "difficulty": "medium",
        "fts_terms": "Fourier transform",
    },
    {
        "query": "topology continuity compactness",
        "domain": "mathematics",
        "difficulty": "hard",
        "fts_terms": "topology continuity compact",
    },
    # --- interview_prep (7) ---
    {
        "query": "system design interview",
        "domain": "interview_prep",
        "difficulty": "easy",
        "fts_terms": "system design interview",
    },
    {
        "query": "behavioral interview STAR method",
        "domain": "interview_prep",
        "difficulty": "easy",
        "fts_terms": "behavioral interview STAR",
    },
    # "data structures array vs linked list" dropped — corpus book (Wengrow) mis-tagged as rag_llm
    {
        "query": "concurrency threading interview",
        "domain": "ml_engineering",
        "difficulty": "medium",
        "fts_terms": "concurrency threading",
    },
    {
        "query": "machine learning interview feature engineering",
        "domain": "interview_prep",
        "difficulty": "medium",
        "fts_terms": "machine learning interview feature",
    },
    {
        "query": "distributed systems consistency models",
        "domain": "interview_prep",
        "difficulty": "medium",
        "fts_terms": "distributed consistency",
    },
    {
        "query": "algorithm complexity analysis interview",
        "domain": "interview_prep",
        "difficulty": "hard",
        "fts_terms": "algorithm complexity analysis",
    },
    # --- finance (7) ---
    {
        "query": "Black-Scholes option pricing",
        "domain": "finance",
        "difficulty": "easy",
        "fts_terms": "Black-Scholes option pricing",
    },
    {
        "query": "GARCH volatility model",
        "domain": "finance",
        "difficulty": "easy",
        "fts_terms": "GARCH volatility",
    },
    {
        "query": "capital asset pricing model CAPM",
        "domain": "finance",
        "difficulty": "easy",
        "fts_terms": "capital asset pricing CAPM",
    },
    {
        "query": "portfolio optimization Markowitz",
        "domain": "mathematics",
        "difficulty": "medium",
        "fts_terms": "portfolio optimization Markowitz",
    },
    {
        "query": "value at risk VaR",
        "domain": "finance",
        "difficulty": "medium",
        "fts_terms": "value at risk VaR",
    },
    {
        "query": "risk neutral pricing derivatives",
        "domain": "finance",
        "difficulty": "medium",
        "fts_terms": "risk neutral pricing derivative",
    },
    {
        "query": "stochastic calculus Ito lemma",
        "domain": "finance",
        "difficulty": "hard",
        "fts_terms": "stochastic differential equation",
    },
    # --- ml_engineering (6) ---
    {
        "query": "model deployment serving",
        "domain": "ml_engineering",
        "difficulty": "easy",
        "fts_terms": "model deployment serving",
    },
    {
        "query": "ML pipeline orchestration",
        "domain": "ml_engineering",
        "difficulty": "easy",
        "fts_terms": "pipeline orchestration",
    },
    {
        "query": "feature store architecture",
        "domain": "ml_engineering",
        "difficulty": "medium",
        "fts_terms": "feature store",
    },
    {
        "query": "model monitoring drift detection",
        "domain": "ml_engineering",
        "difficulty": "medium",
        "fts_terms": "model monitoring drift",
    },
    {
        "query": "experiment tracking MLflow",
        "domain": "ml_engineering",
        "difficulty": "medium",
        "fts_terms": "experiment tracking",
    },
    {
        "query": "A/B testing for ML models",
        "domain": "ml_engineering",
        "difficulty": "hard",
        "fts_terms": "A/B testing model",
    },
    # --- data_science (5) ---
    {
        "query": "exploratory data analysis",
        "domain": "data_science",
        "difficulty": "easy",
        "fts_terms": "exploratory data analysis",
    },
    {
        "query": "feature selection methods",
        "domain": "data_science",
        "difficulty": "easy",
        "fts_terms": "feature selection",
    },
    {
        "query": "missing data imputation",
        "domain": "data_science",
        "difficulty": "medium",
        "fts_terms": "missing data imputation",
    },
    {
        "query": "data pipeline ETL best practices",
        "domain": "data_science",
        "difficulty": "medium",
        "fts_terms": "data pipeline ETL",
    },
    {
        "query": "class imbalance handling strategies",
        "domain": "data_science",
        "difficulty": "hard",
        "fts_terms": "class imbalance",
    },
    # --- portfolio_management (5) ---
    {
        "query": "asset allocation strategy",
        "domain": "portfolio_management",
        "difficulty": "easy",
        "fts_terms": "asset allocation",
    },
    {
        "query": "Sharpe ratio risk adjusted return",
        "domain": "portfolio_management",
        "difficulty": "easy",
        "fts_terms": "Sharpe ratio",
    },
    {
        "query": "factor investing models",
        "domain": "portfolio_management",
        "difficulty": "medium",
        "fts_terms": "factor investing",
    },
    {
        "query": "rebalancing portfolio strategy",
        "domain": "portfolio_management",
        "difficulty": "medium",
        "fts_terms": "rebalancing portfolio",
    },
    {
        "query": "portfolio risk parity approach",
        "domain": "portfolio_management",
        "difficulty": "hard",
        "fts_terms": "risk parity",
    },
    # ===== MEDIUM/HARD ADDITIONS FOR EXISTING DOMAINS (49 queries) =====
    # --- causal_inference (+10: 4 medium, 4 hard single, 2 hard cross-domain) ---
    {
        "query": "SUTVA stable unit treatment value",
        "domain": "causal_inference",
        "difficulty": "medium",
        "fts_terms": "SUTVA stable unit treatment value",
    },
    {
        "query": "propensity score matching limitations",
        "domain": "causal_inference",
        "difficulty": "medium",
        "fts_terms": "propensity score matching limitation",
    },
    {
        "query": "mediation analysis direct indirect effects",
        "domain": "causal_inference",
        "difficulty": "medium",
        "fts_terms": "mediation direct indirect effect",
    },
    {
        "query": "causal forest heterogeneous effects estimation",
        "domain": "causal_inference",
        "difficulty": "medium",
        "fts_terms": "causal forest heterogeneous treatment effect",
    },
    {
        "query": "sensitivity analysis for unobserved confounders",
        "domain": "causal_inference",
        "difficulty": "hard",
        "fts_terms": "sensitivity analysis unobserved confounder",
    },
    {
        "query": "causal discovery from observational data",
        "domain": "causal_inference",
        "difficulty": "hard",
        "fts_terms": "causal discovery observational",
    },
    {
        "query": "transportability of causal effects across populations",
        "domain": "causal_inference",
        "difficulty": "hard",
        "fts_terms": "transportability causal effect population",
    },
    {
        "query": "targeted learning TMLE semiparametric",
        "domain": "causal_inference",
        "difficulty": "hard",
        "fts_terms": "targeted learning TMLE semiparametric",
    },
    {
        "query": "when to use IV versus difference in differences",
        "domain": "causal_inference",
        "difficulty": "hard",
        "fts_terms": "instrumental variable difference-in-differences",
    },
    {
        "query": "Bayesian causal inference with informative priors",
        "domain": "causal_inference",
        "difficulty": "hard",
        "fts_terms": "Bayesian causal inference prior",
    },
    # --- rag_llm (+6: 1 medium, 3 hard single, 2 hard cross-domain) ---
    # NOTE: "retrieval augmented generation architecture" and "hallucination detection
    # in LLMs" already in golden_dataset — excluded as exact duplicates
    {
        "query": "chunk overlap strategy for RAG",
        "domain": "rag_llm",
        "difficulty": "medium",
        "fts_terms": "chunk overlap retrieval",
    },
    # "query routing multi-index retrieval" dropped — no domain-matched content in corpus
    {
        "query": "long context versus RAG tradeoffs",
        "domain": "rag_llm",
        "difficulty": "hard",
        "fts_terms": "context window retrieval",
    },
    {
        "query": "evaluating retrieval quality without relevance labels",
        "domain": "rag_llm",
        "difficulty": "hard",
        "fts_terms": "retrieval quality evaluation relevance",
    },
    {
        "query": "embedding model selection for domain-specific corpora",
        "domain": "rag_llm",
        "difficulty": "hard",
        "fts_terms": "embedding model domain-specific corpus",
    },
    {
        "query": "fine-tuning retriever vs generator in RAG",
        "domain": "rag_llm",
        "difficulty": "hard",
        "fts_terms": "fine-tuning retriever generator RAG",
    },
    # --- time_series (+7: 3 medium, 2 hard single, 2 hard cross-domain) ---
    {
        "query": "ARIMA model selection AIC BIC",
        "domain": "time_series",
        "difficulty": "medium",
        "fts_terms": "ARIMA model selection AIC BIC",
    },
    {
        "query": "exponential smoothing state space",
        "domain": "time_series",
        "difficulty": "medium",
        "fts_terms": "exponential smoothing state space",
    },
    {
        "query": "stationarity unit root test",
        "domain": "time_series",
        "difficulty": "medium",
        "fts_terms": "stationarity unit root test",
    },
    {
        "query": "structural breaks vs regime switching",
        "domain": "finance",
        "difficulty": "hard",
        "fts_terms": "structural break regime switching",
    },
    {
        "query": "multivariate Granger causality interpretation",
        "domain": "time_series",
        "difficulty": "hard",
        "fts_terms": "multivariate Granger causality",
    },
    {
        "query": "GARCH models for financial volatility",
        "domain": "time_series",
        "difficulty": "hard",
        "fts_terms": "GARCH financial volatility",
    },
    {
        "query": "causal impact analysis time series",
        "domain": "time_series",
        "difficulty": "hard",
        "fts_terms": "causal impact time series",
    },
    # --- econometrics (+6: 2 medium, 2 hard single, 2 hard cross-domain) ---
    {
        "query": "panel data fixed vs random effects",
        "domain": "econometrics",
        "difficulty": "medium",
        "fts_terms": "panel data fixed random effects",
    },
    {
        "query": "quantile regression interpretation",
        "domain": "econometrics",
        "difficulty": "medium",
        "fts_terms": "quantile regression interpretation",
    },
    {
        "query": "weak instruments problem and detection",
        "domain": "econometrics",
        "difficulty": "hard",
        "fts_terms": "weak instrument detection",
    },
    {
        "query": "synthetic control method assumptions",
        "domain": "econometrics",
        "difficulty": "hard",
        "fts_terms": "synthetic control assumption",
    },
    {
        "query": "double machine learning for treatment effects",
        "domain": "econometrics",
        "difficulty": "hard",
        "fts_terms": "double machine learning treatment effect",
    },
    {
        "query": "heterogeneous treatment effects econometric methods",
        "domain": "econometrics",
        "difficulty": "hard",
        "fts_terms": "heterogeneous treatment effect econometric",
    },
    # --- software_engineering (+5: 3 medium, 2 hard) ---
    {
        "query": "event driven architecture patterns",
        "domain": "software_engineering",
        "difficulty": "medium",
        "fts_terms": "event driven architecture",
    },
    {
        "query": "database indexing optimization",
        "domain": "software_engineering",
        "difficulty": "medium",
        "fts_terms": "database indexing optimization",
    },
    {
        "query": "distributed systems consensus algorithms",
        "domain": "software_engineering",
        "difficulty": "medium",
        "fts_terms": "consensus algorithm distributed",
    },
    {
        "query": "technical debt measurement and management",
        "domain": "software_engineering",
        "difficulty": "hard",
        "fts_terms": "technical debt measurement",
    },
    {
        "query": "system design scalability patterns",
        "domain": "software_engineering",
        "difficulty": "hard",
        "fts_terms": "scalability design pattern",
    },
    # --- deep_learning (+5: 2 medium, 2 hard single, 1 hard cross-domain) ---
    {
        "query": "learning rate scheduling strategies",
        "domain": "deep_learning",
        "difficulty": "medium",
        "fts_terms": "learning rate schedule",
    },
    {
        "query": "knowledge distillation model compression",
        "domain": "deep_learning",
        "difficulty": "medium",
        "fts_terms": "knowledge distillation compression",
    },
    {
        "query": "neural architecture search methods",
        "domain": "deep_learning",
        "difficulty": "hard",
        "fts_terms": "neural architecture search",
    },
    {
        "query": "self-supervised contrastive learning",
        "domain": "deep_learning",
        "difficulty": "hard",
        "fts_terms": "self-supervised contrastive learning",
    },
    {
        "query": "transformer attention for time series",
        "domain": "deep_learning",
        "difficulty": "hard",
        "fts_terms": "transformer attention time series",
    },
    # --- machine_learning (+5: 2 medium, 2 hard single, 1 hard cross-domain) ---
    {
        "query": "ensemble methods stacking vs blending",
        "domain": "machine_learning",
        "difficulty": "medium",
        "fts_terms": "ensemble stacking blending",
    },
    {
        "query": "hyperparameter optimization Bayesian",
        "domain": "machine_learning",
        "difficulty": "medium",
        "fts_terms": "hyperparameter optimization Bayesian",
    },
    {
        "query": "fairness constraints in ML models",
        "domain": "machine_learning",
        "difficulty": "hard",
        "fts_terms": "fairness constraint model",
    },
    {
        "query": "interpretable ML SHAP vs LIME",
        "domain": "machine_learning",
        "difficulty": "hard",
        "fts_terms": "SHAP LIME interpretable",
    },
    {
        "query": "causal ML uplift modeling",
        "domain": "causal_inference",
        "difficulty": "hard",
        "fts_terms": "causal uplift modeling",
    },
    # --- statistics (+5: 1 medium, 2 hard single, 2 hard cross-domain) ---
    {
        "query": "survival analysis Kaplan-Meier",
        "domain": "causal_inference",
        "difficulty": "medium",
        "fts_terms": "survival analysis Kaplan-Meier",
    },
    {
        "query": "multiple testing correction Bonferroni vs FDR",
        "domain": "statistics",
        "difficulty": "hard",
        "fts_terms": "multiple testing Bonferroni FDR",
    },
    {
        "query": "nonparametric hypothesis testing",
        "domain": "statistics",
        "difficulty": "hard",
        "fts_terms": "nonparametric hypothesis test",
    },
    {
        "query": "causal interpretation of regression coefficients",
        "domain": "statistics",
        "difficulty": "hard",
        "fts_terms": "causal interpretation regression coefficient",
    },
    {
        "query": "Bayesian hierarchical modeling multilevel",
        "domain": "statistics",
        "difficulty": "hard",
        "fts_terms": "Bayesian hierarchical multilevel",
    },
]


async def search_fts(pool, fts_terms: str, domain: str, limit: int = 5):
    """Search using PostgreSQL full-text search within a domain."""
    rows = await pool.fetch(
        """
        SELECT c.id, c.content, s.title,
               ts_rank(c.fts_vector, plainto_tsquery('english', $1)) as rank
        FROM chunks c
        JOIN sources s ON c.source_id = s.id
        WHERE s.metadata->>'domain' = $2
          AND c.fts_vector @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC
        LIMIT $3
        """,
        fts_terms,
        domain,
        limit,
    )
    return rows


async def search_fts_any_domain(pool, fts_terms: str, limit: int = 3):
    """Search using FTS without domain filter (for debugging misses)."""
    return await pool.fetch(
        """
        SELECT c.id, s.title, s.metadata->>'domain' as actual_domain,
               ts_rank(c.fts_vector, plainto_tsquery('english', $1)) as rank
        FROM chunks c
        JOIN sources s ON c.source_id = s.id
        WHERE c.fts_vector @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC
        LIMIT $2
        """,
        fts_terms,
        limit,
    )


async def discover_candidates(pool, dry_run: bool = False):
    """Discover golden candidates using FTS search.

    Args:
        pool: Database connection pool.
        dry_run: If True, print preview with chunk snippets but don't write files.

    Returns:
        List of verified candidate entries.
    """
    results = []
    seen_queries = set()
    stats = {"hit": 0, "partial": 0, "domain_miss": 0, "no_fts": 0, "duplicate": 0}

    for candidate in CANDIDATE_QUERIES:
        query = candidate["query"]
        domain = candidate["domain"]
        difficulty = candidate["difficulty"]
        fts_terms = candidate.get("fts_terms", query)

        if query in seen_queries:
            stats["duplicate"] += 1
            continue
        seen_queries.add(query)

        # Search within target domain using FTS
        rows = await search_fts(pool, fts_terms, domain)

        if not rows:
            # Debug: check if content exists in other domains
            any_rows = await search_fts_any_domain(pool, fts_terms)
            if any_rows:
                actual = any_rows[0]["actual_domain"]
                stats["domain_miss"] += 1
                print(
                    f"  DOMAIN MISS: {query:55s} | expected={domain}, "
                    f"found in {actual} ({any_rows[0]['title'][:40]})"
                )
            else:
                stats["no_fts"] += 1
                print(f"  NO FTS HIT:  {query:55s} | no chunks match '{fts_terms}'")
            continue

        # Take top 2 chunks as targets
        target_ids = [str(rows[0]["id"])]
        source_title = rows[0]["title"]

        # Prefer second chunk from different position in same source
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

        n_chunks = len(target_ids)
        status = "HIT" if n_chunks == 2 else "PARTIAL"
        stats["hit" if n_chunks == 2 else "partial"] += 1

        if dry_run:
            # Show rich preview with chunk snippet
            snippet = rows[0]["content"][:200].replace("\n", " ")
            print(f"  [{status:7s}] {query:55s} | {domain:25s} | {difficulty:6s}")
            print(f"           Source: {source_title[:60]}")
            print(f"           Snippet: {textwrap.shorten(snippet, 120)}")
            print()
        else:
            print(f"  [{status:7s}] {query:55s} | {domain:25s} | " f"{source_title[:40]}")

    print(f"\n  --- Discovery Summary ---")
    print(f"  Candidates searched: {len(CANDIDATE_QUERIES)}")
    print(f"  Hits (2 chunks):     {stats['hit']}")
    print(f"  Partial (1 chunk):   {stats['partial']}")
    print(f"  Domain misses:       {stats['domain_miss']}")
    print(f"  No FTS hits:         {stats['no_fts']}")
    print(f"  Duplicates skipped:  {stats['duplicate']}")
    print(f"  Total verified:      {len(results)}")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Discover golden eval dataset candidates")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview candidates with chunk snippets without writing files",
    )
    args = parser.parse_args()

    pool = await get_connection_pool(DatabaseConfig())

    try:
        results = await discover_candidates(pool, dry_run=args.dry_run)

        if args.dry_run:
            # Show difficulty distribution of found candidates
            from collections import Counter

            diffs = Counter(r["difficulty"] for r in results)
            domains = Counter(r["domain"] for r in results)
            print(f"\n  --- Candidate Distribution ---")
            print(f"  Difficulty: {dict(sorted(diffs.items()))}")
            print(f"  Domains ({len(domains)}): {dict(sorted(domains.items()))}")
            print(f"\n  (Dry run — no files written)")
            return

        # Write candidates file
        candidates_path = _root / "fixtures" / "eval" / "golden_candidates_2026-02-20.json"
        with open(candidates_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Candidates written to: {candidates_path}")

        # Merge into golden dataset
        golden_path = _root / "fixtures" / "eval" / "golden_dataset.deprecated.json"
        with open(golden_path) as f:
            existing = json.load(f)

        existing_queries = {e["query"] for e in existing}
        new_entries = [e for e in results if e["query"] not in existing_queries]

        combined = existing + new_entries
        with open(golden_path, "w") as f:
            json.dump(combined, f, indent=2)

        print(f"  Existing entries:    {len(existing)}")
        print(f"  New entries added:   {len(new_entries)}")
        print(f"  Skipped (duplicate): {len(results) - len(new_entries)}")
        print(f"  Combined total:      {len(combined)}")
        print(f"  Written to: {golden_path}")

        # Show final distribution
        from collections import Counter

        diffs = Counter(e["difficulty"] for e in combined)
        domains = Counter(e["domain"] for e in combined)
        total = len(combined)
        print(f"\n  --- Final Distribution ---")
        for d in ("easy", "medium", "hard"):
            c = diffs.get(d, 0)
            print(f"  {d:8s}: {c:3d} ({100 * c / total:.0f}%)")
        print(f"  Domains ({len(domains)}):")
        for d, c in sorted(domains.items(), key=lambda x: -x[1]):
            print(f"    {d:25s}: {c}")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
