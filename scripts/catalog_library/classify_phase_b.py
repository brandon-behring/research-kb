#!/usr/bin/env python3
"""
Phase B: Enhanced classification for entries that Stage 4 couldn't classify.

Uses aggressive title/filename pattern matching + OpenLibrary metadata.
Outputs classified_results.json for merge into the enrichment pipeline.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Title/filename keyword patterns → domain
# Broader than text_sample keywords since titles are more informative
# ---------------------------------------------------------------------------

TITLE_PATTERNS: list[tuple[str, list[str]]] = [
    # Finance / CFA
    (
        "finance",
        [
            "cfa ",
            "cfa®",
            "chartered financial",
            "derivatives",
            "fixed income",
            "equity valuation",
            "portfolio theory",
            "corporate finance",
            "financial markets",
            "financial engineering",
            "options futures",
            "bond math",
            "quantitative finance",
            "financial risk",
            "asset pricing",
            "black-scholes",
            "hedge fund",
            "investment",
            "forex",
            "trading",
            "banking",
            "financial accounting",
            "mergers acquisitions",
            "enterprise risk management",
            "internal controls",
            "valuation",
            "equity markets",
            "wiley finance",
            "investing for",
            "financial statement",
            "venture capital",
            "private equity",
            "capital markets",
            "security analysis",
            "value investing",
            "financial modeling",
        ],
    ),
    # Portfolio management
    (
        "portfolio_management",
        [
            "portfolio management",
            "portfolio construction",
            "portfolio optimization",
            "asset allocation",
            "modern portfolio",
            "factor investing",
            "wealth management",
            "fund management",
        ],
    ),
    # Actuarial / Insurance
    (
        "actuarial_insurance",
        [
            "actuarial",
            "insurance mathematics",
            "loss model",
            "life contingencies",
            "risk theory insurance",
        ],
    ),
    # Econometrics / Economics
    (
        "econometrics",
        [
            "econometric",
            "panel data econometric",
            "microeconometrics",
            "macroeconometrics",
            "time series econometric",
            "microeconomic",
            "macroeconomic",
            "economics",
        ],
    ),
    # Causal inference
    (
        "causal_inference",
        [
            "causal inference",
            "causal effect",
            "counterfactual",
            "instrumental variable",
            "treatment effect",
            "potential outcomes",
            "rubin causal",
            "causality",
        ],
    ),
    # Statistics
    (
        "statistics",
        [
            "statistical inference",
            "bayesian statistics",
            "mathematical statistics",
            "statistical learning",
            "biostatistics",
            "regression analysis",
            "hypothesis testing",
            "statistical methods",
            "experimental design",
            "all of statistics",
            "statistical rethinking",
            "bayesian optimization",
            "bayesian data analysis",
            "bayesian ",
        ],
    ),
    # Probability theory
    (
        "probability_theory",
        [
            "probability theory",
            "stochastic process",
            "markov chain",
            "random variable",
            "measure theory probability",
            "brownian motion",
            "martingale",
            "stochastic calculus",
            "probability measure",
            "stochastic differential",
            "probability and statistics",
            "probability ",
            "stochastic",
        ],
    ),
    # Machine learning
    (
        "machine_learning",
        [
            "machine learning",
            "pattern recognition",
            "supervised learning",
            "unsupervised learning",
            "classification and regression",
            "ensemble method",
            "random forest",
            "gradient boosting",
            "xgboost",
            "feature engineering",
            "scikit-learn",
        ],
    ),
    # Deep learning
    (
        "deep_learning",
        [
            "deep learning",
            "neural network",
            "convolutional neural",
            "recurrent neural",
            "transformer model",
            "attention mechanism",
            "generative adversarial",
            "autoencoder",
            "pytorch",
            "tensorflow",
            "backpropagation",
            "deep neural",
        ],
    ),
    # Reinforcement learning
    (
        "reinforcement_learning",
        [
            "reinforcement learning",
            "multi-armed bandit",
            "markov decision process",
            "q-learning",
            "policy gradient",
            "deep reinforcement",
            "reward shaping",
        ],
    ),
    # RAG / LLM / NLP
    (
        "rag_llm",
        [
            "large language model",
            "natural language processing",
            "retrieval augmented",
            "prompt engineering",
            "text mining",
            "sentiment analysis",
            "word embedding",
            "language model",
            "transformer architecture",
            "chatbot",
            "conversational ai",
            "llm",
            "gpt",
            "bert model",
            "speech and language processing",
            "ai-powered search",
            "ai_powered_search",
            "generative ai",
            "generative_ai",
            "language processing",
        ],
    ),
    # Data science
    (
        "data_science",
        [
            "data science",
            "data analysis",
            "data wrangling",
            "data engineering",
            "pandas",
            "data visualization",
            "exploratory data",
            "data pipeline",
            "data-driven",
            "data mining",
            "analytics",
            "big data",
            "apache spark",
            "spark in action",
            "spark_in_action",
            "data lake",
            "data warehouse",
            "data engineer study guide",
            "data professional",
            "pandas in action",
        ],
    ),
    # Software engineering
    (
        "software_engineering",
        [
            "software engineering",
            "design patterns",
            "clean code",
            "refactoring",
            "software architecture",
            "microservice",
            "domain-driven",
            "continuous integration",
            "devops",
            "version control",
            "agile",
            "software testing",
            "web development",
            "api design",
            "rest api",
            "cloud computing",
            "aws ",
            "azure ",
            "kubernetes",
            "docker",
            "terraform",
            "infrastructure as code",
            "security",
            "cybersecurity",
            "penetration testing",
            "networking",
            "system design",
            "distributed system",
            "operating system",
            "linux ",
            "unix ",
            "pragmatic programmer",
            "head first python",
            "head first java",
            "head first",
            "python programming",
            "programming with c",
            "absolute c++",
            "c++ ",
            "programming language",
            "computer network",
            "computer science",
            "introduction to computer",
            "design for developer",
            "creative programmer",
            "the creative_programmer",
            "design_for_developer",
            "computer programming",
            "java programming",
            "go programming",
            "rust programming",
            "python cookbook",
            "javascript",
            "typescript",
            "web scraping",
            "git ",
            "github",
            "code complete",
            "program curriculum",
            "grokking",
        ],
    ),
    # Functional programming
    (
        "functional_programming",
        [
            "functional programming",
            "haskell",
            "scala ",
            "lambda calculus",
            "category theory programming",
            "purely functional",
            "monads",
            "type theory",
            "clojure",
            "erlang",
            "elixir programming",
            "lisp ",
            "scheme programming",
            "racket programming",
        ],
    ),
    # ML engineering
    (
        "ml_engineering",
        [
            "ml engineering",
            "mlops",
            "machine learning engineering",
            "model deployment",
            "ml pipeline",
            "ml system",
            "feature store",
            "model serving",
        ],
    ),
    # SQL
    (
        "sql",
        [
            "sql ",
            "database design",
            "relational database",
            "postgresql",
            "mysql",
            "data modeling database",
        ],
    ),
    # Recommender systems
    (
        "recommender_systems",
        [
            "recommender system",
            "recommendation system",
            "collaborative filtering",
            "content-based filtering",
        ],
    ),
    # Interview prep
    (
        "interview_prep",
        [
            "interview",
            "coding interview",
            "cracking the",
            "system design interview",
            "behavioral interview",
            "leetcode",
            "competitive programming",
        ],
    ),
    # Algorithms
    (
        "algorithms",
        [
            "algorithm design",
            "introduction to algorithms",
            "data structures and algorithm",
            "graph algorithm",
            "dynamic programming",
            "computational complexity",
            "combinatorial optimization",
            "algorithmic",
            "algorithm analysis",
            "algorithms illuminated",
            "optimization algorithm",
            "optimization_algorithm",
            "data structures",
        ],
    ),
    # Fitness
    (
        "fitness",
        [
            "fitness",
            "exercise",
            "strength training",
            "bodybuilding",
            "workout",
            "nutrition",
            "sports science",
            "kinesiology",
            "yoga",
            "running ",
            "marathon",
            "physical training",
            "starting strength",
            "kettlebell",
            "low back disorder",
            "bike repair",
            "stretching",
            "flexibility",
            "weight training",
            "powerlifting",
        ],
    ),
    # Physics
    (
        "physics",
        [
            "physics",
            "quantum mechanics",
            "electrodynamics",
            "thermodynamics",
            "classical mechanics",
            "relativity",
            "particle physics",
            "solid state",
            "condensed matter",
            "quantum field",
            "statistical mechanics",
            "astrophysics",
            "cosmology",
            "optics",
            "electromagnetic",
            "photonics",
            "fluid dynamics",
            "fluid mechanics",
            "elementary particles",
            "quantum theory",
            "atmospheric and oceanic",
            "atmospheric fluid",
            "classical and quantum dynamics",
        ],
    ),
    # Dynamical systems
    (
        "dynamical_systems",
        [
            "dynamical system",
            "chaos theory",
            "nonlinear dynamics",
            "differential equation",
            "ordinary differential",
            "partial differential",
            "bifurcation",
            "ergodic",
            "control theory",
            "stability theory",
            "ode ",
            "pde ",
            "dynamical ",
            "celestial mechanics",
            "normal forms",
            "complex systems",
            "kam story",
            "kam theory",
            "hamiltonian",
            "n-body problem",
            "foundations of mechanics",
            "perturbation",
            "linear system theory",
            "feedback control",
            "switching in systems",
            "nonlinear system",
            "orbital motion",
            "chaos and fractals",
            "poincare",
            "lagrangian",
            "symplectic",
            "control system",
            "optimal control",
            "robust control",
            "adaptive control",
        ],
    ),
    # Mathematics (general — catch-all for math that doesn't fit elsewhere)
    (
        "mathematics",
        [
            "calculus",
            "real analysis",
            "complex analysis",
            "abstract algebra",
            "group theory",
            "ring theory",
            "field theory math",
            "number theory",
            "combinatorics",
            "discrete mathematics",
            "mathematical logic",
            "set theory",
            "proof",
            "theorem",
            "mathematical method",
            "mathematical physics",
            "mathematics for",
            "mathematical foundation",
            "mathematical modeling",
            "applied mathematics",
            "math for",
            "mathematical ",
            "engineering mathematics",
            "advanced mathematics",
            "agent-based and individual-based modeling",
            "morse theory",
        ],
    ),
    # Topology / Geometry
    (
        "topology_geometry",
        [
            "topology",
            "topological",
            "algebraic topology",
            "differential geometry",
            "riemannian geometry",
            "manifold",
            "homology",
            "cohomology",
            "geometric ",
            "geometry",
        ],
    ),
    # Linear algebra
    (
        "linear_algebra",
        [
            "linear algebra",
            "matrix theory",
            "matrix analysis",
            "eigenvalue",
            "vector space",
            "linear operator",
        ],
    ),
    # Algebra
    (
        "algebra",
        [
            "algebra",
            "algebraic",
            "group theory",
            "ring theory",
            "galois",
            "commutative algebra",
            "homological algebra",
            "representation theory",
            "lie algebra",
            "lie group",
        ],
    ),
    # Analysis
    (
        "analysis",
        [
            "functional analysis",
            "measure theory",
            "harmonic analysis",
            "fourier analysis",
            "operator theory",
            "banach space",
            "hilbert space",
            "spectral theory",
            "variational",
            "integral equation",
        ],
    ),
    # Numerical methods
    (
        "numerical_methods",
        [
            "numerical method",
            "numerical analysis",
            "finite element",
            "numerical linear algebra",
            "computational method",
            "monte carlo method",
            "numerical solution",
            "numerical recipe",
            "scientific computing",
        ],
    ),
    # Biology / Neuroscience
    (
        "biology_neuroscience",
        [
            "neuroscience",
            "neural computation",
            "brain ",
            "computational neuroscience",
            "bioinformatics",
            "genomics",
            "molecular biology",
            "systems biology",
            "biological",
            "cognitive science",
            "nervous system",
            "anatomy",
            "physiology",
            "evolutionary",
            "ecology",
        ],
    ),
    # Signal processing
    (
        "signal_processing",
        [
            "signal processing",
            "digital signal",
            "image processing",
            "computer vision",
            "fourier transform signal",
            "wavelets",
            "filter design",
        ],
    ),
    # Time series
    (
        "time_series",
        [
            "time series",
            "forecasting",
            "arima",
            "temporal",
            "seasonal",
            "exponential smoothing",
        ],
    ),
]

# Second-pass: broader single-keyword patterns applied only if no first-pass match.
# These are intentionally loose — they only fire when nothing else matches.
BROAD_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "physics",
        [
            "feynman",
            "landau",
            "griffiths",
            "electrodynamics",
            "turbulence",
            "transport phenomena",
            "field quantization",
            "astronomy",
            "astrophysic",
            "tensor",
            "vectors and tensors",
            "analytical mechanics",
        ],
    ),
    (
        "software_engineering",
        [
            "python",
            "java ",
            "c++ ",
            "c programming",
            "tiny c projects",
            "tiny_c_projects",
            "julia_as_a_second",
            "serverless",
            "asyncio",
            "concurrency",
            "microservice",
            "langchain",
            "functions_tools_agents",
        ],
    ),
    (
        "rag_llm",
        [
            "langchain",
            "functions_tools_agents",
        ],
    ),
    (
        "finance",
        [
            "frm exam",
            "frm ",
            "bank credit",
            "credit analysis",
            "risk management",
            "financial analyst",
        ],
    ),
    (
        "actuarial_insurance",
        [
            "soa guide",
            "soa exam",
            "society of actuaries",
        ],
    ),
    (
        "fitness",
        [
            "glute lab",
            "physique training",
            "fat loss",
            "rapid fat loss",
            "stretching",
            "physical therapy",
        ],
    ),
    (
        "analysis",
        [
            "singular integrals",
            "function spaces",
            "complex variable",
            "potential theory",
        ],
    ),
    (
        "data_science",
        [
            "grammar of graphics",
            "controlled experiments",
            "a/b testing",
            "network science",
            "network scientist",
        ],
    ),
    (
        "statistics",
        [
            "openintro statistics",
            "statistics slam",
        ],
    ),
    (
        "algorithms",
        [
            "optimization",
            "fourth paradigm",
            "problem solving",
            "puzzle solving",
        ],
    ),
    (
        "actuarial_insurance",
        [
            "soa ",
            "fsa exam",
            "fsa-",
            "ila ",
            "ila101",
            "earnings emergence",
            "intro-to-fsa",
            "intro to fsa",
        ],
    ),
    (
        "dynamical_systems",
        [
            "introduction to dynamics",
            "stable mappings",
            "singularities",
        ],
    ),
    (
        "algebra",
        [
            "cyclotomic",
        ],
    ),
    (
        "statistics",
        [
            "decision theory",
            "biostatmethod",
            "openintro",
        ],
    ),
    (
        "data_science",
        [
            "r in action",
            "r_in_action",
        ],
    ),
    (
        "software_engineering",
        [
            "cloud engineer",
            "cloud certified",
            "building user-friendly",
            "building_user-friendly",
            "domain-specific_language",
            "domain-specific language",
            "dsl",
            "schaum's outline of fundamentals of computing",
            "tiny_c_",
            "c projects",
        ],
    ),
    (
        "physics",
        [
            "neutrino",
        ],
    ),
]


def _score_patterns(
    combined: str,
    patterns: list[tuple[str, list[str]]],
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Score combined text against pattern list. Returns (scores, matched_keywords)."""
    scores: dict[str, int] = {}
    matched_keywords: dict[str, list[str]] = {}
    for domain, pats in patterns:
        hits = []
        for pat in pats:
            if pat in combined:
                hits.append(pat.strip())
        if hits:
            scores[domain] = scores.get(domain, 0) + len(hits)
            matched_keywords.setdefault(domain, []).extend(hits)
    return scores, matched_keywords


def classify_by_title(
    title: str, filename: str, ol_title: str | None = None
) -> tuple[str | None, str, str]:
    """
    Classify from title/filename/OL title.
    Returns (domain, confidence, reasoning).
    """
    # Combine all title signals
    combined = f"{title} {filename} {ol_title or ''}".lower()

    scores, matched_keywords = _score_patterns(combined, TITLE_PATTERNS)

    # If no primary match, try broad patterns
    if not scores:
        scores, matched_keywords = _score_patterns(combined, BROAD_PATTERNS)
        if not scores:
            return None, "unclassified", ""
        # Broad matches are always low confidence unless multiple hits
        best = max(scores, key=scores.__getitem__)
        best_count = scores[best]
        confidence = "medium" if best_count >= 2 else "low"
        keywords_str = ", ".join(matched_keywords[best][:5])
        reasoning = f"Broad title match [{keywords_str}] for {best}"
        return best, confidence, reasoning

    best = max(scores, key=scores.__getitem__)
    best_count = scores[best]

    # Confidence: multiple keyword hits = higher confidence
    if best_count >= 3:
        confidence = "high"
    elif best_count >= 2:
        confidence = "medium"
    else:
        # Single match — check for ambiguity
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0][1] == sorted_scores[1][1]:
            confidence = "low"
        else:
            confidence = "medium"

    keywords_str = ", ".join(matched_keywords[best][:5])
    reasoning = f"Title/filename matched [{keywords_str}] for {best}"
    return best, confidence, reasoning


def main():
    input_path = Path("fixtures/library_catalog/needs_classification.json")
    output_path = Path("fixtures/library_catalog/classified_results.json")

    entries = json.loads(input_path.read_text())
    print(f"Loaded {len(entries)} entries needing classification", file=sys.stderr)

    results: list[dict] = []
    classified = 0
    confirmed = 0
    still_unclassified = 0

    for entry in entries:
        sha = entry["sha256"]
        title = entry.get("title", "")
        filename = entry.get("filename", "")
        ol_data = entry.get("r2_openlibrary")
        ol_title = ol_data.get("title") if ol_data and ol_data.get("found") else None
        current_domain = entry.get("current_domain", "unclassified")

        domain, confidence, reasoning = classify_by_title(title, filename, ol_title)

        if domain:
            # If title classifier agrees with existing low-confidence domain, boost to medium
            if domain == current_domain and confidence == "low":
                confidence = "medium"
                reasoning += " (confirmed existing classification)"
            results.append(
                {
                    "sha256": sha,
                    "domain": domain,
                    "confidence": confidence,
                    "method": "claude_title_match",
                    "reasoning": reasoning,
                }
            )
            classified += 1
        elif current_domain != "unclassified":
            # Entry already has a domain from Stage 4 (low confidence) — confirm it
            results.append(
                {
                    "sha256": sha,
                    "domain": current_domain,
                    "confidence": "low",
                    "method": "claude_confirmed",
                    "reasoning": f"Confirmed existing Stage 4 classification: {current_domain}",
                }
            )
            confirmed += 1
        else:
            still_unclassified += 1

    output_path.write_text(json.dumps(results, indent=2))
    print(f"Classified: {classified}", file=sys.stderr)
    print(f"Confirmed existing: {confirmed}", file=sys.stderr)
    print(f"Still unclassified: {still_unclassified}", file=sys.stderr)

    # Summary
    domain_counts = Counter(r["domain"] for r in results)
    conf_counts = Counter(r["confidence"] for r in results)
    print(f"\nDomain distribution:", file=sys.stderr)
    for d, c in domain_counts.most_common():
        print(f"  {d}: {c}", file=sys.stderr)
    print(f"\nConfidence distribution:", file=sys.stderr)
    for c, n in conf_counts.most_common():
        print(f"  {c}: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
