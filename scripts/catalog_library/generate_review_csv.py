#!/usr/bin/env python3
"""Generate domain review CSV + symlinked review folder for offline review.

Reads R2 catalog JSONs + enrichment checkpoint, pre-populates domain suggestions
against the canonical 33-domain taxonomy, and flags entries needing human review.

Dedup + auto-accept layers:
  1. is_best_copy=False → skip (370 entries)
  2. KB overlap → already_ingested, auto-accept (23 entries)
  3. Keyword==R2 agreement with no ambiguity → auto-accept (~1,350 entries)
  4. LLM checkpoint (if available) → agent classification with confidence tiers

Usage:
    python scripts/catalog_library/generate_review_csv.py
    python scripts/catalog_library/generate_review_csv.py --no-symlinks  # CSV only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Canonical 33-domain taxonomy
# ---------------------------------------------------------------------------

CANONICAL_DOMAINS: dict[str, list[str]] = {
    # Mathematics — Pure
    "algebra": [
        "abstract algebra",
        "group theory",
        "ring theory",
        "field theory",
        "galois",
        "lie algebra",
        "homological algebra",
        "commutative algebra",
        "representation theory",
        "module theory",
        "category theory",
        "algebraic structure",
        "groups rings",
        "graded algebra",
    ],
    "analysis": [
        "real analysis",
        "complex analysis",
        "functional analysis",
        "measure theory",
        "harmonic analysis",
        "fourier analysis",
        "operator theory",
        "banach space",
        "hilbert space",
        "lebesgue",
        "sobolev",
        "distribution theory",
        "integral equation",
        "mathematical analysis",
    ],
    "topology_geometry": [
        "topology",
        "differential geometry",
        "algebraic topology",
        "manifold",
        "riemannian",
        "homotopy",
        "homology",
        "cohomology",
        "fiber bundle",
        "sheaf",
        "symplectic",
        "knot theory",
        "topological space",
        "metric space",
    ],
    "linear_algebra": [
        "linear algebra",
        "matrix theory",
        "eigenvalue",
        "eigenvector",
        "vector space",
        "singular value",
        "svd",
        "determinant",
        "matrix decomposition",
        "matrix analysis",
        "tensor",
        "inner product space",
    ],
    "probability_theory": [
        "probability theory",
        "stochastic process",
        "martingale",
        "markov chain",
        "brownian motion",
        "random variable",
        "measure-theoretic probability",
        "stochastic calculus",
        "poisson process",
        "ergodic",
        "large deviation",
        "concentration inequality",
        "random walk",
    ],
    "mathematics": [
        "combinatorics",
        "number theory",
        "mathematical logic",
        "foundations of mathematics",
        "set theory",
        "graph theory",
        "discrete mathematics",
        "enumerative",
        "cryptography",
        "lattice theory",
        "mathematical proof",
        "logic and computation",
    ],
    # Mathematics — Applied
    "dynamical_systems": [
        "dynamical system",
        "ordinary differential equation",
        "partial differential equation",
        "chaos theory",
        "bifurcation",
        "control theory",
        "control system",
        "hamiltonian mechanics",
        "hamiltonian system",
        "stability theory",
        "lyapunov",
        "phase space",
        "attractor",
        "ergodic theory",
        "classical mechanics",
        "lagrangian mechanics",
        "nonlinear dynamics",
        "nonlinear system",
        "ode system",
        "pde theory",
        "strogatz",
    ],
    "numerical_methods": [
        "numerical analysis",
        "finite element",
        "fem",
        "finite difference",
        "scientific computing",
        "computational mathematics",
        "numerical linear algebra",
        "interpolation",
        "quadrature",
        "numerical solution",
        "discretization",
        "galerkin",
        "spectral method",
        "multigrid",
    ],
    "optimization": [
        "convex optimization",
        "operations research",
        "variational",
        "linear programming",
        "integer programming",
        "combinatorial optimization",
        "gradient descent",
        "newton method",
        "constrained optimization",
        "nonlinear programming",
        "stochastic optimization",
        "mathematical programming",
        "simplex method",
        "duality",
        "lagrange multiplier",
        "proximal",
        "convex analysis",
        "semidefinite programming",
        "conic programming",
    ],
    # Statistics & Inference
    "statistics": [
        "statistical inference",
        "hypothesis testing",
        "regression",
        "bayesian statistics",
        "frequentist",
        "statistical learning",
        "experimental design",
        "survey sampling",
        "bootstrap",
        "nonparametric statistics",
        "estimation theory",
        "sufficient statistic",
        "maximum likelihood",
        "confidence interval",
    ],
    "causal_inference": [
        "causal inference",
        "causality",
        "counterfactual",
        "do-calculus",
        "treatment effect",
        "propensity score",
        "instrumental variable",
        "difference-in-differences",
        "regression discontinuity",
        "potential outcomes",
        "dag",
        "structural equation",
        "double machine learning",
        "synthetic control",
    ],
    "econometrics": [
        "econometrics",
        "panel data",
        "gmm",
        "instrumental variable",
        "heteroskedasticity",
        "endogeneity",
        "cointegration",
        "simultaneous equations",
        "microeconometrics",
        "macroeconometrics",
        "economic model",
        "economic theory",
        "welfare economics",
        "game theory",
        "general equilibrium",
        "macroeconomics",
        "microeconomics",
        "political economy",
        "development economics",
    ],
    "time_series": [
        "time series",
        "arima",
        "garch",
        "spectral analysis",
        "state space model",
        "kalman filter",
        "exponential smoothing",
        "seasonal",
        "autocorrelation",
        "forecasting",
        "prophet",
        "var model",
        "granger causality",
        "changepoint",
    ],
    # Computer Science
    "algorithms": [
        "algorithm design",
        "data structure",
        "computational complexity",
        "graph algorithm",
        "sorting",
        "dynamic programming",
        "np-hard",
        "approximation algorithm",
        "competitive programming",
        "algorithm analysis",
        "amortized analysis",
        "greedy algorithm",
    ],
    "software_engineering": [
        "software engineering",
        "design pattern",
        "refactoring",
        "clean code",
        "architecture",
        "microservice",
        "devops",
        "continuous integration",
        "testing",
        "agile",
        "scrum",
        "domain-driven design",
        "object-oriented",
        "solid principles",
    ],
    "functional_programming": [
        "functional programming",
        "haskell",
        "lambda calculus",
        "type theory",
        "monad",
        "functor",
        "category theory programming",
        "lisp",
        "scheme",
        "clojure",
        "ocaml",
        "f#",
        "scala",
        "purely functional",
        "immutable",
    ],
    "sql": [
        "sql",
        "relational database",
        "query optimization",
        "database design",
        "normalization",
        "postgresql",
        "mysql",
        "database administration",
        "transaction",
        "acid",
    ],
    # Machine Learning & AI
    "machine_learning": [
        "machine learning",
        "supervised learning",
        "unsupervised learning",
        "classification",
        "clustering",
        "random forest",
        "svm",
        "support vector",
        "decision tree",
        "ensemble method",
        "feature engineering",
        "cross-validation",
        "regularization",
        "statistical learning",
        "pattern recognition",
    ],
    "deep_learning": [
        "deep learning",
        "neural network",
        "cnn",
        "convolutional",
        "recurrent neural",
        "rnn",
        "lstm",
        "transformer",
        "attention mechanism",
        "backpropagation",
        "batch normalization",
        "generative adversarial",
        "autoencoder",
        "diffusion model",
        "computer vision",
        "image recognition",
    ],
    "reinforcement_learning": [
        "reinforcement learning",
        "q-learning",
        "policy gradient",
        "markov decision process",
        "mdp",
        "multi-armed bandit",
        "temporal difference",
        "actor-critic",
        "reward shaping",
        "monte carlo tree search",
    ],
    "rag_llm": [
        "large language model",
        "llm",
        "retrieval augmented generation",
        "rag",
        "prompt engineering",
        "fine-tuning",
        "chatgpt",
        "gpt",
        "bert",
        "embedding model",
        "vector database",
        "natural language processing",
        "nlp",
        "text generation",
        "language model",
        "instruction tuning",
    ],
    "ml_engineering": [
        "ml engineering",
        "mlops",
        "model deployment",
        "model serving",
        "feature store",
        "experiment tracking",
        "model monitoring",
        "data pipeline",
        "kubeflow",
        "mlflow",
        "model registry",
    ],
    "data_science": [
        "data science",
        "data analysis",
        "pandas",
        "data visualization",
        "exploratory data analysis",
        "data wrangling",
        "jupyter",
        "data mining",
        "business analytics",
        "dashboard",
    ],
    "recommender_systems": [
        "recommender system",
        "collaborative filtering",
        "content-based filtering",
        "matrix factorization",
        "recommendation engine",
        "implicit feedback",
    ],
    # Natural Sciences
    "physics": [
        "quantum mechanics",
        "quantum field theory",
        "electrodynamics",
        "thermodynamics",
        "statistical mechanics",
        "general relativity",
        "special relativity",
        "particle physics",
        "condensed matter",
        "solid state physics",
        "nuclear physics",
        "astrophysics",
        "cosmology",
        "fluid dynamics",
        "feynman diagram",
        "many-body problem",
        "many-body physics",
        "quantum physics",
        "classical physics",
        "theoretical physics",
        "physics textbook",
        "lectures on physics",
    ],
    "biology_neuroscience": [
        "biology",
        "neuroscience",
        "bioinformatics",
        "genomics",
        "computational biology",
        "neural circuit",
        "brain",
        "cognitive science",
        "evolution",
        "genetics",
        "molecular biology",
        "systems biology",
        "neuron",
        "connectome",
    ],
    "signal_processing": [
        "signal processing",
        "digital signal",
        "dsp",
        "filter design",
        "wavelet",
        "image processing",
        "audio processing",
        "compressed sensing",
        "sampling theory",
        "information theory",
        "coding theory",
        "communication theory",
        "modulation",
    ],
    # Finance & Business
    "finance": [
        "finance",
        "financial mathematics",
        "derivatives",
        "option pricing",
        "black-scholes",
        "risk management",
        "fixed income",
        "yield curve",
        "credit risk",
        "stochastic finance",
        "financial engineering",
        "quantitative finance",
        "valuation",
        "corporate finance",
        "financial accounting",
    ],
    "portfolio_management": [
        "portfolio management",
        "asset allocation",
        "modern portfolio theory",
        "capm",
        "factor model",
        "markowitz",
        "efficient frontier",
        "wealth management",
        "investment strategy",
        "portfolio optimization",
    ],
    "actuarial_insurance": [
        "actuarial",
        "insurance",
        "life table",
        "mortality",
        "loss model",
        "ruin theory",
        "annuity",
        "premium",
        "reserving",
        "solvency",
        "credibility theory",
    ],
    "adtech": [
        "advertising technology",
        "ad tech",
        "programmatic",
        "real-time bidding",
        "rtb",
        "click-through rate",
        "conversion optimization",
        "attribution model",
        "ad serving",
    ],
    # Professional
    "interview_prep": [
        "interview preparation",
        "coding interview",
        "system design interview",
        "behavioral interview",
        "leetcode",
        "technical interview",
        "career",
        "resume",
        "job search",
    ],
    "fitness": [
        "fitness",
        "exercise",
        "strength training",
        "nutrition",
        "bodybuilding",
        "weightlifting",
        "conditioning",
        "workout",
        "sports science",
        "kinesiology",
    ],
}

# Reverse: keyword → domain(s)
_KEYWORD_INDEX: dict[str, list[str]] = {}
for _domain, _keywords in CANONICAL_DOMAINS.items():
    for _kw in _keywords:
        _KEYWORD_INDEX.setdefault(_kw, []).append(_domain)

# ---------------------------------------------------------------------------
# Known corrections (author/title → domain)
# ---------------------------------------------------------------------------

KNOWN_CORRECTIONS: list[tuple[list[str], str]] = [
    # (patterns_to_match_in_title_or_filename, correct_domain)
    (["strang", "linear algebra"], "linear_algebra"),
    (["pearl", "causality"], "causal_inference"),
    (["dynamics", "geometry", "behavior"], "dynamical_systems"),
    (["introduction to statistical learning"], "machine_learning"),
    (["islr"], "machine_learning"),
    (["elements of statistical learning"], "machine_learning"),
    (["hastie", "statistical learning"], "machine_learning"),
    (["convex optimization"], "optimization"),
    (["boyd", "vandenberghe"], "optimization"),
    (["optimization algorithm"], "optimization"),
    (["numerical optimization"], "optimization"),
    (["quantum field theory"], "physics"),
    (["quantum mechanics"], "physics"),
    (["electrodynamics", "griffiths"], "physics"),
    (["goldstein", "classical mechanics"], "dynamical_systems"),
    (["arnold", "mathematical methods", "classical mechanics"], "dynamical_systems"),
    (["strogatz", "nonlinear dynamics"], "dynamical_systems"),
    (["control system", "control theory"], "dynamical_systems"),
    (["feynman", "lectures on physics"], "physics"),
    (["feynman diagram"], "physics"),
    (["many-body"], "physics"),
    (["rudin", "analysis"], "analysis"),
    (["munkres", "topology"], "topology_geometry"),
    (["hatcher", "algebraic topology"], "topology_geometry"),
    (["dummit", "foote", "abstract algebra"], "algebra"),
    (["artin", "algebra"], "algebra"),
    (["lang", "algebra"], "algebra"),
    (["hungerford", "algebra"], "algebra"),
    (["sheldon ross", "probability"], "probability_theory"),
    (["durrett", "probability"], "probability_theory"),
    (["numerical recipes"], "numerical_methods"),
    (["trefethen", "numerical"], "numerical_methods"),
    (["wooldridge", "econometrics"], "econometrics"),
    (["greene", "econometric"], "econometrics"),
    (["angrist", "mostly harmless"], "econometrics"),
    (["hamilton", "time series"], "time_series"),
    (["box", "jenkins", "time series"], "time_series"),
    (["hyndman", "forecasting"], "time_series"),
    (["goodfellow", "deep learning"], "deep_learning"),
    (["sutton", "barto", "reinforcement learning"], "reinforcement_learning"),
    (["bishop", "pattern recognition"], "machine_learning"),
    (["murphy", "machine learning"], "machine_learning"),
    (["nocedal", "wright", "numerical optimization"], "optimization"),
    (["bertsekas", "optimization"], "optimization"),
    (["bertsimas", "optimization"], "optimization"),
    (["boyd", "vandenberghe"], "optimization"),
    (["luenberger", "optimization"], "optimization"),
    (["shreve", "stochastic calculus", "finance"], "finance"),
    (["hull", "options", "futures"], "finance"),
    (["black", "scholes"], "finance"),
    (["martin fowler", "refactoring"], "software_engineering"),
    (["clean code", "robert martin"], "software_engineering"),
    (["design patterns", "gang of four"], "software_engineering"),
    (["knuth", "art of computer programming"], "algorithms"),
    (["cormen", "introduction to algorithms"], "algorithms"),
    (["sipser", "theory of computation"], "algorithms"),
    (["cracking the coding interview"], "interview_prep"),
    (["system design interview"], "interview_prep"),
]

# Domains to merge (old → new)
DOMAIN_MERGES: dict[str, str] = {
    "forecasting": "time_series",
    "economics": "econometrics",
}

# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for matching.

    Also splits CamelCase into separate words so that
    'GoldsteinClassicalMechanics' → 'goldstein classical mechanics'.
    """
    # Insert space before uppercase letters preceded by lowercase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Insert space between letter and digit boundaries
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def _check_known_corrections(title: str, filename: str, authors: list[str]) -> str | None:
    """Check against known author/title → domain corrections."""
    combined = _normalize(f"{title} {filename} {' '.join(authors)}")
    for patterns, domain in KNOWN_CORRECTIONS:
        if all(p in combined for p in patterns):
            return domain
    return None


# ---------------------------------------------------------------------------
# KB overlap detection
# ---------------------------------------------------------------------------


def _build_kb_lookup(inventory: dict[str, Any]) -> dict[str, str]:
    """Build {normalized_title_prefix: domain} from owned_inventory.json.

    Uses first 80 chars of normalized title for matching.
    """
    lookup: dict[str, str] = {}
    for source in inventory.get("sources", []):
        title = source.get("title", "")
        if title:
            key = _normalize(title)[:80]
            if key:
                lookup[key] = source.get("domain", "")
    return lookup


def _check_kb_overlap(title: str, filename: str, kb_lookup: dict[str, str]) -> str | None:
    """Check if entry matches a KB source. Returns domain if match found."""
    if not kb_lookup:
        return None
    norm_title = _normalize(title)[:80]
    if norm_title and norm_title in kb_lookup:
        return kb_lookup[norm_title]
    norm_fn = _normalize(filename)[:80]
    if norm_fn and norm_fn in kb_lookup:
        return kb_lookup[norm_fn]
    return None


# ---------------------------------------------------------------------------
# Domain scoring
# ---------------------------------------------------------------------------


def _score_domains(
    title: str,
    filename: str,
    text_sample: str,
    authors: list[str],
    current_domain: str,
) -> list[tuple[str, float]]:
    """Score each canonical domain. Returns sorted (domain, score) pairs.

    Scoring strategy:
    - Title/filename matches: 3.0 per keyword (high signal)
    - Text sample matches: 0.5 per occurrence, capped at 3 (noisy signal)
    - Multi-word keywords get 1.5x weight (more specific)
    - Current domain gets 1.2x boost (usually correct)
    """
    combined_meta = _normalize(f"{title} {filename} {' '.join(authors)}")
    combined_text = _normalize(text_sample[:2000]) if text_sample else ""

    scores: dict[str, float] = {}
    for domain, keywords in CANONICAL_DOMAINS.items():
        score = 0.0
        for kw in keywords:
            kw_lower = kw.lower()
            is_phrase = " " in kw_lower  # Multi-word = more specific
            weight = 1.5 if is_phrase else 1.0

            # Title/filename matches (high signal)
            if kw_lower in combined_meta:
                score += 3.0 * weight

            # Text sample matches (lower signal, capped)
            if combined_text and kw_lower in combined_text:
                count = min(combined_text.count(kw_lower), 3)
                score += 0.5 * count * weight

        if score > 0:
            scores[domain] = score

    # Boost current domain (it's often right, especially from R2)
    effective_current = DOMAIN_MERGES.get(current_domain, current_domain)
    if effective_current in scores:
        scores[effective_current] *= 1.2
    elif effective_current in CANONICAL_DOMAINS:
        # Give a baseline score to the current domain even without keyword hits
        scores[effective_current] = 2.0

    return sorted(scores.items(), key=lambda x: -x[1])


def suggest_domains(entry: dict[str, Any], text_sample: str) -> tuple[str, list[str], bool]:
    """Return (primary_domain, secondary_domains, needs_review).

    Returns:
        primary: best-guess domain from canonical taxonomy
        secondary: list of other plausible domains
        needs_review: whether this entry should be flagged for human review
    """
    title = entry.get("title", "") or ""
    filename = entry.get("filename", "") or ""
    authors = entry.get("authors", []) or []
    current = entry.get("domain", "unclassified")
    confidence = entry.get("confidence", "unclassified")

    # Apply domain merges to current
    effective_current = DOMAIN_MERGES.get(current, current)

    # 1. Check known corrections first (high confidence)
    correction = _check_known_corrections(title, filename, authors)
    if correction:
        # Still check for secondary domains
        scored = _score_domains(title, filename, text_sample, authors, current)
        secondaries = [d for d, s in scored if d != correction and s > 3.0][:3]
        needs_review = correction != effective_current
        return correction, secondaries, needs_review

    # 2. Score against taxonomy
    scored = _score_domains(title, filename, text_sample, authors, current)

    if not scored:
        # No matches at all — flag for review, keep current or "unclassified"
        primary = effective_current if effective_current in CANONICAL_DOMAINS else "unclassified"
        return primary, [], True

    primary = scored[0][0]
    top_score = scored[0][1]

    # Secondary: domains with >40% of top score and >3.0 absolute
    secondaries = [d for d, s in scored[1:4] if s > 3.0 and s > top_score * 0.4]

    # Auto-accept: keyword scorer agrees with R2 and no strong secondary
    if primary == effective_current and effective_current != "unclassified":
        if not secondaries:
            return primary, secondaries, False

    # Determine needs_review
    needs_review = False
    if primary != effective_current and effective_current != "unclassified":
        needs_review = True  # Domain changed
    if confidence in ("low", "unclassified"):
        needs_review = True  # Low confidence
    if len(scored) >= 2 and scored[1][1] > top_score * 0.8:
        needs_review = True  # Close competition between domains
    if current == "unclassified":
        needs_review = True  # Was unclassified

    return primary, secondaries, needs_review


# ---------------------------------------------------------------------------
# CSV generation
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "sha256",
    "filename",
    "title",
    "current_domain",
    "suggested_domain",
    "secondary_domains",
    "skip",
    "needs_review",
    "already_ingested",
    "duplicate_group",
    "confidence",
    "priority_score",
    "is_book",
    "llm_domain",
    "llm_confidence",
    "llm_reasoning",
    "text_sample",
    "ol_title",
    "file_size_mb",
]


def build_row(
    entry: dict[str, Any],
    checkpoint: dict[str, Any],
    kb_lookup: dict[str, str],
    llm_checkpoint: dict[str, Any],
) -> dict[str, str]:
    """Build a CSV row dict from a catalog entry + checkpoint data."""
    sha = entry["sha256"]
    cp = checkpoint.get(sha, {})
    text_sample = cp.get("text_sample", "")

    # Clean text_sample: collapse newlines/whitespace, truncate to 200 chars
    text_sample_display = ""
    if text_sample:
        text_sample_display = re.sub(r"\s+", " ", text_sample).strip()[:200]

    # OL title
    ol = entry.get("r2_openlibrary") or {}
    ol_title = ol.get("title", "")

    # R2 dedup info
    r2_dedup = entry.get("r2_dedup", {})
    dup_group = r2_dedup.get("content_fingerprint_group", "")

    # Shared fields for early-return rows
    base_fields = {
        "sha256": sha,
        "filename": entry.get("filename", ""),
        "title": entry.get("title", ""),
        "current_domain": entry.get("domain", "unclassified"),
        "priority_score": str(entry.get("priority_score", 0)),
        "is_book": "TRUE" if entry.get("is_book") else "FALSE",
        "text_sample": text_sample_display,
        "ol_title": ol_title,
        "file_size_mb": f"{entry.get('file_size_mb', 0):.1f}",
        "duplicate_group": str(dup_group) if dup_group else "",
    }

    # --- Layer 1: Auto-skip non-best copies ---
    if r2_dedup.get("is_best_copy") is False:
        return {
            **base_fields,
            "suggested_domain": entry.get("domain", "unclassified"),
            "secondary_domains": "",
            "skip": "TRUE",
            "needs_review": "FALSE",
            "already_ingested": "FALSE",
            "confidence": "dedup_skip",
            "llm_domain": "",
            "llm_confidence": "",
            "llm_reasoning": "",
        }

    # --- Layer 2: KB overlap check ---
    kb_domain = _check_kb_overlap(entry.get("title", ""), entry.get("filename", ""), kb_lookup)
    already_ingested = kb_domain is not None

    # --- Layer 3: Keyword-based suggestion (includes auto-accept) ---
    primary, secondaries, needs_review = suggest_domains(entry, cp.get("text_sample", ""))

    # If already ingested in KB, auto-accept
    if already_ingested:
        needs_review = False

    # --- Layer 4: LLM checkpoint integration ---
    llm = llm_checkpoint.get(sha, {})
    llm_domain = llm.get("domain", "")
    llm_confidence = llm.get("confidence", "")
    llm_reasoning = llm.get("reasoning", "")

    if llm_domain:
        llm_conf_val = float(llm_confidence) if llm_confidence else 0.0

        # Agent said "skip" → non-academic junk
        if llm_domain == "skip":
            return {
                **base_fields,
                "suggested_domain": "skip",
                "secondary_domains": "",
                "skip": "TRUE",
                "needs_review": "FALSE",
                "already_ingested": "TRUE" if already_ingested else "FALSE",
                "confidence": "llm_skip",
                "llm_domain": llm_domain,
                "llm_confidence": str(llm_confidence),
                "llm_reasoning": (llm_reasoning or "")[:100],
            }

        # High confidence (≥0.85) → override domain, auto-accept
        if llm_conf_val >= 0.85:
            primary = llm_domain
            needs_review = False
        # Medium confidence + agrees with keyword → auto-accept
        elif llm_conf_val >= 0.70 and llm_domain == primary:
            needs_review = False
        # Low-medium confidence → use as suggestion, keep flagged
        elif llm_conf_val >= 0.60:
            primary = llm_domain
            # needs_review stays as-is
        # < 0.60: ignore LLM classification

    return {
        **base_fields,
        "suggested_domain": primary,
        "secondary_domains": ",".join(secondaries),
        "skip": "FALSE",
        "needs_review": "TRUE" if needs_review else "FALSE",
        "already_ingested": "TRUE" if already_ingested else "FALSE",
        "confidence": entry.get("confidence", "unclassified"),
        "llm_domain": llm_domain,
        "llm_confidence": str(llm_confidence),
        "llm_reasoning": (llm_reasoning or "")[:100],
    }


def generate_csv(
    books: list[dict],
    other: list[dict],
    checkpoint: dict,
    kb_lookup: dict[str, str],
    llm_checkpoint: dict[str, Any],
    output_path: Path,
) -> list[dict[str, str]]:
    """Generate the review CSV. Returns all rows for symlink generation."""
    all_entries = books + other
    rows = [build_row(e, checkpoint, kb_lookup, llm_checkpoint) for e in all_entries]

    # Sort: needs_review DESC → suggested_domain ASC → priority_score DESC
    rows.sort(
        key=lambda r: (
            r["needs_review"] != "TRUE",  # TRUE first (False sorts before True)
            r["suggested_domain"],
            -float(r["priority_score"]),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


# ---------------------------------------------------------------------------
# Symlink generation
# ---------------------------------------------------------------------------


def generate_review_symlinks(
    rows: list[dict[str, str]],
    all_entries: list[dict],
    review_dir: Path,
) -> tuple[int, int]:
    """Create symlinks for needs_review entries. Returns (created, skipped)."""
    # Build sha → entry lookup for full_path
    sha_to_entry = {e["sha256"]: e for e in all_entries}

    review_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    seen_names: set[str] = set()

    for row in rows:
        if row["needs_review"] != "TRUE":
            continue

        sha = row["sha256"]
        entry = sha_to_entry.get(sha)
        if not entry:
            skipped += 1
            continue

        full_path = entry.get("full_path", "")
        if not full_path or not os.path.exists(full_path):
            skipped += 1
            continue

        # Build symlink name: [domain]_originalfilename
        domain = row["suggested_domain"]
        original_name = os.path.basename(full_path)
        link_name = f"[{domain}]_{original_name}"

        # Deduplicate
        if link_name in seen_names:
            # Add sha prefix for uniqueness
            link_name = f"[{domain}]_{sha[:8]}_{original_name}"
        seen_names.add(link_name)

        # Truncate to filesystem limit (255 bytes)
        if len(link_name.encode("utf-8")) > 250:
            ext = Path(original_name).suffix
            link_name = link_name[: 245 - len(ext)] + ext

        link_path = review_dir / link_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        try:
            os.symlink(full_path, link_path)
            created += 1
        except OSError as e:
            print(f"  Warning: Could not create symlink for {original_name}: {e}", file=sys.stderr)
            skipped += 1

    return created, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate domain review CSV + symlinks")
    parser.add_argument("--no-symlinks", action="store_true", help="Skip symlink generation")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]  # repo root
    catalog_dir = base / "fixtures" / "library_catalog"

    # Load inputs
    print("Loading catalog data...")
    books = json.loads((catalog_dir / "catalog_books_r2.json").read_text())
    other = json.loads((catalog_dir / "catalog_other_r2.json").read_text())
    checkpoint = json.loads((catalog_dir / ".enrichment_checkpoint.json").read_text())

    print(f"  Books: {len(books)}, Other: {len(other)}, Checkpoint: {len(checkpoint)}")

    # Load KB inventory for overlap detection
    inv_path = base / "docs" / "owned_inventory.json"
    kb_lookup: dict[str, str] = {}
    if inv_path.exists():
        inv = json.loads(inv_path.read_text())
        kb_lookup = _build_kb_lookup(inv)
        print(f"  KB sources for overlap: {len(kb_lookup)}")

    # Load LLM classification checkpoint (if exists)
    llm_path = catalog_dir / ".llm_classification_checkpoint.json"
    llm_checkpoint: dict[str, Any] = {}
    if llm_path.exists():
        llm_checkpoint = json.loads(llm_path.read_text())
        print(f"  LLM classifications: {len(llm_checkpoint)}")

    # Generate CSV
    csv_path = catalog_dir / "domain_review.csv"
    print(f"\nGenerating review CSV → {csv_path}")
    rows = generate_csv(books, other, checkpoint, kb_lookup, llm_checkpoint, csv_path)

    # Stats
    needs_review = sum(1 for r in rows if r["needs_review"] == "TRUE")
    skipped = sum(1 for r in rows if r["skip"] == "TRUE")
    already_ingested = sum(1 for r in rows if r["already_ingested"] == "TRUE")
    llm_classified = sum(1 for r in rows if r["llm_domain"])
    domain_changed = sum(
        1
        for r in rows
        if r["suggested_domain"] != DOMAIN_MERGES.get(r["current_domain"], r["current_domain"])
        and r["current_domain"] != "unclassified"
    )
    unclassified = sum(1 for r in rows if r["current_domain"] == "unclassified")

    print(f"\n  Total rows: {len(rows)}")
    print(f"  Needs review: {needs_review}")
    print(f"  Auto-skipped (dedup): {skipped}")
    print(f"  Already ingested (KB): {already_ingested}")
    print(f"  LLM classified: {llm_classified}")
    print(f"  Domain changed (non-unclassified): {domain_changed}")
    print(f"  Currently unclassified: {unclassified}")

    # Domain distribution
    suggested = Counter(r["suggested_domain"] for r in rows)
    print("\n  Suggested domain distribution:")
    for domain, count in sorted(suggested.items(), key=lambda x: -x[1]):
        print(f"    {domain}: {count}")

    # Symlinks
    if not args.no_symlinks:
        review_dir = catalog_dir / "review"
        print(f"\nGenerating review symlinks → {review_dir}/")
        created, sym_skipped = generate_review_symlinks(rows, books + other, review_dir)
        print(f"  Created: {created}, Skipped (file not found): {sym_skipped}")
    else:
        print("\nSkipping symlink generation (--no-symlinks)")

    # Spot checks
    print("\n--- Spot Checks ---")
    for row in rows:
        if "strang" in row["filename"].lower() and "linear" in row["filename"].lower():
            print(
                f"  Strang: current={row['current_domain']} → suggested={row['suggested_domain']}"
            )
            break
    for row in rows:
        if "goldstein" in row["filename"].lower() and "mechanics" in row["filename"].lower():
            print(
                f"  Goldstein: current={row['current_domain']} → suggested={row['suggested_domain']}"
            )
            break
    for row in rows:
        if "convex" in row["title"].lower() and "optimization" in row["title"].lower():
            print(
                f"  Convex Opt: current={row['current_domain']} → suggested={row['suggested_domain']}"
            )
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
