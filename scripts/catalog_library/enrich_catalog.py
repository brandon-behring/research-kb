#!/usr/bin/env python3
"""
Round 2 Catalog Enrichment Pipeline.

Extracts text from PDFs, looks up ISBNs via OpenLibrary, classifies domains
using expanded keyword matching, rescores priorities, and deduplicates by
content fingerprint. All new fields are r2_* namespaced — Round 1 data is
never overwritten.

Usage:
    # Full automated pipeline (Stages 1-5)
    python scripts/catalog_library/enrich_catalog.py \
      --input fixtures/library_catalog/catalog_books.json \
      --input-other fixtures/library_catalog/catalog_other.json \
      --output-dir fixtures/library_catalog/ \
      --workers 8 --quiet

    # Merge Claude Code classifications (Phase C)
    python scripts/catalog_library/enrich_catalog.py \
      --merge fixtures/library_catalog/classified_results.json \
      --output-dir fixtures/library_catalog/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("enrich_catalog")


def _setup_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Expanded keyword map for text-based classification (Stage 4, Tier A)
# Round 1 keywords matched filenames only; these match extracted body text.
# ---------------------------------------------------------------------------

TEXT_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "algebra": [
        "group theory",
        "ring theory",
        "field extension",
        "galois theory",
        "homomorphism",
        "isomorphism",
        "module theory",
        "commutative algebra",
        "representation theory",
        "lie algebra",
        "lie group",
        "ideal theory",
        "noetherian ring",
        "polynomial ring",
        "quotient group",
        "sylow theorem",
        "abstract algebra",
        "algebraic structure",
        "homological algebra",
    ],
    "analysis": [
        "real analysis",
        "complex analysis",
        "functional analysis",
        "harmonic analysis",
        "measure theory",
        "lebesgue integral",
        "banach space",
        "hilbert space",
        "sobolev space",
        "distribution theory",
        "fourier series",
        "fourier transform",
        "spectral theory",
        "operator theory",
        "compact operator",
        "bounded linear",
        "dominated convergence",
        "monotone convergence",
        "lp space",
    ],
    "topology_geometry": [
        "topological space",
        "manifold",
        "riemannian geometry",
        "differential geometry",
        "algebraic topology",
        "cohomology",
        "homotopy",
        "homology group",
        "fundamental group",
        "fiber bundle",
        "sheaf theory",
        "symplectic geometry",
        "knot theory",
        "covering space",
        "simplicial complex",
        "euler characteristic",
        "differential form",
        "tangent bundle",
        "curvature tensor",
    ],
    "dynamical_systems": [
        "dynamical system",
        "bifurcation",
        "chaos theory",
        "hamiltonian system",
        "perturbation theory",
        "stability theory",
        "nonlinear dynamics",
        "celestial mechanics",
        "n-body problem",
        "ergodic theory",
        "lyapunov exponent",
        "poincare map",
        "phase space",
        "attractor",
        "lagrangian mechanics",
        "classical mechanics",
        "euler-lagrange",
        "canonical transformation",
        "action-angle",
    ],
    "probability_theory": [
        "probability measure",
        "stochastic process",
        "stochastic calculus",
        "markov chain",
        "martingale",
        "brownian motion",
        "ito integral",
        "stochastic differential equation",
        "random variable",
        "poisson process",
        "central limit theorem",
        "law of large numbers",
        "conditional expectation",
        "measure-theoretic probability",
        "filtration",
    ],
    "numerical_methods": [
        "numerical analysis",
        "finite element method",
        "finite difference",
        "numerical integration",
        "quadrature",
        "interpolation",
        "numerical linear algebra",
        "iterative method",
        "convergence rate",
        "runge-kutta",
        "newton method",
        "conjugate gradient",
        "convex optimization",
        "approximation theory",
    ],
    "linear_algebra": [
        "linear algebra",
        "matrix analysis",
        "eigenvalue",
        "eigenvector",
        "singular value decomposition",
        "vector space",
        "linear transformation",
        "determinant",
        "matrix factorization",
        "jordan form",
        "positive definite",
        "orthogonal matrix",
        "unitary matrix",
        "tensor product",
        "inner product space",
    ],
    "causal_inference": [
        "causal inference",
        "causal effect",
        "counterfactual",
        "instrumental variable",
        "propensity score",
        "treatment effect",
        "difference-in-differences",
        "regression discontinuity",
        "double machine learning",
        "potential outcome",
        "rubin causal model",
        "average treatment effect",
        "selection bias",
        "confounding",
        "directed acyclic graph",
        "structural causal model",
    ],
    "deep_learning": [
        "deep learning",
        "neural network",
        "transformer architecture",
        "attention mechanism",
        "convolutional neural network",
        "recurrent neural network",
        "batch normalization",
        "dropout",
        "backpropagation",
        "gradient descent",
        "loss function",
        "generative adversarial",
        "autoencoder",
        "diffusion model",
        "residual network",
        "transfer learning",
        "fine-tuning",
        "embedding layer",
        "activation function",
    ],
    "machine_learning": [
        "machine learning",
        "supervised learning",
        "unsupervised learning",
        "classification",
        "regression",
        "clustering",
        "random forest",
        "support vector machine",
        "decision tree",
        "ensemble method",
        "boosting",
        "bagging",
        "kernel method",
        "feature selection",
        "cross-validation",
        "bias-variance tradeoff",
        "overfitting",
        "regularization",
        "probabilistic graphical model",
    ],
    "reinforcement_learning": [
        "reinforcement learning",
        "reward function",
        "policy gradient",
        "q-learning",
        "temporal difference",
        "markov decision process",
        "bellman equation",
        "actor-critic",
        "monte carlo tree search",
        "exploration exploitation",
        "value function",
        "deep q-network",
        "multi-armed bandit",
    ],
    "rag_llm": [
        "natural language processing",
        "language model",
        "information retrieval",
        "text classification",
        "named entity recognition",
        "sentiment analysis",
        "word embedding",
        "word2vec",
        "bert",
        "gpt",
        "tokenization",
        "text generation",
        "prompt engineering",
        "retrieval augmented generation",
        "large language model",
        "question answering",
        "text mining",
        "semantic search",
        "vector database",
    ],
    "statistics": [
        "statistical inference",
        "hypothesis testing",
        "confidence interval",
        "maximum likelihood",
        "bayesian inference",
        "bayesian statistics",
        "generalized linear model",
        "survival analysis",
        "nonparametric",
        "experimental design",
        "analysis of variance",
        "regression analysis",
        "multivariate analysis",
        "sufficient statistic",
        "exponential family",
        "fisher information",
        "likelihood ratio",
    ],
    "econometrics": [
        "econometrics",
        "panel data",
        "generalized method of moments",
        "cointegration",
        "heteroskedasticity",
        "endogeneity",
        "two-stage least squares",
        "instrumental variable",
        "fixed effects",
        "random effects",
        "hausman test",
        "granger causality",
        "vector autoregression",
    ],
    "finance": [
        "derivatives pricing",
        "option pricing",
        "black-scholes",
        "quantitative finance",
        "algorithmic trading",
        "credit risk",
        "value at risk",
        "fixed income",
        "yield curve",
        "interest rate model",
        "stochastic volatility",
        "risk management",
        "financial engineering",
        "monte carlo simulation",
        "financial mathematics",
    ],
    "portfolio_management": [
        "portfolio optimization",
        "asset allocation",
        "modern portfolio theory",
        "mean-variance optimization",
        "efficient frontier",
        "sharpe ratio",
        "capital asset pricing model",
        "factor model",
        "risk parity",
        "portfolio construction",
        "rebalancing",
    ],
    "actuarial_insurance": [
        "actuarial science",
        "insurance mathematics",
        "loss model",
        "ruin theory",
        "life contingencies",
        "mortality table",
        "premium calculation",
        "reserving",
        "credibility theory",
        "risk theory",
        "casualty actuarial",
    ],
    "physics": [
        "quantum mechanics",
        "classical mechanics",
        "electrodynamics",
        "thermodynamics",
        "statistical mechanics",
        "quantum field theory",
        "general relativity",
        "special relativity",
        "condensed matter",
        "particle physics",
        "schrodinger equation",
        "lagrangian",
        "hamiltonian",
        "maxwell equations",
        "feynman diagram",
        "path integral",
        "wave function",
        "angular momentum",
    ],
    "biology_neuroscience": [
        "molecular biology",
        "genetics",
        "genomics",
        "bioinformatics",
        "computational neuroscience",
        "neural circuit",
        "neuroscience",
        "protein structure",
        "dna sequence",
        "gene expression",
        "evolution",
        "ecology",
        "population genetics",
        "systems biology",
    ],
    "signal_processing": [
        "signal processing",
        "digital signal processing",
        "control theory",
        "control system",
        "feedback control",
        "transfer function",
        "filter design",
        "fourier transform",
        "wavelet transform",
        "system identification",
        "adaptive filter",
        "z-transform",
        "frequency response",
        "bode plot",
    ],
    "time_series": [
        "time series analysis",
        "forecasting",
        "arima",
        "garch",
        "kalman filter",
        "state space model",
        "spectral analysis",
        "autoregressive",
        "moving average",
        "seasonal decomposition",
        "exponential smoothing",
        "unit root",
        "stationarity",
    ],
    "algorithms": [
        "algorithm design",
        "data structure",
        "computational complexity",
        "graph algorithm",
        "dynamic programming",
        "greedy algorithm",
        "sorting algorithm",
        "binary search",
        "hash table",
        "balanced tree",
        "shortest path",
        "minimum spanning tree",
        "np-complete",
        "big-o notation",
        "amortized analysis",
    ],
    "software_engineering": [
        "software engineering",
        "design pattern",
        "software architecture",
        "refactoring",
        "continuous integration",
        "microservice",
        "distributed system",
        "version control",
        "test-driven development",
        "clean code",
        "code review",
        "agile methodology",
        "kubernetes",
        "docker",
        "cloud native",
        "devops",
        "site reliability engineering",
        "infrastructure as code",
        "object-oriented",
        "api design",
    ],
    "data_science": [
        "data science",
        "data mining",
        "data analysis",
        "data visualization",
        "exploratory data analysis",
        "feature engineering",
        "data pipeline",
        "pandas",
        "numpy",
        "matplotlib",
        "jupyter notebook",
        "scientific computing",
        "data wrangling",
    ],
    "functional_programming": [
        "functional programming",
        "haskell",
        "lambda calculus",
        "category theory",
        "monad",
        "type theory",
        "scala",
        "purely functional",
        "immutable data",
        "pattern matching",
        "higher-order function",
        "algebraic data type",
        "functor",
    ],
    "mathematics": [
        "number theory",
        "combinatorics",
        "discrete mathematics",
        "graph theory",
        "set theory",
        "mathematical logic",
        "differential equation",
        "calculus",
        "integral",
        "mathematical proof",
        "theorem",
        "lemma",
        "corollary",
    ],
    "interview_prep": [
        "coding interview",
        "system design interview",
        "behavioral interview",
        "leetcode",
        "programming interview",
        "cracking the coding",
        "technical interview",
        "algorithm interview",
    ],
    "sql": [
        "sql query",
        "relational database",
        "database design",
        "query optimization",
        "join operation",
        "stored procedure",
        "database management system",
        "normalization",
    ],
    "recommender_systems": [
        "recommender system",
        "collaborative filtering",
        "content-based filtering",
        "matrix factorization",
        "recommendation engine",
        "user-item interaction",
    ],
    "adtech": [
        "computational advertising",
        "real-time bidding",
        "ad serving",
        "click-through rate",
        "programmatic advertising",
        "advertising technology",
    ],
    "fitness": [
        "strength training",
        "resistance training",
        "exercise physiology",
        "bodybuilding",
        "athletic performance",
        "sport science",
        "weightlifting",
        "mobility",
        "muscle hypertrophy",
    ],
    "ml_engineering": [
        "mlops",
        "model deployment",
        "model serving",
        "feature store",
        "ml pipeline",
        "machine learning engineering",
        "model monitoring",
        "experiment tracking",
        "model registry",
    ],
    "forecasting": [
        "demand forecasting",
        "predictive analytics",
        "prophet",
        "exponential smoothing",
        "forecast accuracy",
        "time series forecasting",
    ],
}

# ---------------------------------------------------------------------------
# OpenLibrary subject → domain mapping (Stage 4, Tier B)
# ---------------------------------------------------------------------------

OL_SUBJECT_TO_DOMAIN: dict[str, str] = {
    # Mathematics
    "mathematics": "mathematics",
    "algebra": "algebra",
    "linear algebra": "linear_algebra",
    "abstract algebra": "algebra",
    "group theory": "algebra",
    "ring theory": "algebra",
    "topology": "topology_geometry",
    "geometry": "topology_geometry",
    "differential geometry": "topology_geometry",
    "algebraic topology": "topology_geometry",
    "real analysis": "analysis",
    "complex analysis": "analysis",
    "functional analysis": "analysis",
    "measure theory": "analysis",
    "harmonic analysis": "analysis",
    "number theory": "mathematics",
    "combinatorics": "mathematics",
    "discrete mathematics": "mathematics",
    "graph theory": "mathematics",
    "probability": "probability_theory",
    "probability theory": "probability_theory",
    "stochastic processes": "probability_theory",
    "numerical analysis": "numerical_methods",
    "differential equations": "dynamical_systems",
    "dynamical systems": "dynamical_systems",
    "mathematical optimization": "numerical_methods",
    "calculus": "mathematics",
    "mathematical logic": "mathematics",
    "set theory": "mathematics",
    "category theory": "functional_programming",
    # Statistics
    "statistics": "statistics",
    "mathematical statistics": "statistics",
    "bayesian statistics": "statistics",
    "multivariate analysis": "statistics",
    "regression analysis": "statistics",
    "experimental design": "statistics",
    "time series": "time_series",
    "time series analysis": "time_series",
    "forecasting": "forecasting",
    # CS / ML
    "computer science": "algorithms",
    "algorithms": "algorithms",
    "data structures": "algorithms",
    "computational complexity": "algorithms",
    "machine learning": "machine_learning",
    "artificial intelligence": "machine_learning",
    "deep learning": "deep_learning",
    "neural networks": "deep_learning",
    "natural language processing": "rag_llm",
    "information retrieval": "rag_llm",
    "computer programming": "software_engineering",
    "software engineering": "software_engineering",
    "computer architecture": "software_engineering",
    "operating systems": "software_engineering",
    "databases": "sql",
    "data mining": "data_science",
    "data science": "data_science",
    "reinforcement learning": "reinforcement_learning",
    "recommender systems": "recommender_systems",
    # Physics
    "physics": "physics",
    "quantum mechanics": "physics",
    "quantum theory": "physics",
    "classical mechanics": "dynamical_systems",
    "electrodynamics": "physics",
    "thermodynamics": "physics",
    "statistical mechanics": "physics",
    "quantum field theory": "physics",
    "relativity": "physics",
    "condensed matter": "physics",
    "particle physics": "physics",
    "optics": "physics",
    # Economics / Finance
    "econometrics": "econometrics",
    "economics": "econometrics",
    "finance": "finance",
    "financial mathematics": "finance",
    "portfolio management": "portfolio_management",
    "investment analysis": "portfolio_management",
    "actuarial science": "actuarial_insurance",
    "insurance": "actuarial_insurance",
    # Biology
    "biology": "biology_neuroscience",
    "neuroscience": "biology_neuroscience",
    "genetics": "biology_neuroscience",
    "bioinformatics": "biology_neuroscience",
    "evolution": "biology_neuroscience",
    # Engineering
    "signal processing": "signal_processing",
    "control theory": "signal_processing",
    "electrical engineering": "signal_processing",
    # Other
    "functional programming": "functional_programming",
    "haskell": "functional_programming",
    "causal inference": "causal_inference",
    "exercise": "fitness",
    "physical fitness": "fitness",
    "strength training": "fitness",
}

# ---------------------------------------------------------------------------
# LC classification prefix → domain (Stage 4, Tier C)
# ---------------------------------------------------------------------------

LC_PREFIX_TO_DOMAIN: list[tuple[str, str, str]] = [
    # (prefix_start, prefix_end_or_None, domain)
    ("QA1", "QA43", "mathematics"),
    ("QA75", "QA76.95", "software_engineering"),
    ("QA76", "QA76.9", "software_engineering"),
    ("QA150", "QA272", "algebra"),
    ("QA273", "QA280", "statistics"),
    ("QA299", "QA433", "analysis"),
    ("QA440", "QA699", "topology_geometry"),
    ("QA801", "QA939", "dynamical_systems"),
    ("QA297", "QA298", "numerical_methods"),
    ("QA164", "QA167", "mathematics"),  # combinatorics
    ("QA241", "QA247", "mathematics"),  # number theory
    ("QC", None, "physics"),
    ("QD", None, "physics"),  # chemistry (close enough)
    ("QH", None, "biology_neuroscience"),
    ("QR", None, "biology_neuroscience"),
    ("QP", None, "biology_neuroscience"),
    ("HB", None, "econometrics"),
    ("HC", None, "econometrics"),
    ("HG", None, "finance"),
    ("HG4501", "HG6051", "portfolio_management"),
    ("TK5102", "TK5106", "signal_processing"),
    ("T57", None, "statistics"),
]

# ---------------------------------------------------------------------------
# ISBN extraction patterns (Stage 2)
# ---------------------------------------------------------------------------

_RE_ISBN13 = re.compile(r"ISBN[\s:\-]*(97[89][\d\-]{10,17})", re.I)
_RE_ISBN10 = re.compile(r"ISBN[\s:\-]*(\d{9}[\dXx])", re.I)
_RE_BARE_ISBN13 = re.compile(r"\b(97[89]\d{10})\b")


def extract_isbns(text: str) -> list[str]:
    """Extract and normalize ISBNs from text."""
    raw: list[str] = []
    raw.extend(_RE_ISBN13.findall(text))
    raw.extend(_RE_ISBN10.findall(text))
    raw.extend(_RE_BARE_ISBN13.findall(text))

    seen: set[str] = set()
    result: list[str] = []
    for isbn in raw:
        clean = isbn.replace("-", "").replace(" ", "").strip()
        if len(clean) in (10, 13) and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


# ---------------------------------------------------------------------------
# Stage 1: Text extraction (runs in worker processes)
# ---------------------------------------------------------------------------


def _extract_text_from_pdf(filepath: str, max_pages: int = 5) -> dict[str, Any]:
    """Extract text from first N pages of a PDF. Runs in subprocess."""
    import logging as _logging
    import warnings

    # Suppress pypdf's noisy "Ignoring wrong pointing object" warnings
    _logging.getLogger("pypdf").setLevel(_logging.ERROR)
    _logging.getLogger("pypdf._reader").setLevel(_logging.ERROR)

    result: dict[str, Any] = {
        "chars_extracted": 0,
        "extractable": True,
        "has_toc": False,
        "content_fingerprint": None,
        "text_pages": [],
        "error": None,
    }

    try:
        import pypdf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = pypdf.PdfReader(filepath)
            pages_text: list[str] = []
            for i, page in enumerate(reader.pages[:max_pages]):
                try:
                    text = page.extract_text() or ""
                    pages_text.append(text)
                except Exception:
                    pages_text.append("")

            full_text = "\n".join(pages_text)
            result["chars_extracted"] = len(full_text)
            result["text_pages"] = pages_text

            # Extractable?
            if len(full_text.strip()) < 100:
                result["extractable"] = False

            # TOC detection
            first_3k = full_text[:3000].lower()
            if "table of contents" in first_3k or "contents\n" in first_3k:
                result["has_toc"] = True

            # Content fingerprint (pages 1-3 normalized)
            fingerprint_text = "\n".join(pages_text[:3])
            normalized = re.sub(r"\s+", " ", fingerprint_text.lower().strip())
            if len(normalized) > 50:
                result["content_fingerprint"] = hashlib.sha256(
                    normalized.encode("utf-8", errors="replace")
                ).hexdigest()[:16]

    except Exception as e:
        result["error"] = str(e)[:200]
        result["extractable"] = False

    return result


# ---------------------------------------------------------------------------
# Stage 3: OpenLibrary lookup
# ---------------------------------------------------------------------------


def _load_ol_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def _save_ol_cache(cache: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))


def _lookup_isbn_openlibrary(isbn: str, cache: dict, cache_path: Path) -> dict | None:
    """Query OpenLibrary for ISBN. Returns parsed data or None."""
    if isbn in cache:
        return cache[isbn]

    url = f"https://openlibrary.org/isbn/{isbn}.json"
    retries = 3
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-kb-catalog/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            result = {
                "isbn_queried": isbn,
                "found": True,
                "title": data.get("title"),
                "authors": _resolve_ol_authors(data.get("authors", [])),
                "publisher": (data.get("publishers") or [None])[0],
                "publish_date": data.get("publish_date"),
                "subjects": [
                    s if isinstance(s, str) else s.get("name", str(s))
                    for s in data.get("subjects", [])
                ],
                "lc_classifications": data.get("lc_classifications", []),
                "pages": data.get("number_of_pages"),
            }
            cache[isbn] = result
            return result

        except urllib.error.HTTPError as e:
            if e.code == 404:
                cache[isbn] = {"isbn_queried": isbn, "found": False}
                return cache[isbn]
            if e.code == 429:
                wait = 2**attempt
                log.warning("OpenLibrary rate limited, waiting %ds", wait)
                time.sleep(wait)
                continue
            cache[isbn] = {"isbn_queried": isbn, "found": False, "error": str(e)}
            return cache[isbn]

        except Exception as e:
            if attempt == retries - 1:
                cache[isbn] = {"isbn_queried": isbn, "found": False, "error": str(e)[:200]}
                return cache[isbn]
            time.sleep(2**attempt)

    return None


def _resolve_ol_authors(authors_raw: list) -> list[str]:
    """Resolve OpenLibrary author references to names (best-effort)."""
    names: list[str] = []
    for a in authors_raw:
        if isinstance(a, str):
            names.append(a)
        elif isinstance(a, dict):
            if "name" in a:
                names.append(a["name"])
            elif "key" in a:
                # Could fetch /authors/OL...A.json, but skip for speed
                names.append(a["key"].split("/")[-1])
    return names


# ---------------------------------------------------------------------------
# Stage 4: Rule-based classification
# ---------------------------------------------------------------------------


def _classify_by_text_keywords(text: str) -> tuple[str | None, str, int]:
    """Classify domain from extracted text body. Returns (domain, confidence, match_count)."""
    if not text or len(text) < 50:
        return None, "unclassified", 0

    text_lower = text.lower()
    scores: dict[str, int] = {}

    for domain, keywords in TEXT_DOMAIN_KEYWORDS.items():
        count = 0
        for kw in keywords:
            # Count occurrences (capped at 10 per keyword to avoid bias)
            occurrences = text_lower.count(kw)
            count += min(occurrences, 10)
        if count > 0:
            scores[domain] = count

    if not scores:
        return None, "unclassified", 0

    best = max(scores, key=scores.__getitem__)
    best_count = scores[best]

    # Require meaningful signal
    if best_count < 3:
        return None, "unclassified", best_count

    # Confidence based on separation from runner-up
    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_domains) >= 2:
        ratio = sorted_domains[0][1] / max(sorted_domains[1][1], 1)
        if ratio >= 3.0 and best_count >= 10:
            confidence = "high"
        elif ratio >= 1.5 and best_count >= 5:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = "high" if best_count >= 10 else "medium" if best_count >= 5 else "low"

    return best, confidence, best_count


def _classify_by_ol_subjects(subjects: list[str]) -> tuple[str | None, str]:
    """Classify domain from OpenLibrary subjects. Returns (domain, confidence)."""
    if not subjects:
        return None, "unclassified"

    domain_votes: Counter[str] = Counter()
    for subj in subjects:
        subj_lower = subj.lower().strip()
        # Exact match
        if subj_lower in OL_SUBJECT_TO_DOMAIN:
            domain_votes[OL_SUBJECT_TO_DOMAIN[subj_lower]] += 2
        # Substring match
        for key, domain in OL_SUBJECT_TO_DOMAIN.items():
            if key in subj_lower or subj_lower in key:
                domain_votes[domain] += 1

    if not domain_votes:
        return None, "unclassified"

    best, count = domain_votes.most_common(1)[0]
    confidence = "high" if count >= 4 else "medium" if count >= 2 else "low"
    return best, confidence


def _classify_by_lc(lc_classifications: list[str]) -> tuple[str | None, str]:
    """Classify domain from LC classification codes. Returns (domain, confidence)."""
    if not lc_classifications:
        return None, "unclassified"

    for lc in lc_classifications:
        lc_clean = lc.strip().replace(" ", "")
        for prefix_start, prefix_end, domain in LC_PREFIX_TO_DOMAIN:
            if prefix_end is None:
                if lc_clean.startswith(prefix_start):
                    return domain, "medium"
            else:
                if lc_clean >= prefix_start and lc_clean <= prefix_end + "z":
                    return domain, "medium"

    return None, "unclassified"


def classify_entry(
    entry: dict,
    checkpoint_data: dict | None = None,
) -> dict | None:
    """Run all classification tiers on an entry. Returns r2_classification or None."""

    # Get text from checkpoint
    text = ""
    if checkpoint_data:
        text = checkpoint_data.get("text_sample", "")

    # Tier A: Keywords on extracted text
    domain, confidence, match_count = _classify_by_text_keywords(text)
    if domain and confidence in ("high", "medium"):
        return {
            "domain": domain,
            "confidence": confidence,
            "method": "keyword_text",
            "reasoning": f"Matched {match_count} keywords for {domain} in text",
        }

    # Tier B: OpenLibrary subjects
    ol_data = entry.get("r2_openlibrary")
    if ol_data and ol_data.get("found"):
        ol_domain, ol_conf = _classify_by_ol_subjects(ol_data.get("subjects", []))
        if ol_domain and ol_conf in ("high", "medium"):
            return {
                "domain": ol_domain,
                "confidence": ol_conf,
                "method": "ol_subject",
                "reasoning": f"OpenLibrary subjects mapped to {ol_domain}",
            }

    # Tier C: LC classification
    if ol_data and ol_data.get("found"):
        lc_domain, lc_conf = _classify_by_lc(ol_data.get("lc_classifications", []))
        if lc_domain:
            return {
                "domain": lc_domain,
                "confidence": lc_conf,
                "method": "lc_class",
                "reasoning": f"LC classification mapped to {lc_domain}",
            }

    # Tier A fallback: low confidence text match
    if domain:
        return {
            "domain": domain,
            "confidence": "low",
            "method": "keyword_text",
            "reasoning": f"Weak text match ({match_count} keywords) for {domain}",
        }

    return None


# ---------------------------------------------------------------------------
# Stage 5: Rescore + deduplicate
# ---------------------------------------------------------------------------

R2_SCORE_ADJUSTMENTS = {
    "ol_confirmed": 15,
    "isbn_found": 10,
    "toc_detected": 10,
    "high_confidence": 5,
    "unique_fingerprint": 5,
    "hash_duplicate_not_preferred": -20,
    "scanned_image_only": -10,
}


def rescore_entry(entry: dict) -> tuple[int, list[str]]:
    """Calculate R2 priority score adjustments. Returns (delta, reasons)."""
    delta = 0
    reasons: list[str] = []

    ol = entry.get("r2_openlibrary")
    if ol and ol.get("found"):
        delta += R2_SCORE_ADJUSTMENTS["ol_confirmed"]
        reasons.append("r2:ol_confirmed")

    isbns = entry.get("r2_isbns", [])
    if isbns:
        delta += R2_SCORE_ADJUSTMENTS["isbn_found"]
        reasons.append(f"r2:isbn_found({len(isbns)})")

    text_ext = entry.get("r2_text_extraction", {})
    if text_ext.get("has_toc"):
        delta += R2_SCORE_ADJUSTMENTS["toc_detected"]
        reasons.append("r2:toc_detected")

    r2_class = entry.get("r2_classification")
    if r2_class and r2_class.get("confidence") == "high":
        delta += R2_SCORE_ADJUSTMENTS["high_confidence"]
        reasons.append("r2:high_confidence")

    if not text_ext.get("extractable", True):
        delta += R2_SCORE_ADJUSTMENTS["scanned_image_only"]
        reasons.append("r2:scanned_image_only")

    return delta, reasons


def deduplicate_by_fingerprint(entries: list[dict]) -> dict[str, list[str]]:
    """Group entries by content fingerprint. Returns fingerprint → list of sha256s."""
    groups: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        fp = (entry.get("r2_text_extraction") or {}).get("content_fingerprint")
        if fp:
            groups[fp].append(entry["sha256"])
    # Only return groups with >1 member
    return {fp: shas for fp, shas in groups.items() if len(shas) > 1}


def pick_best_copy(entries_in_group: list[dict]) -> str:
    """Pick the best copy from a fingerprint group. Returns sha256 of winner."""
    # Prefer: highest priority_score → largest file → first alphabetically
    return max(
        entries_in_group,
        key=lambda e: (e.get("priority_score", 0), e.get("file_size_mb", 0)),
    )["sha256"]


# ---------------------------------------------------------------------------
# Needs classification output (for Claude Code Phase B)
# ---------------------------------------------------------------------------


def build_needs_classification(
    entries: list[dict],
    checkpoint: dict,
) -> list[dict]:
    """Build list of entries that need Claude Code classification."""
    needs: list[dict] = []
    for entry in entries:
        r2_class = entry.get("r2_classification")
        # Unclassified or low confidence
        if not r2_class or r2_class.get("confidence") in ("unclassified", None):
            needs.append(entry)
        elif (
            entry.get("domain") == "unclassified"
            and r2_class
            and r2_class.get("confidence") == "low"
        ):
            needs.append(entry)

    result = []
    for entry in needs:
        sha = entry["sha256"]
        cp = checkpoint.get(sha, {})
        result.append(
            {
                "sha256": sha,
                "filename": entry.get("filename", ""),
                "title": entry.get("title", ""),
                "authors": entry.get("authors", []),
                "text_sample": cp.get("text_sample", "")[:2000],
                "r2_openlibrary": entry.get("r2_openlibrary"),
                "current_domain": entry.get("domain", "unclassified"),
                "current_confidence": entry.get("confidence", "unclassified"),
            }
        )

    return result


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "sha256",
    "filename",
    "file_size_mb",
    "domain",
    "confidence",
    "priority_score",
    "title",
    "authors",
    "year",
    "publisher",
    "is_book",
    "r2_isbns",
    "r2_ol_title",
    "r2_ol_found",
    "r2_classification_domain",
    "r2_classification_confidence",
    "r2_classification_method",
    "r2_chars_extracted",
    "r2_extractable",
    "r2_has_toc",
    "r2_content_fingerprint",
    "reclassified_as_book",
    "duplicate_group",
    "r2_dedup_is_best_copy",
]


def _flatten_for_csv(entry: dict) -> dict[str, Any]:
    """Flatten nested R2 fields for CSV output."""
    flat: dict[str, Any] = {}
    for k in [
        "sha256",
        "filename",
        "file_size_mb",
        "domain",
        "confidence",
        "priority_score",
        "title",
        "year",
        "publisher",
        "is_book",
        "duplicate_group",
        "reclassified_as_book",
    ]:
        flat[k] = entry.get(k, "")

    flat["authors"] = "; ".join(entry.get("authors") or [])
    flat["r2_isbns"] = "; ".join(entry.get("r2_isbns") or [])

    ol = entry.get("r2_openlibrary") or {}
    flat["r2_ol_title"] = ol.get("title", "")
    flat["r2_ol_found"] = ol.get("found", "")

    cl = entry.get("r2_classification") or {}
    flat["r2_classification_domain"] = cl.get("domain", "")
    flat["r2_classification_confidence"] = cl.get("confidence", "")
    flat["r2_classification_method"] = cl.get("method", "")

    te = entry.get("r2_text_extraction") or {}
    flat["r2_chars_extracted"] = te.get("chars_extracted", "")
    flat["r2_extractable"] = te.get("extractable", "")
    flat["r2_has_toc"] = te.get("has_toc", "")
    flat["r2_content_fingerprint"] = te.get("content_fingerprint", "")

    dd = entry.get("r2_dedup") or {}
    flat["r2_dedup_is_best_copy"] = dd.get("is_best_copy", "")

    return flat


def write_csv(entries: list[dict], path: Path) -> None:
    """Write flattened CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            writer.writerow(_flatten_for_csv(entry))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def load_catalog(path: Path) -> list[dict]:
    """Load a catalog JSON file."""
    if not path.exists():
        log.warning("Catalog not found: %s", path)
        return []
    data = json.loads(path.read_text())
    log.info("Loaded %d entries from %s", len(data), path.name)
    return data


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoint (keyed by sha256)."""
    if path.exists():
        data = json.loads(path.read_text())
        log.info("Loaded checkpoint with %d entries", len(data))
        return data
    return {}


def save_checkpoint(checkpoint: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2))


def run_stage1(
    entries: list[dict],
    checkpoint: dict,
    workers: int,
    quiet: bool,
) -> dict:
    """Stage 1: Text extraction from PDFs."""
    log.info("Stage 1: Text extraction (%d entries, %d workers)", len(entries), workers)

    # Find entries needing extraction
    to_extract: list[dict] = []
    for entry in entries:
        sha = entry["sha256"]
        if sha in checkpoint and checkpoint[sha].get("text_done"):
            continue
        to_extract.append(entry)

    if not to_extract:
        log.info("Stage 1: All entries already extracted (checkpoint)")
        return checkpoint

    log.info("Stage 1: Extracting text from %d PDFs", len(to_extract))
    t0 = time.time()
    done = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for entry in to_extract:
            fp = entry["full_path"]
            fut = executor.submit(_extract_text_from_pdf, fp)
            futures[fut] = entry

        for fut in as_completed(futures):
            entry = futures[fut]
            sha = entry["sha256"]
            try:
                result = fut.result()
                # Build text sample (first 2000 chars for classification)
                all_text = "\n".join(result.get("text_pages", []))
                text_sample = all_text[:2000]

                checkpoint[sha] = {
                    "text_done": True,
                    "text_sample": text_sample,
                    "chars_extracted": result["chars_extracted"],
                    "content_fingerprint": result["content_fingerprint"],
                    "has_toc": result["has_toc"],
                    "extractable": result["extractable"],
                    "isbns": extract_isbns(all_text),
                    "error": result.get("error"),
                }
                done += 1
            except Exception as e:
                checkpoint[sha] = {
                    "text_done": True,
                    "text_sample": "",
                    "chars_extracted": 0,
                    "content_fingerprint": None,
                    "has_toc": False,
                    "extractable": False,
                    "isbns": [],
                    "error": str(e)[:200],
                }
                errors += 1
                done += 1

            if not quiet and done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                log.info(
                    "  Stage 1: %d/%d done (%.0f/sec, %d errors)",
                    done,
                    len(to_extract),
                    rate,
                    errors,
                )

    elapsed = time.time() - t0
    log.info(
        "Stage 1 complete: %d extracted in %.1fs (%d errors)",
        done,
        elapsed,
        errors,
    )
    return checkpoint


def run_stage2_3(
    entries: list[dict],
    checkpoint: dict,
    cache_dir: Path,
    skip_openlibrary: bool,
    quiet: bool,
) -> tuple[dict, dict]:
    """Stages 2+3: ISBN extraction (from checkpoint) + OpenLibrary lookup."""
    # Stage 2: Collect all unique ISBNs
    isbn_to_shas: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        sha = entry["sha256"]
        cp = checkpoint.get(sha, {})
        for isbn in cp.get("isbns", []):
            isbn_to_shas[isbn].append(sha)

    unique_isbns = list(isbn_to_shas.keys())
    log.info("Stage 2: Found %d unique ISBNs from %d entries", len(unique_isbns), len(entries))

    if skip_openlibrary or not unique_isbns:
        log.info("Stage 3: Skipped (--skip-openlibrary or no ISBNs)")
        return checkpoint, {}

    # Stage 3: OpenLibrary lookup
    cache_path = cache_dir / "openlibrary_cache.json"
    ol_cache = _load_ol_cache(cache_path)
    already_cached = sum(1 for isbn in unique_isbns if isbn in ol_cache)
    to_lookup = [isbn for isbn in unique_isbns if isbn not in ol_cache]

    log.info(
        "Stage 3: OpenLibrary lookup — %d cached, %d to fetch",
        already_cached,
        len(to_lookup),
    )

    t0 = time.time()
    for i, isbn in enumerate(to_lookup):
        _lookup_isbn_openlibrary(isbn, ol_cache, cache_path)
        # Batched cache save every 50 lookups (avoid per-lookup I/O storm)
        if (i + 1) % 50 == 0:
            _save_ol_cache(ol_cache, cache_path)
        if not quiet and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            log.info("  Stage 3: %d/%d looked up (%.1fs)", i + 1, len(to_lookup), elapsed)
        # Rate limit: 0.6s between requests
        time.sleep(0.6)

    # Final save after all lookups complete
    _save_ol_cache(ol_cache, cache_path)

    elapsed = time.time() - t0
    log.info("Stage 3 complete: %d lookups in %.1fs", len(to_lookup), elapsed)

    return checkpoint, ol_cache


def run_stage4(
    entries: list[dict],
    checkpoint: dict,
    quiet: bool,
) -> list[dict]:
    """Stage 4: Rule-based classification. Mutates entries in place."""
    log.info("Stage 4: Rule-based classification (%d entries)", len(entries))
    classified = 0
    for entry in entries:
        sha = entry["sha256"]
        cp = checkpoint.get(sha, {})

        result = classify_entry(entry, cp)
        if result:
            entry["r2_classification"] = result
            classified += 1

            # Update top-level domain/confidence only if R2 is better
            old_conf = entry.get("confidence", "unclassified")
            new_conf = result["confidence"]
            conf_rank = {"unclassified": 0, "low": 1, "medium": 2, "high": 3}
            if conf_rank.get(new_conf, 0) > conf_rank.get(old_conf, 0):
                entry["domain"] = result["domain"]
                entry["confidence"] = new_conf

    log.info("Stage 4 complete: %d classified", classified)
    return entries


def run_stage5(
    books: list[dict],
    others: list[dict],
    checkpoint: dict,
    quiet: bool,
) -> tuple[list[dict], list[dict], dict]:
    """Stage 5: Rescore + deduplicate + reclassify others as books."""
    log.info("Stage 5: Rescore + deduplicate")

    all_entries = books + others

    # Apply R2 text extraction fields from checkpoint
    for entry in all_entries:
        sha = entry["sha256"]
        cp = checkpoint.get(sha, {})

        entry["r2_text_extraction"] = {
            "chars_extracted": cp.get("chars_extracted", 0),
            "extractable": cp.get("extractable", True),
            "has_toc": cp.get("has_toc", False),
            "content_fingerprint": cp.get("content_fingerprint"),
        }
        entry["r2_isbns"] = cp.get("isbns", [])

        # Enrichment metadata
        entry["enrichment"] = {
            "round": 2,
            "stages_completed": ["text_extraction", "isbn", "classification", "rescore"],
        }

    # Rescore
    for entry in all_entries:
        delta, reasons = rescore_entry(entry)
        old_score = entry.get("priority_score", 0)
        entry["priority_score"] = max(0, old_score + delta)
        if reasons:
            entry["priority_reasons"] = entry.get("priority_reasons", []) + reasons

    # Content fingerprint dedup
    fp_groups = deduplicate_by_fingerprint(all_entries)
    entry_by_sha = {e["sha256"]: e for e in all_entries}

    for fp, shas in fp_groups.items():
        group_entries = [entry_by_sha[s] for s in shas if s in entry_by_sha]
        best_sha = pick_best_copy(group_entries)
        for e in group_entries:
            e["r2_dedup"] = {
                "content_fingerprint_group": fp,
                "is_best_copy": e["sha256"] == best_sha,
            }
            if e["sha256"] != best_sha:
                e["priority_score"] = max(
                    0, e["priority_score"] + R2_SCORE_ADJUSTMENTS["hash_duplicate_not_preferred"]
                )
                e["priority_reasons"] = e.get("priority_reasons", []) + [
                    "r2:hash_dup_not_preferred"
                ]

    # Mark non-grouped entries
    grouped_shas = {s for shas in fp_groups.values() for s in shas}
    for entry in all_entries:
        if entry["sha256"] not in grouped_shas:
            entry.setdefault(
                "r2_dedup",
                {
                    "content_fingerprint_group": None,
                    "is_best_copy": True,
                },
            )
            delta_unique = R2_SCORE_ADJUSTMENTS["unique_fingerprint"]
            if entry.get("r2_text_extraction", {}).get("content_fingerprint"):
                entry["priority_score"] = entry.get("priority_score", 0) + delta_unique
                entry["priority_reasons"] = entry.get("priority_reasons", []) + [
                    "r2:unique_fingerprint"
                ]

    # Reclassify "other" entries as books if evidence warrants
    reclassified_count = 0
    reclassified_entries: list[dict] = []
    remaining_others: list[dict] = []

    for entry in others:
        sha = entry["sha256"]
        cp = checkpoint.get(sha, {})

        should_promote = False
        if cp.get("has_toc"):
            should_promote = True
        elif cp.get("isbns"):
            should_promote = True
        elif cp.get("chars_extracted", 0) > 5000 and entry.get("file_size_mb", 0) > 2.0:
            should_promote = True

        if should_promote:
            entry["reclassified_as_book"] = True
            entry["is_book"] = True
            entry["is_book_reason"] = "r2_reclassified"
            reclassified_entries.append(entry)
            reclassified_count += 1
        else:
            entry["reclassified_as_book"] = False
            remaining_others.append(entry)

    books_final = books + reclassified_entries
    books_final.sort(key=lambda e: e.get("priority_score", 0), reverse=True)
    remaining_others.sort(key=lambda e: e.get("priority_score", 0), reverse=True)

    log.info(
        "Stage 5 complete: %d fingerprint groups, %d reclassified as books",
        len(fp_groups),
        reclassified_count,
    )

    summary = {
        "total_books_r2": len(books_final),
        "total_other_r2": len(remaining_others),
        "reclassified_as_books": reclassified_count,
        "content_fingerprint_groups": len(fp_groups),
        "domain_distribution": dict(
            Counter(e.get("domain", "unclassified") for e in books_final).most_common()
        ),
        "confidence_distribution": dict(
            Counter(e.get("confidence", "unclassified") for e in books_final).most_common()
        ),
        "priority_distribution": {
            "80+": sum(1 for e in books_final if e.get("priority_score", 0) >= 80),
            "60-79": sum(1 for e in books_final if 60 <= e.get("priority_score", 0) < 80),
            "40-59": sum(1 for e in books_final if 40 <= e.get("priority_score", 0) < 60),
            "20-39": sum(1 for e in books_final if 20 <= e.get("priority_score", 0) < 40),
            "0-19": sum(1 for e in books_final if e.get("priority_score", 0) < 20),
        },
        "isbns_found": sum(1 for e in books_final if e.get("r2_isbns")),
        "ol_hits": sum(1 for e in books_final if (e.get("r2_openlibrary") or {}).get("found")),
        "extractable": sum(
            1 for e in books_final if (e.get("r2_text_extraction") or {}).get("extractable")
        ),
        "non_extractable": sum(
            1
            for e in books_final
            if not (e.get("r2_text_extraction") or {}).get("extractable", True)
        ),
        "avg_priority_score": round(
            sum(e.get("priority_score", 0) for e in books_final) / max(len(books_final), 1), 1
        ),
    }

    return books_final, remaining_others, summary


# ---------------------------------------------------------------------------
# Merge mode (Phase C)
# ---------------------------------------------------------------------------


def run_merge(
    classified_path: Path,
    output_dir: Path,
) -> None:
    """Merge Claude Code classifications into enriched catalog."""
    # Load existing R2 catalogs
    books_path = output_dir / "catalog_books_r2.json"
    other_path = output_dir / "catalog_other_r2.json"

    if not books_path.exists():
        log.error("No R2 catalog found at %s — run full pipeline first", books_path)
        sys.exit(1)

    books = json.loads(books_path.read_text())
    others = json.loads(other_path.read_text()) if other_path.exists() else []
    classified = json.loads(classified_path.read_text())

    log.info(
        "Merging %d classifications into %d books + %d others",
        len(classified),
        len(books),
        len(others),
    )

    # Index by sha256
    all_entries = {e["sha256"]: e for e in books + others}

    merged = 0
    for item in classified:
        sha = item.get("sha256")
        if sha not in all_entries:
            log.warning("Classification for unknown sha256: %s", sha[:12])
            continue

        entry = all_entries[sha]
        entry["r2_classification"] = {
            "domain": item["domain"],
            "confidence": item["confidence"],
            "method": "claude_code",
            "reasoning": item.get("reasoning", ""),
        }

        # Update top-level if better
        conf_rank = {"unclassified": 0, "low": 1, "medium": 2, "high": 3}
        old_conf = entry.get("confidence", "unclassified")
        new_conf = item["confidence"]
        if conf_rank.get(new_conf, 0) > conf_rank.get(old_conf, 0):
            entry["domain"] = item["domain"]
            entry["confidence"] = new_conf

        if "classification" not in entry.get("enrichment", {}).get("stages_completed", []):
            entry.setdefault("enrichment", {"round": 2, "stages_completed": []})
            entry["enrichment"]["stages_completed"].append("claude_code_classification")

        merged += 1

    # Re-split
    books_final = [e for e in all_entries.values() if e.get("is_book")]
    others_final = [e for e in all_entries.values() if not e.get("is_book")]
    books_final.sort(key=lambda e: e.get("priority_score", 0), reverse=True)
    others_final.sort(key=lambda e: e.get("priority_score", 0), reverse=True)

    # Write outputs
    _write_outputs(books_final, others_final, output_dir, is_merge=True)
    log.info("Merge complete: %d classifications applied", merged)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _write_outputs(
    books: list[dict],
    others: list[dict],
    output_dir: Path,
    is_merge: bool = False,
    summary: dict | None = None,
) -> None:
    """Write all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    books_path = output_dir / "catalog_books_r2.json"
    other_path = output_dir / "catalog_other_r2.json"
    books_path.write_text(json.dumps(books, indent=2, ensure_ascii=False))
    other_path.write_text(json.dumps(others, indent=2, ensure_ascii=False))
    log.info("Wrote %s (%d entries)", books_path.name, len(books))
    log.info("Wrote %s (%d entries)", other_path.name, len(others))

    # CSV
    write_csv(books, output_dir / "catalog_books_r2.csv")
    write_csv(others, output_dir / "catalog_other_r2.csv")
    log.info("Wrote CSV files")

    # Summary
    if summary is None:
        summary = {
            "total_books_r2": len(books),
            "total_other_r2": len(others),
            "domain_distribution": dict(
                Counter(e.get("domain", "unclassified") for e in books).most_common()
            ),
            "confidence_distribution": dict(
                Counter(e.get("confidence", "unclassified") for e in books).most_common()
            ),
            "merged": is_merge,
        }

    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary_path = output_dir / "catalog_summary_r2.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", summary_path.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Round 2 Catalog Enrichment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--input",
        type=Path,
        default=Path("fixtures/library_catalog/catalog_books.json"),
        help="Round 1 books catalog",
    )
    p.add_argument(
        "--input-other",
        type=Path,
        default=Path("fixtures/library_catalog/catalog_other.json"),
        help="Round 1 other catalog",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fixtures/library_catalog/"),
        help="Output directory",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("fixtures/library_catalog/.enrichment_checkpoint.json"),
        help="Checkpoint file for resumability",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("fixtures/library_catalog/.cache/"),
        help="Cache directory for API responses",
    )
    p.add_argument("--workers", type=int, default=8, help="Parallel workers for text extraction")
    p.add_argument(
        "--skip-openlibrary", action="store_true", help="Skip OpenLibrary lookups (offline mode)"
    )
    p.add_argument("--limit", type=int, default=0, help="Process only first N entries (testing)")
    p.add_argument("--quiet", action="store_true", help="Minimal output")
    p.add_argument(
        "--merge", type=Path, default=None, help="Merge classified_results.json (Phase C)"
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.quiet)

    # Phase C: Merge mode
    if args.merge:
        run_merge(args.merge, args.output_dir)
        return

    # Load catalogs
    books = load_catalog(args.input)
    others = load_catalog(args.input_other)

    if args.limit > 0:
        books = books[: args.limit]
        others = others[: max(1, args.limit // 3)]
        log.info("Limited to %d books + %d others", len(books), len(others))

    all_entries = books + others
    log.info(
        "Processing %d total entries (%d books + %d other)",
        len(all_entries),
        len(books),
        len(others),
    )

    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint)

    # Stage 1: Text extraction
    t_start = time.time()
    checkpoint = run_stage1(all_entries, checkpoint, args.workers, args.quiet)
    save_checkpoint(checkpoint, args.checkpoint)

    # Stage 2+3: ISBN extraction + OpenLibrary
    checkpoint, ol_cache = run_stage2_3(
        all_entries,
        checkpoint,
        args.cache_dir,
        args.skip_openlibrary,
        args.quiet,
    )
    save_checkpoint(checkpoint, args.checkpoint)

    # Apply OpenLibrary data to entries
    if ol_cache:
        for entry in all_entries:
            sha = entry["sha256"]
            cp = checkpoint.get(sha, {})
            for isbn in cp.get("isbns", []):
                if isbn in ol_cache and ol_cache[isbn].get("found"):
                    entry["r2_openlibrary"] = ol_cache[isbn]
                    break
            else:
                entry["r2_openlibrary"] = None
    else:
        for entry in all_entries:
            entry["r2_openlibrary"] = None

    # Stage 4: Classification
    run_stage4(all_entries, checkpoint, args.quiet)

    # Stage 5: Rescore + dedup
    books_final, others_final, summary = run_stage5(
        books,
        others,
        checkpoint,
        args.quiet,
    )

    # Build needs_classification.json
    needs = build_needs_classification(books_final, checkpoint)
    needs_path = args.output_dir / "needs_classification.json"
    needs_path.write_text(json.dumps(needs, indent=2, ensure_ascii=False))
    log.info("Wrote needs_classification.json (%d entries)", len(needs))
    summary["needs_classification"] = len(needs)

    # Write outputs
    _write_outputs(books_final, others_final, args.output_dir, summary=summary)

    elapsed = time.time() - t_start
    summary["elapsed_seconds"] = round(elapsed, 1)
    # Re-write summary with elapsed
    (args.output_dir / "catalog_summary_r2.json").write_text(json.dumps(summary, indent=2))

    # Print summary to stdout
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
