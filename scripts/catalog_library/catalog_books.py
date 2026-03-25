#!/usr/bin/env python3
"""
Full Library Catalog & Classification Pipeline.

Scans a directory of PDFs, extracts metadata, classifies domains,
detects duplicates, assigns priority scores, and outputs structured
catalogs (JSON + CSV) split into books vs. non-books.

Usage:
    python scripts/catalog_library/catalog_books.py \
      --input-dir /media/.../books/ \
      --output-dir fixtures/library_catalog/ \
      --workers 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Domain taxonomy
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    # Math sub-domains
    "algebra": [
        "algebra",
        "galois",
        "group theory",
        "ring theory",
        "field theory",
        "commutative",
        "homological",
        "representation theory",
        "module theory",
        "abstract algebra",
        "algebraic structure",
        "lie algebra",
        "lie group",
    ],
    "analysis": [
        "real analysis",
        "complex analysis",
        "functional analysis",
        "harmonic analysis",
        "measure theory",
        "fourier",
        "operator theory",
        "spectral theory",
        "lebesgue",
        "banach",
        "hilbert space",
        "sobolev",
        "distribution theory",
    ],
    "topology_geometry": [
        "topology",
        "manifold",
        "riemannian",
        "cohomology",
        "homotopy",
        "differential geometry",
        "algebraic geometry",
        "knot theory",
        "symplectic",
        "differential forms",
        "fiber bundle",
        "sheaf",
        "homology",
        "algebraic topology",
    ],
    "dynamical_systems": [
        "dynamical system",
        "bifurcation",
        "chaos",
        "hamiltonian",
        "perturbation",
        "stability theory",
        "nonlinear dynamics",
        "celestial mechanics",
        "n-body",
        "vortex",
        "vortices",
        "orbit",
        "ergodic",
        "kam theory",
        "kam theorem",
        "three-body",
        "poincare",
        "lyapunov",
        "classical dynamics",
        "classical mechanics",
        "lagrangian mechanics",
    ],
    "probability_theory": [
        "probability theory",
        "stochastic process",
        "stochastic calculus",
        "markov chain",
        "martingale",
        "random variable",
        "brownian motion",
        "measure-theoretic probability",
        "stochastic differential",
        "random walk",
        "poisson process",
    ],
    "numerical_methods": [
        "numerical analysis",
        "numerical method",
        "convex optimization",
        "finite element",
        "approximation theory",
        "interpolation",
        "computational mathematics",
        "numerical linear algebra",
        "numerical solution",
        "quadrature",
    ],
    "linear_algebra": [
        "linear algebra",
        "matrix analysis",
        "eigenvalue",
        "vector space",
        "singular value",
        "svd",
        "tensor decomposition",
        "matrix theory",
        "matrix computation",
    ],
    # Existing research-kb domains (expanded)
    "causal_inference": [
        "causal inference",
        "causal effect",
        "counterfactual",
        "instrumental variable",
        "propensity score",
        "treatment effect",
        "difference-in-difference",
        "regression discontinuity",
        "double machine learning",
        "potential outcome",
    ],
    "deep_learning": [
        "deep learning",
        "neural network",
        "transformer",
        "attention mechanism",
        "bert",
        "convolutional neural",
        "diffusion model",
        "generative adversarial",
        "autoencoder",
        "recurrent neural",
        "lstm",
        "batch normalization",
    ],
    "machine_learning": [
        "machine learning",
        "probabilistic graphical",
        "mixture model",
        "kernel method",
        "ensemble method",
        "boosting",
        "random forest",
        "support vector",
        "decision tree",
        "classification",
        "supervised learning",
        "unsupervised learning",
    ],
    "reinforcement_learning": [
        "reinforcement learning",
        "reward function",
        "policy gradient",
        "q-learning",
        "bandit",
        "markov decision",
        "temporal difference",
        "actor-critic",
        "monte carlo tree",
    ],
    "rag_llm": [
        "natural language processing",
        "language model",
        "information retrieval",
        "text mining",
        "word embedding",
        "sentiment analysis",
        "named entity",
        "question answering",
        "text generation",
        "prompt engineering",
        "retrieval augmented",
        "large language model",
    ],
    "statistics": [
        "statistical inference",
        "hypothesis testing",
        "generalized linear",
        "survival analysis",
        "bayesian statistics",
        "bayesian inference",
        "multivariate statistics",
        "nonparametric statistics",
        "time series analysis",
        "regression analysis",
        "mixed model",
        "experimental design",
        "statistical learning",
    ],
    "econometrics": [
        "econometrics",
        "panel data",
        "instrumental variable",
        "generalized method of moments",
        "cointegration",
        "heteroskedasticity",
        "endogeneity",
        "two-stage least squares",
    ],
    "finance": [
        "value at risk",
        "derivatives pricing",
        "quantitative finance",
        "algorithmic trading",
        "fixed income",
        "credit risk",
        "option pricing",
        "portfolio theory",
        "asset pricing",
        "black-scholes",
        "stochastic finance",
        "financial engineering",
        "risk management",
    ],
    "portfolio_management": [
        "portfolio management",
        "portfolio optimization",
        "asset allocation",
        "modern portfolio theory",
        "factor model",
        "capm",
        "mean-variance",
        "sharpe ratio",
    ],
    "actuarial_insurance": [
        "actuarial",
        "insurance mathematics",
        "loss model",
        "ruin theory",
        "life contingencies",
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
        "solid state physics",
        "optics",
    ],
    "biology_neuroscience": [
        "biology",
        "neuroscience",
        "genetics",
        "evolution",
        "ecology",
        "molecular biology",
        "neural network biology",
        "brain science",
        "computational neuroscience",
        "bioinformatics",
        "genomics",
    ],
    "signal_processing": [
        "signal processing",
        "digital signal",
        "control theory",
        "control systems",
        "feedback control",
        "filter design",
        "fourier transform",
        "wavelet",
        "system identification",
    ],
    "time_series": [
        "time series",
        "forecasting",
        "arima",
        "garch",
        "kalman filter",
        "state space model",
        "spectral analysis",
    ],
    "algorithms": [
        "algorithm design",
        "data structure",
        "computational complexity",
        "graph algorithm",
        "combinatorial optimization",
        "dynamic programming",
        "divide and conquer",
        "greedy algorithm",
    ],
    "software_engineering": [
        "software engineering",
        "design pattern",
        "software architecture",
        "refactoring",
        "continuous integration",
        "devops",
        "microservice",
        "distributed system",
        "version control",
        "clean code",
        "test-driven",
        "agile",
        "kubernetes",
        "docker",
        "cloud native",
        "site reliability",
        "programming language",
        "compiler",
        "continuous delivery",
        "infrastructure as code",
    ],
    "data_science": [
        "data science",
        "data mining",
        "data analysis",
        "data visualization",
        "exploratory data",
        "feature engineering",
        "data pipeline",
        "data analytics",
        "scientific computing",
        "python data",
    ],
    "functional_programming": [
        "functional programming",
        "haskell",
        "lambda calculus",
        "category theory",
        "monad",
        "type theory",
        "scala functional",
    ],
    "mathematics": [
        # Catch-all for math that doesn't fit sub-domains
        "mathematics",
        "mathematical",
        "theorem",
        "proof",
        "lemma",
        "calculus",
        "differential equation",
        "number theory",
        "combinatorics",
        "discrete mathematics",
        "graph theory",
        "set theory",
        "logic",
    ],
    "interview_prep": [
        "interview",
        "coding interview",
        "system design interview",
        "cracking the code",
        "programming interview",
    ],
    "sql": [
        "sql",
        "database design",
        "relational database",
        "query optimization",
    ],
    "recommender_systems": [
        "recommender system",
        "recommendation",
        "collaborative filtering",
        "content-based filtering",
        "matrix factorization",
    ],
    "adtech": [
        "advertising technology",
        "computational advertising",
        "ad serving",
        "real-time bidding",
        "programmatic",
    ],
    "fitness": [
        "fitness",
        "strength training",
        "exercise",
        "bodybuilding",
        "supple leopard",
        "mobility",
        "crossfit",
        "weightlifting",
        "athletic performance",
        "sport science",
    ],
    "ml_engineering": [
        "mlops",
        "ml engineering",
        "model deployment",
        "model serving",
        "feature store",
        "ml pipeline",
        "machine learning engineering",
    ],
    "forecasting": [
        "forecasting method",
        "demand forecasting",
        "predictive analytics",
        "prophet",
        "exponential smoothing",
    ],
}

# ODE/PDE keywords — check context to route correctly
ODE_PDE_KEYWORDS = [
    "ordinary differential equation",
    "partial differential equation",
    "ode",
    "pde",
    "boundary value",
    "initial value problem",
    "differential equation",
]


# ---------------------------------------------------------------------------
# Non-book detection patterns
# ---------------------------------------------------------------------------

NON_BOOK_FILENAME_PATTERNS = [
    re.compile(r"\bexam\b", re.I),
    re.compile(r"\bsyllabus\b", re.I),
    re.compile(r"\bhomework\b", re.I),
    re.compile(r"\bassignment\b", re.I),
    re.compile(r"\bquiz\b", re.I),
    re.compile(r"\bmidterm\b", re.I),
    re.compile(r"\bfinal\s*exam\b", re.I),
    re.compile(r"\bHW_", re.I),
    re.compile(r"_CV\b"),
    re.compile(r"\bbrochure\b", re.I),
    re.compile(r"\binvoice\b", re.I),
    re.compile(r"\breceipt\b", re.I),
    re.compile(r"\bslide[s]?\b", re.I),
    re.compile(r"\blecture_?\d", re.I),
    re.compile(r"^compressed_", re.I),
    re.compile(r"\.pdf\.pdf$", re.I),
    re.compile(r"^\d{5,}\.pdf$"),  # purely numeric
    re.compile(r"^ch(apter)?\s*\d+", re.I),
    re.compile(r"^\d{1,2}_intro", re.I),
    re.compile(r"\bsolution\s*manual\b", re.I),
    re.compile(r"\bresum[eé]\b", re.I),
    re.compile(r"\bcertificate\b", re.I),
]

NON_BOOK_METADATA_PATTERNS = [
    re.compile(r"\bexam\b", re.I),
    re.compile(r"\bassignment\b", re.I),
    re.compile(r"\bslide\b", re.I),
    re.compile(r"\blecture note\b", re.I),
    re.compile(r"\bpresentation\b", re.I),
]


# ---------------------------------------------------------------------------
# Priority score components
# ---------------------------------------------------------------------------

KNOWN_PUBLISHERS_TIER1 = {
    "springer",
    "cambridge",
    "mit press",
    "oxford",
    "wiley",
    "princeton",
    "ams",
    "american mathematical society",
    "siam",
    "world scientific",
}
KNOWN_PUBLISHERS_TIER2 = {
    "pearson",
    "mcgraw-hill",
    "mcgraw hill",
    "crc",
    "academic press",
    "elsevier",
    "addison-wesley",
    "addison wesley",
    "o'reilly",
    "oreilly",
    "manning",
    "no starch",
    "prentice hall",
    "cengage",
    "brooks/cole",
    "john wiley",
    "chapman",
    "birkhauser",
    "birkhäuser",
}

KNOWN_SERIES = [
    "graduate texts in mathematics",
    "springer graduate",
    "undergraduate texts in mathematics",
    "lecture notes in mathematics",
    "applied mathematical sciences",
    "cambridge tracts",
    "princeton lectures",
    "ams graduate studies",
    "dover books on mathematics",
    "springer texts in statistics",
    "information science and statistics",
    "texts in applied mathematics",
    "classics in mathematics",
    "adaptive computation and machine learning",
]

CANONICAL_AUTHORS = [
    "rudin",
    "strang",
    "lang",
    "munkres",
    "strogatz",
    "durrett",
    "boyd",
    "axler",
    "murphy",
    "sutton",
    "bishop",
    "hastie",
    "goodfellow",
    "arnold",
    "marsden",
    "folland",
    "royden",
    "hungerford",
    "do carmo",
    "nocedal",
    "trefethen",
    "halmos",
    "hoffman",
    "kunze",
    "artin",
    "dummit",
    "foote",
    "hatcher",
    "lee",
    "milnor",
    "spivak",
    "stein",
    "shakarchi",
    "evans",
    "brezis",
    "conway",
    "kreyszig",
    "berger",
    "ahlfors",
    "serre",
    "atiyah",
    "macdonald",
    "hartshorne",
    "griffiths",
    "karatzas",
    "shreve",
    "oksendal",
    "billingsley",
    "kolmogorov",
    "shiryaev",
    "vapnik",
    "tibshirani",
    "gelman",
    "wasserman",
    "casella",
    "berger",
    "bertsekas",
    "tsitsiklis",
    "koller",
    "friedman",
    "jurafsky",
    "manning",
    "bengio",
    "lecun",
    "cover",
    "thomas",
    "haykin",
    "shalev-shwartz",
    "rockafellar",
    "bertsekas",
    "luenberger",
    "hamilton",
    "enders",
    "wooldridge",
    "angrist",
    "pischke",
    "pearl",
    "imbens",
    "chernozhukov",
    "meyer",
    "moler",
    "burden",
    "faires",
    "feynman",
    "landau",
    "lifshitz",
    "jackson",
    "griffiths",
    "sakurai",
    "weinberg",
    "peskin",
    "schroeder",
    "goldstein",
    "poole",
    "safko",
    "cohen-tannoudji",
    "shankar",
]


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Pattern: (Author) Title (Year, Publisher)
_RE_PAREN_AUTHOR = re.compile(
    r"^\((?P<series_or_tag>[^)]*)\)\s+"
    r"(?P<author_block>(?:[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*(?:,\s*)?)+)\s*[-_]\s*"
    r"(?P<title>.+?)(?:\s*[-_]\s*(?P<publisher>[A-Za-z][A-Za-z &.']+))?"
    r"\s*\((?P<year>\d{4})\)"
)

# Pattern: Author - Title - Publisher (Year)
_RE_AUTHOR_TITLE = re.compile(
    r"^(?P<author_block>[A-Z][A-Za-z .,']+(?:,\s*[A-Z][A-Za-z .,']+)*)\s*[-_]+\s*"
    r"(?P<title>.+?)(?:\s*[-_]+\s*(?P<publisher>[A-Za-z][A-Za-z &.']+))?"
    r"\s*\((?P<year>\d{4})\)"
)

# Pattern: Title (Year)
_RE_TITLE_YEAR = re.compile(r"^(?P<title>.+?)\s*\((?P<year>\d{4})\)")

# Year anywhere
_RE_YEAR_ANYWHERE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")

# Edition
_RE_EDITION = re.compile(r"(\d+)(?:st|nd|rd|th)\s*(?:ed(?:ition)?)", re.I)
_RE_EDITION_WORD = re.compile(r"\bedition\b", re.I)

# Series in parentheses
_RE_SERIES = re.compile(r"\(([^)]*(?:series|texts|lectures|notes|studies)[^)]*)\)", re.I)


def parse_filename(filename: str) -> dict[str, Any]:
    """Extract metadata from PDF filename.

    Returns dict with keys: title, authors, year, publisher, series, edition.
    """
    stem = Path(filename).stem
    # Clean up common artifacts
    stem = stem.replace("_ ", ": ").replace("_-_", " - ")

    result: dict[str, Any] = {
        "title": None,
        "authors": [],
        "year": None,
        "publisher": None,
        "series": None,
        "edition": None,
    }

    # Try structured patterns
    for pat in [_RE_PAREN_AUTHOR, _RE_AUTHOR_TITLE]:
        m = pat.match(stem)
        if m:
            gd = m.groupdict()
            result["title"] = _clean_title(gd.get("title") or "")
            result["authors"] = _parse_authors(gd.get("author_block") or "")
            result["year"] = int(gd["year"]) if gd.get("year") else None
            result["publisher"] = (gd.get("publisher") or "").strip(" -_") or None
            if "series_or_tag" in gd and gd["series_or_tag"]:
                tag = gd["series_or_tag"]
                if not tag.replace(".", "").isdigit():
                    result["series"] = tag
            break

    # Fallback: title + year
    if result["title"] is None:
        m = _RE_TITLE_YEAR.match(stem)
        if m:
            result["title"] = _clean_title(m.group("title"))
            result["year"] = int(m.group("year"))
        else:
            result["title"] = _clean_title(stem)

    # Year fallback
    if result["year"] is None:
        m = _RE_YEAR_ANYWHERE.search(stem)
        if m:
            result["year"] = int(m.group(1))

    # Edition
    m = _RE_EDITION.search(stem)
    if m:
        result["edition"] = int(m.group(1))
    elif _RE_EDITION_WORD.search(stem):
        result["edition"] = 1  # "edition" mentioned but no number

    # Series
    m = _RE_SERIES.search(stem)
    if m:
        result["series"] = m.group(1).strip()

    # Check for known series in title
    if result["series"] is None and result["title"]:
        title_lower = result["title"].lower()
        for series in KNOWN_SERIES:
            if series in title_lower:
                result["series"] = series.title()
                break

    return result


def _clean_title(raw: str) -> str:
    """Clean up extracted title string."""
    t = raw.strip(" -_.,")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^[\(\[].*?[\)\]]\s*", "", t)  # leading tags
    t = re.sub(r"\s*-\s*libgen\.?li\.?$", "", t, flags=re.I)
    t = re.sub(r"\s*-\s*libgenli\.?$", "", t, flags=re.I)
    t = t.strip(" -_.,")
    return t if t else "Unknown"


def _parse_authors(raw: str) -> list[str]:
    """Parse author block into list of author names."""
    if not raw:
        return []
    # Split on common separators
    parts = re.split(r"\s*(?:,\s*(?:and\s+)?|;\s*|\s+and\s+)", raw.strip())
    authors = []
    for p in parts:
        p = p.strip(" .-_")
        if p and len(p) > 2 and not p.isdigit():
            authors.append(p)
    return authors


# ---------------------------------------------------------------------------
# PDF metadata extraction
# ---------------------------------------------------------------------------


def extract_pdf_metadata(filepath: str) -> dict[str, str | None]:
    """Extract metadata from PDF file using pypdf.

    Returns dict with keys: title, author, subject, creator, producer, pages.
    Catches all exceptions — corrupt PDFs return empty metadata.
    Suppresses pypdf's noisy warnings about malformed PDFs.
    """
    import logging
    import warnings

    result: dict[str, str | None] = {
        "title": None,
        "author": None,
        "subject": None,
        "creator": None,
        "producer": None,
        "pages": None,
    }
    try:
        from pypdf import PdfReader

        # Suppress all pypdf warnings (corrupt PDFs are extremely noisy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.getLogger("pypdf").setLevel(logging.CRITICAL)
            reader = PdfReader(filepath)
            info = reader.metadata
            if info:
                result["title"] = _safe_str(info.title)
                result["author"] = _safe_str(info.author)
                result["subject"] = _safe_str(info.subject)
                result["creator"] = _safe_str(info.creator)
                result["producer"] = _safe_str(info.producer)
            result["pages"] = str(len(reader.pages))
    except Exception:
        pass
    return result


def _safe_str(val: Any) -> str | None:
    """Convert PDF metadata value to string, returning None for empty."""
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# SHA256 hashing
# ---------------------------------------------------------------------------


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file. Reads in 64KB chunks."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
    except Exception:
        return "ERROR"
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------


def classify_domain(title: str, filename: str, pdf_meta: dict[str, str | None]) -> tuple[str, str]:
    """Classify a document into a domain based on title/filename/metadata.

    Returns (domain_id, confidence) where confidence is high/medium/low/unclassified.
    """
    # Build search corpus from all available text
    texts = []
    for s in [title, filename, pdf_meta.get("title"), pdf_meta.get("subject")]:
        if s:
            texts.append(s.lower())
    corpus = " ".join(texts)

    # Score each domain
    scores: dict[str, int] = defaultdict(int)
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in corpus:
                scores[domain] += 1

    # Handle ODE/PDE ambiguity
    has_ode_pde = any(kw in corpus for kw in ODE_PDE_KEYWORDS)
    if has_ode_pde:
        # Route to dynamical_systems if chaos/stability signals present
        if scores.get("dynamical_systems", 0) > 0:
            scores["dynamical_systems"] += 2
        # Route to numerical_methods if numerical signals present
        elif scores.get("numerical_methods", 0) > 0:
            scores["numerical_methods"] += 2
        # Default to mathematics
        else:
            scores["mathematics"] += 1

    if not scores:
        return "unclassified", "unclassified"

    # Pick winner
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_domain, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # Confidence based on score and margin
    if best_score >= 3:
        confidence = "high"
    elif best_score >= 2 and best_score > second_score:
        confidence = "medium"
    elif best_score >= 1:
        confidence = "low"
    else:
        confidence = "unclassified"

    return best_domain, confidence


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------


def compute_priority_score(
    entry: dict[str, Any],
    is_preferred_duplicate: bool = True,
) -> tuple[int, list[str]]:
    """Compute priority score (0-100) for a catalog entry.

    Returns (score, list_of_reasons).
    """
    score = 0
    reasons: list[str] = []

    filename_lower = entry.get("filename", "").lower()
    title_lower = (entry.get("title") or "").lower()
    authors = entry.get("authors") or []
    authors_str = " ".join(authors).lower()
    size_mb = entry.get("file_size_mb", 0)
    publisher = (entry.get("publisher") or "").lower()
    pdf_producer = (entry.get("pdf_metadata", {}).get("producer") or "").lower()
    pdf_creator = (entry.get("pdf_metadata", {}).get("creator") or "").lower()
    combined_pub = f"{publisher} {pdf_producer} {pdf_creator}"

    # Publisher signals (+20 max)
    matched_t1 = [p for p in KNOWN_PUBLISHERS_TIER1 if p in combined_pub]
    matched_t2 = [p for p in KNOWN_PUBLISHERS_TIER2 if p in combined_pub]
    if matched_t1:
        score += 20
        reasons.append(f"publisher_tier1:{matched_t1[0]}")
    elif matched_t2:
        score += 15
        reasons.append(f"publisher_tier2:{matched_t2[0]}")

    # Also check filename for publisher
    if not matched_t1 and not matched_t2:
        fn_combined = f"{filename_lower} {title_lower}"
        matched_fn_t1 = [p for p in KNOWN_PUBLISHERS_TIER1 if p in fn_combined]
        matched_fn_t2 = [p for p in KNOWN_PUBLISHERS_TIER2 if p in fn_combined]
        if matched_fn_t1:
            score += 20
            reasons.append(f"publisher_tier1:{matched_fn_t1[0]}")
        elif matched_fn_t2:
            score += 15
            reasons.append(f"publisher_tier2:{matched_fn_t2[0]}")

    # Metadata quality (+20 max)
    if authors:
        score += 10
        reasons.append("has_author")
    if entry.get("year"):
        score += 5
        reasons.append("has_year")
    if entry.get("series"):
        score += 5
        reasons.append(f"series:{entry['series'][:30]}")

    # Size as proxy (+20 max)
    if size_mb > 50:
        score += 20
        reasons.append("size>50MB")
    elif size_mb > 20:
        score += 15
        reasons.append("size>20MB")
    elif size_mb > 10:
        score += 10
        reasons.append("size>10MB")
    elif size_mb > 5:
        score += 5
        reasons.append("size>5MB")

    # Edition signal (+10)
    if (
        entry.get("edition")
        or _RE_EDITION.search(filename_lower)
        or _RE_EDITION_WORD.search(filename_lower)
    ):
        score += 10
        reasons.append("has_edition")

    # Canonical authors (+15)
    matched_authors = [a for a in CANONICAL_AUTHORS if a in authors_str or a in filename_lower]
    if matched_authors:
        score += 15
        reasons.append(f"canonical_author:{matched_authors[0]}")

    # Page count signal (+5)
    pages_str = entry.get("pdf_metadata", {}).get("pages")
    if pages_str and pages_str.isdigit():
        pages = int(pages_str)
        if pages > 300:
            score += 5
            reasons.append(f"pages:{pages}")

    # Negative signals
    if "meap" in filename_lower:
        score -= 10
        reasons.append("meap_prerelease")
    if re.search(r"_v\d", filename_lower):
        score -= 5
        reasons.append("version_variant")
    if not is_preferred_duplicate and entry.get("duplicate_group"):
        score -= 20
        reasons.append("non_preferred_duplicate")
    if "draft" in filename_lower:
        score -= 5
        reasons.append("draft_version")

    return max(0, min(100, score)), reasons


# ---------------------------------------------------------------------------
# Book vs non-book classification
# ---------------------------------------------------------------------------


def classify_is_book(
    filename: str,
    size_mb: float,
    pdf_meta: dict[str, str | None],
) -> tuple[bool, str]:
    """Determine if a PDF is a book or non-book (slides, exams, etc.).

    Returns (is_book, reason).
    """
    # Size check
    if size_mb < 0.5:
        return False, "size<0.5MB"

    # Filename pattern checks
    for pat in NON_BOOK_FILENAME_PATTERNS:
        if pat.search(filename):
            return False, f"filename_pattern:{pat.pattern}"

    # PDF metadata checks
    for meta_field in ["title", "subject"]:
        val = pdf_meta.get(meta_field)
        if val:
            for pat in NON_BOOK_METADATA_PATTERNS:
                if pat.search(val):
                    return False, f"metadata_pattern:{pat.pattern}"

    # Size threshold for ambiguous cases
    if size_mb < 1.0:
        return False, "size<1MB"

    return True, "passes_all_checks"


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def detect_duplicates(entries: list[dict[str, Any]]) -> None:
    """Detect duplicates by SHA256 hash and assign duplicate_group IDs.

    Modifies entries in-place, setting 'duplicate_group' and 'flags'.
    """
    # Exact hash duplicates
    hash_groups: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        sha = e.get("sha256", "")
        if sha and sha != "ERROR":
            hash_groups[sha].append(i)

    group_counter = 0
    for sha, indices in hash_groups.items():
        if len(indices) > 1:
            group_counter += 1
            group_id = f"hash_dup_{group_counter}"
            # Prefer larger filename (often more descriptive)
            best_idx = max(indices, key=lambda i: len(entries[i].get("filename", "")))
            for idx in indices:
                entries[idx]["duplicate_group"] = group_id
                if "possible_duplicate" not in entries[idx].get("flags", []):
                    entries[idx].setdefault("flags", []).append("possible_duplicate")
                entries[idx]["_is_preferred_duplicate"] = idx == best_idx

    # Fuzzy title duplicates (within same domain, different hash)
    domain_title_groups: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        if e.get("duplicate_group"):
            continue  # already grouped
        title = (e.get("title") or "").lower()
        title_key = _normalize_title(title)
        if title_key and len(title_key) > 10:
            domain = e.get("domain", "unclassified")
            key = f"{domain}::{title_key}"
            domain_title_groups[key].append(i)

    for key, indices in domain_title_groups.items():
        if len(indices) > 1:
            group_counter += 1
            group_id = f"fuzzy_dup_{group_counter}"
            # Prefer largest file
            best_idx = max(indices, key=lambda i: entries[i].get("file_size_mb", 0))
            for idx in indices:
                entries[idx]["duplicate_group"] = group_id
                if "possible_duplicate" not in entries[idx].get("flags", []):
                    entries[idx].setdefault("flags", []).append("possible_duplicate")
                entries[idx]["_is_preferred_duplicate"] = idx == best_idx


def _normalize_title(title: str) -> str:
    """Normalize title for fuzzy duplicate matching."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Remove common suffixes
    for suffix in ["pdf", "libgenli", "libgen li", "ebook"]:
        t = t.replace(suffix, "")
    return t.strip()


# ---------------------------------------------------------------------------
# Catalog entry dataclass
# ---------------------------------------------------------------------------


@dataclass
class CatalogEntry:
    filename: str
    full_path: str
    file_size_mb: float
    sha256: str
    domain: str
    confidence: str
    is_book: bool
    is_book_reason: str
    flags: list[str] = field(default_factory=list)
    priority_score: int = 0
    priority_reasons: list[str] = field(default_factory=list)
    title: str | None = None
    title_source: str = "filename"
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    series: str | None = None
    edition: int | None = None
    publisher: str | None = None
    pdf_metadata: dict[str, str | None] = field(default_factory=dict)
    duplicate_group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_csv_row(self) -> dict[str, str]:
        return {
            "filename": self.filename,
            "size_mb": f"{self.file_size_mb:.1f}",
            "domain": self.domain,
            "confidence": self.confidence,
            "is_book": str(self.is_book),
            "title": self.title or "",
            "authors_str": "; ".join(self.authors),
            "year": str(self.year) if self.year else "",
            "priority_score": str(self.priority_score),
            "duplicate_group": self.duplicate_group or "",
            "flags_str": ", ".join(self.flags),
            "sha256": self.sha256,
            "pages": self.pdf_metadata.get("pages") or "",
        }


# ---------------------------------------------------------------------------
# Worker functions for parallel execution
# ---------------------------------------------------------------------------


def _worker_extract_metadata(filepath: str) -> tuple[str, dict[str, str | None]]:
    """Worker: extract PDF metadata for a single file."""
    return filepath, extract_pdf_metadata(filepath)


def _worker_hash_file(filepath: str) -> tuple[str, str]:
    """Worker: compute SHA256 for a single file."""
    return filepath, compute_sha256(filepath)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(input_dir: Path, output_dir: Path, workers: int = 8) -> None:
    """Execute the full catalog pipeline."""
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Collect all PDF files
    # -----------------------------------------------------------------------
    print(f"[1/8] Scanning {input_dir} for PDFs...")
    pdf_files: list[tuple[str, float]] = []
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() == ".pdf" and p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            pdf_files.append((str(p), size_mb))
    print(f"  Found {len(pdf_files)} PDFs ({sum(s for _, s in pdf_files):.0f} MB total)")

    # -----------------------------------------------------------------------
    # Step 2: Extract PDF metadata (parallel)
    # -----------------------------------------------------------------------
    print(f"[2/8] Extracting PDF metadata ({workers} workers)...")
    t2 = time.time()
    pdf_meta_map: dict[str, dict[str, str | None]] = {}
    errors = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker_extract_metadata, fp): fp for fp, _ in pdf_files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0:
                print(f"  ... {done}/{len(pdf_files)} metadata extracted")
            try:
                fp, meta = fut.result()
                pdf_meta_map[fp] = meta
            except Exception:
                errors += 1
                pdf_meta_map[futures[fut]] = {
                    "title": None,
                    "author": None,
                    "subject": None,
                    "creator": None,
                    "producer": None,
                    "pages": None,
                }
    print(f"  Metadata extracted in {time.time() - t2:.1f}s ({errors} errors)")

    # -----------------------------------------------------------------------
    # Step 3: Compute SHA256 hashes (parallel)
    # -----------------------------------------------------------------------
    print(f"[3/8] Computing SHA256 hashes ({workers} workers)...")
    t3 = time.time()
    hash_map: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker_hash_file, fp): fp for fp, _ in pdf_files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0:
                print(f"  ... {done}/{len(pdf_files)} hashed")
            try:
                fp, h = fut.result()
                hash_map[fp] = h
            except Exception:
                hash_map[futures[fut]] = "ERROR"
    print(f"  Hashing completed in {time.time() - t3:.1f}s")

    # -----------------------------------------------------------------------
    # Step 4: Build catalog entries
    # -----------------------------------------------------------------------
    print("[4/8] Parsing filenames and merging metadata...")
    entries: list[dict[str, Any]] = []
    for filepath, size_mb in pdf_files:
        filename = Path(filepath).name
        pdf_meta = pdf_meta_map.get(filepath, {})

        # Parse filename
        fn_parsed = parse_filename(filename)

        # Merge: prefer PDF metadata when filename is cryptic or noisy
        title = fn_parsed["title"] or "Unknown"
        title_source = "filename"
        pdf_title = pdf_meta.get("title")
        if pdf_title and len(pdf_title) > 5:
            pdf_title_clean = pdf_title.strip()
            # Reject generic/useless PDF titles
            _generic_pdf_titles = {
                "untitled",
                "microsoft word",
                "doc000",
                "print.indd",
                "djvu document",
                "djvu postscript",
                "unknown",
            }
            is_generic_pdf = any(g in pdf_title_clean.lower() for g in _generic_pdf_titles)

            if not is_generic_pdf:
                # Use PDF title if filename is cryptic/hash-like/too short
                if len(title) < 10 or re.match(r"^[a-f0-9]+$", title.lower()) or title == "Unknown":
                    title = pdf_title_clean
                    title_source = "pdf_metadata"
                # Prefer PDF title when it's much cleaner (shorter, no author bleed)
                elif len(pdf_title_clean) < len(title) * 0.7 and len(pdf_title_clean) > 10:
                    title = pdf_title_clean
                    title_source = "pdf_metadata"
                else:
                    title_source = "merged"  # both available, kept filename

        # Authors: merge
        authors = fn_parsed["authors"]
        pdf_author = pdf_meta.get("author")
        if not authors and pdf_author:
            authors = _parse_authors(pdf_author)
            if authors:
                title_source = "pdf_metadata" if title_source == "pdf_metadata" else "merged"

        # Classify domain
        domain, confidence = classify_domain(title, filename, pdf_meta)

        # Book vs non-book
        is_book, is_book_reason = classify_is_book(filename, size_mb, pdf_meta)

        # Flags
        flags: list[str] = []
        if "meap" in filename.lower() or "MEAP" in filename:
            flags.append("meap_version")
        if size_mb > 200:
            flags.append("oversized")
        if "draft" in filename.lower():
            flags.append("draft")

        # Publisher from filename or PDF
        publisher = fn_parsed.get("publisher")
        if not publisher and pdf_meta.get("producer"):
            # Extract publisher from PDF producer if it matches known publishers
            prod = pdf_meta["producer"].lower()
            for p in KNOWN_PUBLISHERS_TIER1 | KNOWN_PUBLISHERS_TIER2:
                if p in prod:
                    publisher = p.title()
                    break

        entry = {
            "filename": filename,
            "full_path": filepath,
            "file_size_mb": round(size_mb, 2),
            "sha256": hash_map.get(filepath, "ERROR"),
            "domain": domain,
            "confidence": confidence,
            "is_book": is_book,
            "is_book_reason": is_book_reason,
            "flags": flags,
            "priority_score": 0,
            "priority_reasons": [],
            "title": title,
            "title_source": title_source,
            "authors": authors,
            "year": fn_parsed.get("year"),
            "series": fn_parsed.get("series"),
            "edition": fn_parsed.get("edition"),
            "publisher": publisher,
            "pdf_metadata": pdf_meta,
            "duplicate_group": None,
            "_is_preferred_duplicate": True,
        }
        entries.append(entry)

    # -----------------------------------------------------------------------
    # Step 5: Detect duplicates
    # -----------------------------------------------------------------------
    print("[5/8] Detecting duplicates...")
    detect_duplicates(entries)
    dup_count = sum(1 for e in entries if e.get("duplicate_group"))
    dup_groups = len({e["duplicate_group"] for e in entries if e.get("duplicate_group")})
    print(f"  Found {dup_count} files in {dup_groups} duplicate groups")

    # -----------------------------------------------------------------------
    # Step 6: Compute priority scores
    # -----------------------------------------------------------------------
    print("[6/8] Computing priority scores...")
    for entry in entries:
        is_pref = entry.pop("_is_preferred_duplicate", True)
        score, reasons = compute_priority_score(entry, is_preferred_duplicate=is_pref)
        entry["priority_score"] = score
        entry["priority_reasons"] = reasons

    # -----------------------------------------------------------------------
    # Step 7: Split books vs non-books
    # -----------------------------------------------------------------------
    print("[7/8] Splitting books vs non-books...")
    books = [e for e in entries if e["is_book"]]
    other = [e for e in entries if not e["is_book"]]
    books.sort(key=lambda e: e["priority_score"], reverse=True)
    other.sort(key=lambda e: e["filename"])
    print(f"  Books: {len(books)}  |  Other: {len(other)}")

    # -----------------------------------------------------------------------
    # Step 8: Write output files
    # -----------------------------------------------------------------------
    print("[8/8] Writing output files...")

    # JSON files
    _write_json(output_dir / "catalog_books.json", books)
    _write_json(output_dir / "catalog_other.json", other)

    # CSV files
    csv_fields = [
        "filename",
        "size_mb",
        "domain",
        "confidence",
        "is_book",
        "title",
        "authors_str",
        "year",
        "priority_score",
        "duplicate_group",
        "flags_str",
        "sha256",
        "pages",
    ]
    _write_csv(output_dir / "catalog_books.csv", books, csv_fields)
    _write_csv(output_dir / "catalog_other.csv", other, csv_fields)

    # Summary
    summary = _build_summary(entries, books, other, time.time() - t_start)
    _write_json(output_dir / "catalog_summary.json", summary)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Books:   {len(books):,} entries → {output_dir / 'catalog_books.json'}")
    print(f"  Other:   {len(other):,} entries → {output_dir / 'catalog_other.json'}")
    print(f"  Summary: {output_dir / 'catalog_summary.json'}")

    # Print domain distribution
    print("\n--- Domain Distribution (Books) ---")
    domain_counts = Counter(e["domain"] for e in books)
    for domain, count in domain_counts.most_common():
        print(f"  {domain:30s}  {count:5d}")

    # Print top-10 by priority
    print("\n--- Top 10 by Priority Score ---")
    for e in books[:10]:
        authors_str = ", ".join(e["authors"][:2]) if e["authors"] else "?"
        print(f"  [{e['priority_score']:3d}] {e['title'][:60]}  ({authors_str})")

    # Print duplicate summary
    print(f"\n--- Duplicates ---")
    print(f"  {dup_count} files in {dup_groups} groups")


def _write_json(path: Path, data: Any) -> None:
    """Write JSON file with pretty formatting."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _write_csv(path: Path, entries: list[dict[str, Any]], fields: list[str]) -> None:
    """Write CSV file from catalog entries."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            row = {
                "filename": entry["filename"],
                "size_mb": f"{entry['file_size_mb']:.1f}",
                "domain": entry["domain"],
                "confidence": entry["confidence"],
                "is_book": str(entry["is_book"]),
                "title": entry.get("title") or "",
                "authors_str": "; ".join(entry.get("authors") or []),
                "year": str(entry["year"]) if entry.get("year") else "",
                "priority_score": str(entry["priority_score"]),
                "duplicate_group": entry.get("duplicate_group") or "",
                "flags_str": ", ".join(entry.get("flags") or []),
                "sha256": entry.get("sha256") or "",
                "pages": entry.get("pdf_metadata", {}).get("pages") or "",
            }
            writer.writerow(row)


def _build_summary(
    all_entries: list[dict[str, Any]],
    books: list[dict[str, Any]],
    other: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    """Build summary statistics."""
    domain_counts = Counter(e["domain"] for e in books)
    confidence_counts = Counter(e["confidence"] for e in books)
    dup_groups = {e["duplicate_group"] for e in all_entries if e.get("duplicate_group")}
    dup_files = sum(1 for e in all_entries if e.get("duplicate_group"))

    priority_scores = [e["priority_score"] for e in books]
    priority_buckets = {
        "80+": sum(1 for s in priority_scores if s >= 80),
        "60-79": sum(1 for s in priority_scores if 60 <= s < 80),
        "40-59": sum(1 for s in priority_scores if 40 <= s < 60),
        "20-39": sum(1 for s in priority_scores if 20 <= s < 40),
        "0-19": sum(1 for s in priority_scores if s < 20),
    }

    return {
        "total_pdfs": len(all_entries),
        "total_books": len(books),
        "total_other": len(other),
        "total_size_gb": round(sum(e["file_size_mb"] for e in all_entries) / 1024, 2),
        "books_size_gb": round(sum(e["file_size_mb"] for e in books) / 1024, 2),
        "domain_distribution": dict(domain_counts.most_common()),
        "confidence_distribution": dict(confidence_counts),
        "duplicate_groups": len(dup_groups),
        "duplicate_files": dup_files,
        "priority_distribution": priority_buckets,
        "elapsed_seconds": round(elapsed, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Catalog and classify a library of PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fixtures/library_catalog"),
        help="Output directory for catalog files (default: fixtures/library_catalog/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for hashing/metadata (default: 8)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    run_pipeline(args.input_dir, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
