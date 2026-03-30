#!/usr/bin/env python3
"""Classify uningested library_books PDFs by domain.

Pipeline (order matters — each stage filters before the next):
  1. HASH DEDUP: Skip files already in DB or seen before
  2. TEXT CHECK: Extract first 3 pages via pdftotext. No text = "needs_ocr" bucket
  3. JUNK FILTER: Reject by filename patterns (ISO specs, product brochures, etc.)
  4. AUTHORSHIP FILTER: Reject Brandon Behring's own work
  5. BOOK CHECK: Reject <50 pages (papers, slides, not textbooks)
  6. MEAP DEDUP: Keep only latest version of Manning MEAP books
  7. DOMAIN CLASSIFY: Keyword match on title + first-page text
  8. ORGANIZE: Move classified books into domain subdirectories + write sidecars

Usage:
    python scripts/classify_library_books.py --dry-run          # Show counts per stage
    python scripts/classify_library_books.py --organize         # Move into domain dirs + sidecars
    python scripts/classify_library_books.py --output out.csv   # CSV for manual review
"""

import argparse
import asyncio
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))


# ============================================================
# Stage 3: JUNK FILTER — filename patterns that are never books
# ============================================================
JUNK_FILENAME_PATTERNS = [
    r"^iso[\s._-]?\d+",  # ISO specification documents
    r"^DLD_tile",  # Overleaf image artifacts
    r"^0_\d{10,}",  # Scanner output (numeric IDs)
    r"^<[0-9A-Fa-f]+>",  # Hex-named files
    r"^BTH\d+",  # Product brochures
    r"guitar|flamenco",  # Music (not CS/math)
    r"german.grammar",  # Language learning
    r"home.inspect|subaru|impreza",  # Personal documents
    r"consumer.report",  # Magazine
    r"eps-converted",  # Image conversion artifacts
    r"conflicted.copy",  # Sync conflict duplicates
    r"homeowner|brochure",  # Marketing materials
    r"^Formula.Sheet",  # Cheat sheets
    r"^Lecture_\d+",  # Lecture slides (authored)
    r"^\d{1,3}\.pdf$",  # Bare numbers
    r"Notebook.PDF",  # Jupyter notebook exports
]

# ============================================================
# Stage 4: AUTHORSHIP — patterns indicating Brandon's own work
# ============================================================
AUTHOR_FILENAME_PATTERNS = [
    r"^Brandon",
    r"^brandon",
    r"^main \d+",  # Research paper drafts ("main 11.pdf" etc.)
    r"^main\.pdf$",
    r"Proposal",  # Grant proposals
    r"Research.Statement",
    r"Teaching.Statement",
    r"escape_of_vortex",
    r"Leap.Frog",
    r"project\d{4}nsf",  # NSF proposals
]

AUTHOR_TEXT_PATTERNS = [
    "brandon behring",
    "brandon m. behring",
    "lever_of_archimedes",
    "archimedes_lever",
    "lever of archimedes",
]

# ============================================================
# Stage 7: DOMAIN CLASSIFICATION — keyword rules
# Matched against TITLE + FIRST PAGE TEXT (not filename).
# More specific terms first to avoid false positives.
# ============================================================
DOMAIN_RULES = [
    # --- Skip: not primary textbooks ---
    (
        "SKIP",
        ["homework", "solutions manual", "exam answer", "quiz solution", "student chapter report"],
    ),
    # --- High-specificity domains ---
    (
        "causal_inference",
        [
            "causal inference",
            "treatment effect",
            "instrumental variable",
            "counterfactual",
            "propensity score",
            "average treatment effect",
            "regression discontinuity",
        ],
    ),
    (
        "econometrics",
        [
            "econometrics",
            "panel data",
            "heteroskedasticity",
            "endogeneity",
            "two.stage least squares",
        ],
    ),
    (
        "reinforcement_learning",
        [
            "reinforcement learning",
            "markov decision process",
            "q.learning",
            "policy gradient",
            "temporal difference",
        ],
    ),
    (
        "deep_learning",
        [
            "deep learning",
            "convolutional neural",
            "recurrent neural",
            "generative adversarial",
            "backpropagation",
            "neural network architecture",
        ],
    ),
    (
        "machine_learning",
        [
            "machine learning",
            "supervised learning",
            "unsupervised learning",
            "gradient descent",
            "support vector",
            "random forest",
            "cross.validation",
            "bias.variance",
        ],
    ),
    (
        "rag_llm",
        [
            "language model",
            "retrieval.augmented",
            "information retrieval",
            "natural language processing",
            "word embedding",
            "large language model",
            "prompt engineering",
        ],
    ),
    # --- Quantitative ---
    (
        "time_series",
        [
            "time series analysis",
            "time series forecast",
            "arima",
            "garch",
            "autoregressive",
            "state space model",
        ],
    ),
    (
        "statistics",
        [
            "mathematical statistics",
            "statistical inference",
            "hypothesis testing",
            "confidence interval",
            "maximum likelihood",
            "bootstrap",
            "bayesian statistics",
            "probability distribution",
            "regression analysis",
        ],
    ),
    (
        "finance",
        [
            "financial",
            "portfolio",
            "option pricing",
            "black.scholes",
            "risk management",
            "actuarial",
            "insurance",
            "annuity",
            "cfa program",
            "cfa curriculum",
            "derivatives",
            "fixed income",
            "credit risk",
            "asset pricing",
            "valuation",
        ],
    ),
    # --- CS/Engineering ---
    (
        "software_engineering",
        [
            "programming language",
            "software engineering",
            "design patterns",
            "clean code",
            "refactoring",
            "web development",
            "microservice",
            "docker",
            "kubernetes",
            "cloud computing",
            "aws certified",
            "devops",
            "api design",
        ],
    ),
    (
        "algorithms",
        [
            "algorithm design",
            "computational complexity",
            "graph algorithm",
            "data structures and algorithms",
            "dynamic programming",
        ],
    ),
    (
        "sql",
        ["sql ", "database systems", "relational database", "query optimization", "apache spark"],
    ),
    (
        "data_science",
        [
            "data science",
            "data analysis",
            "pandas",
            "data visualization",
            "exploratory data",
            "feature engineering",
        ],
    ),
    (
        "ml_engineering",
        ["mlops", "ml system", "model deployment", "production ml", "ml pipeline", "model serving"],
    ),
    # --- Science (BEFORE math — more specific) ---
    (
        "biology_neuroscience",
        ["neuroscience", "brain", "genetics", "genomics", "molecular biology", "infant brain"],
    ),
    (
        "physics",
        [
            "quantum mechanics",
            "classical mechanics",
            "thermodynamics",
            "electrodynamics",
            "optics",
            "general relativity",
            "particle physics",
            "condensed matter",
            "astrophysics",
            "cosmology",
            "astronomy",
            "statistical mechanics",
            "quantum field theory",
            "feynman",
            "sakurai",
            "griffiths electrodynamics",
            "conformal field theory",
            "nonlinear optics",
        ],
    ),
    (
        "signal_processing",
        [
            "signal processing",
            "fourier transform",
            "wavelet",
            "spectral analysis",
            "discrete fourier",
        ],
    ),
    # --- Pure math (specific → general ordering) ---
    # Topology/geometry FIRST — catches homotopy, manifold before algebra grabs them
    (
        "topology_geometry",
        [
            "algebraic topology",
            "differential geometry",
            "topology",
            "manifold",
            "homology",
            "cohomology",
            "riemannian",
            "homotopy theory",
            "homotopy type",
            "hyperbolic manifold",
            "fiber bundle",
            "knot theory",
            "geometric group",
        ],
    ),
    # Analysis BEFORE algebra — catches Sobolev/Banach/Hilbert before "algebra" in series names
    (
        "analysis",
        [
            "real analysis",
            "complex analysis",
            "functional analysis",
            "measure theory",
            "harmonic analysis",
            "lebesgue",
            "hilbert space",
            "banach space",
            "sobolev space",
            "operator theory",
            "semigroup",
            "spectral theory",
        ],
    ),
    # Statistics — catches probability before algebra grabs "probability" books
    (
        "probability_theory",
        [
            "probability theory",
            "probability and measure",
            "stochastic",
            "random variable",
            "martingale",
            "probability distribution",
        ],
    ),
    # Dynamical systems
    (
        "dynamical_systems",
        [
            "dynamical systems",
            "chaos theory",
            "bifurcation theory",
            "ordinary differential equations",
            "partial differential equations",
            "nonlinear dynamics",
            "ergodic theory",
            "hamiltonian system",
            "lyapunov",
            "strange attractor",
            "celestial mechanics",
            "phase space",
            "differential equation",
        ],
    ),
    # Numerical
    (
        "numerical_methods",
        [
            "numerical methods",
            "finite element",
            "finite difference",
            "numerical analysis",
            "pseudospectral",
            "computational mathematics",
        ],
    ),
    # Optimization
    (
        "optimization",
        [
            "convex optimization",
            "linear programming",
            "variational method",
            "optimal control",
            "optimization algorithm",
        ],
    ),
    # Algebra AFTER topology/analysis/probability — avoids false positives
    (
        "algebra",
        [
            "abstract algebra",
            "linear algebra",
            "group theory",
            "ring theory",
            "galois theory",
            "commutative algebra",
            "module theory",
            "representation theory",
        ],
    ),
    # General math (last resort for math books)
    (
        "mathematics",
        [
            "number theory",
            "mathematical logic",
            "discrete mathematics",
            "advanced engineering mathematics",
            "mathematical methods",
            "mathematical physics",
        ],
    ),
    # Calculus textbooks — only match "calculus" in the TITLE, not body text
    (
        "analysis",
        [
            "^calculus",
            "thomas.* calculus",
            "stewart.* calculus",
            "calculus.*early transcendental",
            "calculus.*several variable",
        ],
    ),
    # --- Other ---
    (
        "functional_programming",
        ["haskell", "functional programming", "category theory", "monad", "lambda calculus"],
    ),
    (
        "fitness",
        [
            "strength training",
            "bodybuilding",
            "nutrition",
            "muscle",
            "workout",
            "exercise science",
            "flexibility",
        ],
    ),
    ("recommender_systems", ["recommender system", "collaborative filtering"]),
    ("economics", ["microeconomics", "macroeconomics", "game theory"]),
]


# ============================================================
# Utility functions
# ============================================================


def compute_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def extract_text(path: Path, pages: int = 3) -> str:
    """Extract text from first N pages. Returns empty string on failure."""
    try:
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", str(pages), str(path), "-"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout[:3000]
    except (subprocess.TimeoutExpired, Exception):
        return ""


def extract_metadata(path: Path) -> dict:
    """Extract PDF metadata via pdfinfo."""
    info = {"title": "", "author": "", "pages": 0}
    try:
        result = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            if line.startswith("Title:"):
                info["title"] = line.split(":", 1)[1].strip()
            elif line.startswith("Author:"):
                info["author"] = line.split(":", 1)[1].strip()
            elif line.startswith("Pages:"):
                try:
                    info["pages"] = int(line.split(":")[1].strip())
                except ValueError:
                    pass
    except (subprocess.TimeoutExpired, Exception):
        pass
    return info


def best_title(filename: str, pdf_title: str) -> str:
    """Get best available title."""
    if pdf_title and len(pdf_title) > 5 and pdf_title.lower() not in ("untitled", "microsoft word"):
        return pdf_title[:120]
    name = filename.replace(".pdf", "").replace("_", " ")
    name = re.sub(r"^\(.*?\)\s*", "", name)  # Remove publisher series prefix
    return name[:120]


def is_junk_filename(filename: str) -> bool:
    """Stage 3: Detect non-book files by filename."""
    for pattern in JUNK_FILENAME_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return True
    return False


def is_your_work(filename: str, text: str) -> bool:
    """Stage 4: Detect Brandon's authored content."""
    for pattern in AUTHOR_FILENAME_PATTERNS:
        if re.search(pattern, filename):
            return True
    text_lower = text.lower()
    for pattern in AUTHOR_TEXT_PATTERNS:
        if pattern in text_lower:
            return True
    return False


def classify_domain(title: str, text: str) -> str:
    """Stage 7: Classify into domain by keyword matching.

    Strategy: Match against TITLE ONLY first (two passes).
    Body text is too noisy — introductory pages mention concepts from
    many domains, causing false positives.

    Pass 1: Try all rules against title only.
    Pass 2: If no match, try rules against title + first 500 chars of body.
    This keeps specificity high while still catching books with bad titles.
    """
    title_lower = title.lower()

    # Pass 1: Title only (high precision)
    for domain_id, keywords in DOMAIN_RULES:
        for kw in keywords:
            clean_kw = kw.lstrip("^")
            if re.search(clean_kw, title_lower):
                return domain_id

    # Pass 2 DISABLED — body text causes too many false positives.
    # Books with unclear titles stay "uncategorized" (safe default).
    # These can be manually reviewed or classified by LLM in a future pass.

    return "uncategorized"


# ============================================================
# Main pipeline
# ============================================================


async def main():
    parser = argparse.ArgumentParser(description="Classify library_books PDFs by domain")
    parser.add_argument("--dry-run", action="store_true", help="Show stage counts only")
    parser.add_argument(
        "--organize", action="store_true", help="Move into domain dirs + write sidecar JSONs"
    )
    parser.add_argument(
        "--min-size-mb", type=float, default=1.0, help="Minimum PDF size in MB (default: 1)"
    )
    parser.add_argument(
        "--min-pages",
        type=int,
        default=50,
        help="Minimum pages to be considered a book (default: 50)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output CSV path (default: stdout)"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Directory to scan (default: fixtures/library_books)",
    )
    args = parser.parse_args()

    if args.directory:
        lib_dir = Path(args.directory)
    else:
        lib_dir = Path(__file__).parent.parent / "fixtures" / "library_books"

    # Stage 1: HASH DEDUP — get existing hashes from DB
    from research_kb_storage import DatabaseConfig, get_connection_pool, close_connection_pool

    pool = await get_connection_pool(DatabaseConfig())
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT file_hash, title FROM sources WHERE file_hash IS NOT NULL")
        existing_hashes = {r["file_hash"] for r in rows}
        existing_titles = {re.sub(r"[^a-z0-9]", "", r["title"].lower())[:25] for r in rows}
    await close_connection_pool()

    # Scan all PDFs
    min_bytes = int(args.min_size_mb * 1_000_000)
    all_pdfs = sorted([p for p in lib_dir.glob("*.pdf") if p.stat().st_size > min_bytes])
    print(f"Scanning {len(all_pdfs)} PDFs (>{args.min_size_mb}MB)...", file=sys.stderr)

    # Pipeline counters
    stats = {
        "total": len(all_pdfs),
        "already_in_db": 0,
        "duplicate_hash": 0,
        "no_text": 0,
        "junk_filename": 0,
        "your_work": 0,
        "too_short": 0,
        "meap_dedup": 0,
        "duplicate_title": 0,
        "classified": 0,
        "uncategorized": 0,
        "skip": 0,
    }

    seen_hashes = set()
    seen_titles = {}  # norm_title -> entry (for MEAP dedup)
    results = []

    for i, pdf in enumerate(all_pdfs):
        # --- Stage 1: HASH DEDUP ---
        file_hash = compute_hash(pdf)
        if file_hash in existing_hashes:
            stats["already_in_db"] += 1
            continue
        if file_hash in seen_hashes:
            stats["duplicate_hash"] += 1
            continue
        seen_hashes.add(file_hash)

        # --- Stage 2: TEXT CHECK ---
        text = extract_text(pdf)
        has_text = len(text.strip()) > 50
        if not has_text:
            stats["no_text"] += 1
            continue  # Skip scanned PDFs without text

        # --- Stage 3: JUNK FILTER ---
        if is_junk_filename(pdf.name):
            stats["junk_filename"] += 1
            continue

        # --- Stage 4: AUTHORSHIP FILTER ---
        if is_your_work(pdf.name, text):
            stats["your_work"] += 1
            continue

        # --- Stage 5: BOOK CHECK ---
        meta = extract_metadata(pdf)
        pages = meta["pages"]
        if 0 < pages < args.min_pages:
            stats["too_short"] += 1
            continue

        # --- Stage 6: MEAP DEDUP ---
        title = best_title(pdf.name, meta["title"])
        norm_title = re.sub(r"\s*MEAP\s*V\d+", "", title, flags=re.IGNORECASE).strip()
        norm_title = re.sub(r"\s*\(\d+\)$", "", norm_title).strip()
        norm_key = re.sub(r"[^a-z0-9]", "", norm_title.lower())[:30]

        # Also check against DB titles
        if norm_key in existing_titles:
            stats["duplicate_title"] += 1
            continue

        if norm_key in seen_titles:
            prev = seen_titles[norm_key]
            size_mb = pdf.stat().st_size / 1_000_000
            if size_mb > prev["size_mb"]:
                seen_titles[norm_key] = {
                    "path": str(pdf),
                    "title": title,
                    "size_mb": size_mb,
                    "pages": pages,
                    "author": meta["author"],
                    "file_hash": file_hash,
                    "text": text,
                }
            stats["meap_dedup"] += 1
            continue
        seen_titles[norm_key] = {
            "path": str(pdf),
            "title": title,
            "size_mb": round(pdf.stat().st_size / 1_000_000, 1),
            "pages": pages,
            "author": meta["author"],
            "file_hash": file_hash,
            "text": text,
        }

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(all_pdfs)}...", file=sys.stderr)

    # --- Stage 7: DOMAIN CLASSIFY ---
    for norm_key, entry in seen_titles.items():
        domain = classify_domain(entry["title"], entry["text"])
        entry["domain"] = domain
        if domain == "SKIP":
            stats["skip"] += 1
        elif domain == "uncategorized":
            stats["uncategorized"] += 1
        else:
            stats["classified"] += 1
        results.append(entry)

    # --- Report ---
    print(f"\nPIPELINE RESULTS:", file=sys.stderr)
    print(f"  Total PDFs scanned:     {stats['total']:5d}", file=sys.stderr)
    print(f"  Already in DB:          {stats['already_in_db']:5d}", file=sys.stderr)
    print(f"  Duplicate hash:         {stats['duplicate_hash']:5d}", file=sys.stderr)
    print(f"  No extractable text:    {stats['no_text']:5d}", file=sys.stderr)
    print(f"  Junk filename:          {stats['junk_filename']:5d}", file=sys.stderr)
    print(f"  Your work:              {stats['your_work']:5d}", file=sys.stderr)
    print(f"  Too short (<{args.min_pages}pp):    {stats['too_short']:5d}", file=sys.stderr)
    print(f"  MEAP dedup:             {stats['meap_dedup']:5d}", file=sys.stderr)
    print(f"  Duplicate title in DB:  {stats['duplicate_title']:5d}", file=sys.stderr)
    print(f"  ─────────────────────────────", file=sys.stderr)
    print(f"  Classified:             {stats['classified']:5d}", file=sys.stderr)
    print(f"  Uncategorized:          {stats['uncategorized']:5d}", file=sys.stderr)
    print(f"  Skip (not textbook):    {stats['skip']:5d}", file=sys.stderr)

    from collections import Counter

    domain_counts = Counter(
        r["domain"] for r in results if r["domain"] not in ("SKIP", "uncategorized")
    )
    print(f"\nDomain distribution:", file=sys.stderr)
    for domain, count in domain_counts.most_common():
        print(f"  {domain:30s} {count:5d}", file=sys.stderr)

    if args.dry_run:
        return

    # --- Output CSV ---
    classified = [r for r in results if r["domain"] not in ("SKIP", "uncategorized")]
    classified.sort(key=lambda x: (x["domain"], -x["pages"]))

    if args.output or not args.organize:
        out = open(args.output, "w", newline="") if args.output else sys.stdout
        writer = csv.DictWriter(
            out, fieldnames=["domain", "title", "author", "pages", "size_mb", "path"]
        )
        writer.writeheader()
        for r in classified:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
        if args.output:
            out.close()
            print(f"Wrote {args.output}", file=sys.stderr)

    # --- Organize into domain dirs ---
    if args.organize:
        moved = Counter()
        for r in classified:
            domain = r["domain"]
            pdf = Path(r["path"])
            if not pdf.exists():
                continue

            dest_dir = lib_dir / domain
            dest_dir.mkdir(exist_ok=True)

            dest_pdf = dest_dir / pdf.name
            dest_sidecar = dest_dir / pdf.name.replace(".pdf", ".json")

            if not dest_pdf.exists():
                pdf.rename(dest_pdf)

            sidecar_data = {
                "title": r["title"],
                "authors": [r["author"]] if r["author"] else [],
                "domain_id": domain,
                "source_type": "textbook",
                "classifier": "classify_library_books.py",
            }
            dest_sidecar.write_text(json.dumps(sidecar_data, indent=2))
            moved[domain] += 1

        print(f"\nOrganized {sum(moved.values())} books:", file=sys.stderr)
        for d, c in moved.most_common():
            print(f"  {d:30s} {c:5d}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
