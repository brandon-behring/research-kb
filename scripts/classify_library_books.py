#!/usr/bin/env python3
"""Classify uningested library_books PDFs by domain using pdfinfo + pdftotext.

Extracts title, author, and first page text from each PDF, then classifies
by keyword matching. Outputs a CSV for review before ingestion.

Three-phase workflow:
  1. python scripts/classify_library_books.py > library_classification.csv
  2. Review CSV, edit domain assignments, mark SKIP rows
  3. python scripts/ingest_missing_textbooks.py  (reads sidecars created by this script)

Usage:
    python scripts/classify_library_books.py                    # Full run
    python scripts/classify_library_books.py --dry-run          # Show counts only
    python scripts/classify_library_books.py --create-sidecars  # Write .json sidecars
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


# Domain classification rules: (domain_id, [keywords])
# Order matters — first match wins
DOMAIN_RULES = [
    ("SKIP", ["homework", "solution manual", "exam ", "quiz", "lecture note",
              "mueller", "covid", "consumer report", "cheat sheet",
              "home.inspect", "subaru", "impreza", "multimedia.system.manual",
              "eps.converted", "tile.zoom", "201.home", "car manual",
              "owner.s manual", "inspection report", "bike repair",
              "pain free life", "student chapter report", "manuscript",
              "board of advisory"]),
    ("machine_learning", ["machine learn", "deep learn", "neural net", "tensorflow",
                          "pytorch", "transformer", "gradient descent", "backpropagation",
                          "classification", "supervised learn", "unsupervised learn"]),
    ("reinforcement_learning", ["reinforcement learn", "markov decision", "q-learning",
                                 "policy gradient", "multi-armed bandit"]),
    ("deep_learning", ["convolutional", "recurrent neural", "generative adversarial",
                       "attention mechanism", "bert", "gpt"]),
    ("rag_llm", ["language model", "retrieval augmented", "information retrieval",
                 "text mining", "nlp", "natural language process", "embedding",
                 "semantic search"]),
    ("causal_inference", ["causal", "treatment effect", "instrumental variable",
                          "counterfactual", "propensity score", "difference.in.difference"]),
    ("econometrics", ["econometr", "panel data", "heteroskedast", "endogen"]),
    ("statistics", ["statistic", "probability", "bayesian", "regression", "hypothesis test",
                    "confidence interval", "maximum likelihood", "bootstrap",
                    "monte carlo", "sampling"]),
    ("time_series", ["time series", "forecast", "arima", "garch", "stochastic process",
                     "autoregressive"]),
    ("finance", ["finance", "portfolio", "option pricing", "black.scholes", "risk manage",
                 "actuari", "insurance", "annuit", "cfa ", "derivatives", "fixed income",
                 "credit risk", "asset pricing"]),
    ("software_engineering", ["python", "java ", "javascript", "programming", "software",
                              "algorithm", "data structure", "docker", "devops", "git ",
                              "testing", "design pattern", "refactor", "clean code",
                              "microservice", "api design", "web develop", "aws certif",
                              "cloud computing", "kubernetes", "terraform"]),
    ("algorithms", ["algorithm", "complexity", "graph theory", "combinatori",
                    "data structure", "sorting", "dynamic programming"]),
    ("sql", ["sql", "database", "data engineer", "postgresql", "query optim"]),
    ("data_science", ["data science", "data analy", "pandas", "visualization",
                      "exploratory data", "feature engineer"]),
    ("ml_engineering", ["mlops", "ml system", "model deploy", "production ml",
                        "ml pipeline", "feature store"]),
    ("physics", ["quantum", "physics", "mechanic", "thermodynamic", "electrodynamic",
                 "optic", "relativity", "particle", "field theory", "condensed matter",
                 "astrophys", "cosmolog", "astrono", "orbit", "kepler", "gravitat",
                 "celestial", "planetary", "n.body", "vortex", "fluid dynamic"]),
    ("algebra", ["algebra", "linear algebra", "matrix", "tensor", "group theory",
                 "ring ", "galois", "representation theory"]),
    ("linear_algebra", ["linear algebra", "matrix decomp", "eigenvalue", "singular value"]),
    ("analysis", ["calculus", "real analysis", "complex analysis", "measure theory",
                  "functional analysis", "harmonic analysis", "lebesgue",
                  "hilbert space", "banach space", "applied math",
                  "mathematical method", "schaum"]),
    ("mathematics", ["discrete math", "number theory", "abstract algebra",
                     "mathematical logic", "set theory", "combinatoric"]),
    ("topology_geometry", ["topology", "geometry", "manifold", "differential geom",
                           "algebraic topology", "homology", "cohomology", "riemannian"]),
    ("dynamical_systems", ["dynamical system", "chaos", "bifurcation", "ode", "pde",
                           "nonlinear", "ergodic", "hamiltonian", "celestial mechanic",
                           "stability", "lyapunov", "three.body", "n.body",
                           "ordinary differential eq", "partial differential eq",
                           "differential equation", "perturbation", "wave motion",
                           "almost periodic"]),
    ("numerical_methods", ["numerical", "finite element", "finite difference",
                           "computational", "simulation", "approximation theory"]),
    ("optimization", ["optimi", "convex", "linear program", "variational",
                      "optimal control"]),
    ("biology_neuroscience", ["biology", "neurosci", "brain", "genetics", "genomic",
                              "bioinformatic", "evolution"]),
    ("signal_processing", ["signal process", "fourier", "wavelet", "spectral",
                           "digital signal"]),
    ("functional_programming", ["haskell", "scala", "functional program", "category theory",
                                "monad", "lambda calculus"]),
    ("fitness", ["fitness", "exercise", "strength train", "bodybuilding", "nutrition",
                 "muscle", "workout", "supple leopard", "stretching", "mobility"]),
    ("recommender_systems", ["recommender", "collaborative filter", "matrix factorization"]),
    ("economics", ["economics", "microecon", "macroecon", "game theory", "auction"]),
]


def pdfinfo(path: Path) -> dict:
    """Extract metadata via pdfinfo (poppler). ~30ms per PDF."""
    try:
        result = subprocess.run(
            ["pdfinfo", str(path)], capture_output=True, text=True, timeout=10
        )
        info = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                info[key.strip()] = value.strip()
        return info
    except (subprocess.TimeoutExpired, Exception):
        return {}


def pdftotext_first_page(path: Path) -> str:
    """Extract first page text via pdftotext (poppler). ~40ms per PDF."""
    try:
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "2", str(path), "-"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout[:2000]  # First 2000 chars
    except (subprocess.TimeoutExpired, Exception):
        return ""


def classify(title: str, author: str, first_page: str, filename: str) -> str:
    """Classify a PDF into a domain by keyword matching.

    Searches title, author, first page text, and filename.
    Returns domain_id or 'uncategorized'.
    """
    # Combine all text for matching
    text = f"{title} {author} {first_page} {filename}".lower()

    for domain_id, keywords in DOMAIN_RULES:
        for kw in keywords:
            if re.search(kw, text):
                return domain_id

    return "uncategorized"


def clean_title(filename: str, pdf_title: str) -> str:
    """Get best available title from PDF metadata or filename."""
    if pdf_title and len(pdf_title) > 5 and pdf_title.lower() not in ("untitled", "microsoft word"):
        return pdf_title[:120]
    # Fall back to filename
    name = filename.replace(".pdf", "").replace("_", " ")
    # Remove common noise prefixes
    name = re.sub(r"^\(.*?\)\s*", "", name)  # Remove (Publisher Series) prefix
    return name[:120]


def compute_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


async def main():
    parser = argparse.ArgumentParser(description="Classify library_books PDFs by domain")
    parser.add_argument("--dry-run", action="store_true", help="Show counts only")
    parser.add_argument("--create-sidecars", action="store_true",
                        help="Write .json sidecar files for classified books")
    parser.add_argument("--min-size-mb", type=float, default=5.0,
                        help="Minimum PDF size in MB (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    lib_dir = Path(__file__).parent.parent / "fixtures" / "library_books"

    # Get existing hashes from DB
    from research_kb_storage import DatabaseConfig, get_connection_pool, close_connection_pool
    pool = await get_connection_pool(DatabaseConfig())
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT file_hash FROM sources WHERE file_hash IS NOT NULL")
        existing = {r["file_hash"] for r in rows}
    await close_connection_pool()

    # Find uningested PDFs
    min_bytes = int(args.min_size_mb * 1_000_000)
    pdfs = sorted([p for p in lib_dir.glob("*.pdf") if p.stat().st_size > min_bytes])

    print(f"Scanning {len(pdfs)} PDFs (>{args.min_size_mb}MB)...", file=sys.stderr)

    results = []
    for i, pdf in enumerate(pdfs):
        file_hash = compute_hash(pdf)
        if file_hash in existing:
            continue  # Already ingested

        # Extract metadata (~40ms per PDF)
        info = pdfinfo(pdf)
        first_page = pdftotext_first_page(pdf)
        pdf_title = info.get("Title", "")
        author = info.get("Author", "")
        pages = info.get("Pages", "0")

        title = clean_title(pdf.name, pdf_title)
        domain = classify(title, author, first_page, pdf.name)
        size_mb = pdf.stat().st_size / 1_000_000

        results.append({
            "filename": pdf.name,
            "title": title,
            "author": author[:80],
            "pages": pages,
            "size_mb": round(size_mb, 1),
            "domain": domain,
            "file_hash": file_hash,
            "first_words": first_page[:150].replace("\n", " ").strip(),
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(pdfs)}...", file=sys.stderr)

    print(f"Classified {len(results)} uningested PDFs", file=sys.stderr)

    # Summary
    from collections import Counter
    domain_counts = Counter(r["domain"] for r in results)
    print(f"\nDomain distribution:", file=sys.stderr)
    for domain, count in domain_counts.most_common():
        print(f"  {domain:30s} {count:5d}", file=sys.stderr)

    if args.dry_run:
        return

    # Output CSV
    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=["domain", "title", "author", "pages", "size_mb",
                                              "filename", "first_words"])
    writer.writeheader()
    for r in sorted(results, key=lambda x: (x["domain"], x["title"])):
        writer.writerow({k: r[k] for k in writer.fieldnames})
    if args.output:
        out.close()
        print(f"Wrote {args.output}", file=sys.stderr)

    # Create sidecars if requested
    if args.create_sidecars:
        created = 0
        for r in results:
            if r["domain"] in ("SKIP", "uncategorized"):
                continue
            sidecar = lib_dir / r["filename"].replace(".pdf", ".json")
            if sidecar.exists():
                continue
            data = {
                "title": r["title"],
                "authors": [r["author"]] if r["author"] else [],
                "domain_id": r["domain"],
                "source_type": "textbook",
                "classifier": "classify_library_books.py",
            }
            sidecar.write_text(json.dumps(data, indent=2))
            created += 1
        print(f"Created {created} sidecar JSONs", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
