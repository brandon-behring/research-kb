#!/usr/bin/env python3
"""Classify arXiv papers via Semantic Scholar and create sidecar JSONs.

Fetches paper metadata from S2 API, classifies into research-kb domains,
and creates sidecar JSON files for ingestion via ingest_missing_papers.py.

Usage:
    python scripts/classify_arxiv_papers.py \
        --source-dir ~/Claude/lever_of_archimedes/knowledge/sources/papers/arxiv/ \
        --dest-dir fixtures/papers/arxiv/ \
        --dry-run

    python scripts/classify_arxiv_papers.py \
        --source-dir ~/Claude/lever_of_archimedes/knowledge/sources/papers/arxiv/ \
        --dest-dir fixtures/papers/arxiv/
"""

import argparse
import asyncio
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "s2-client" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))

from research_kb_common import configure_logging, get_logger
from s2_client.client import S2Client

logger = get_logger(__name__)

# Domain classification rules based on S2 fields of study + title/abstract keywords
DOMAIN_RULES = [
    # (keywords in title/abstract, s2_fields, assigned_domain)
    # More specific rules first
    (
        {
            "causal inference",
            "causal effect",
            "treatment effect",
            "instrumental variable",
            "double machine learning",
            "propensity score",
            "counterfactual",
            "do-calculus",
            "potential outcomes",
            "average treatment effect",
        },
        None,
        "causal_inference",
    ),
    (
        {
            "reinforcement learning",
            "policy gradient",
            "q-learning",
            "reward shaping",
            "multi-armed bandit",
            "markov decision",
        },
        None,
        "reinforcement_learning",
    ),
    ({"time series", "temporal", "forecasting", "autoregressive"}, None, "time_series"),
    (
        {
            "retrieval augmented",
            "RAG",
            "knowledge graph",
            "vector database",
            "semantic search",
            "document retrieval",
            "embedding model",
        },
        None,
        "rag_llm",
    ),
    (
        {
            "large language model",
            "LLM",
            "GPT",
            "BERT",
            "transformer",
            "prompt engineering",
            "chain-of-thought",
            "instruction tuning",
            "fine-tuning",
            "RLHF",
            "language model",
            "foundation model",
            "text generation",
            "in-context learning",
        },
        None,
        "rag_llm",
    ),
    (
        {
            "attention mechanism",
            "self-attention",
            "neural network architecture",
            "deep neural",
            "convolutional neural",
            "recurrent neural",
            "generative adversarial",
            "variational autoencoder",
            "diffusion model",
            "vision transformer",
        },
        None,
        "deep_learning",
    ),
    (
        {
            "machine learning",
            "classification",
            "regression",
            "random forest",
            "gradient boosting",
            "feature selection",
            "model selection",
            "supervised learning",
            "unsupervised learning",
        },
        None,
        "machine_learning",
    ),
    (
        {
            "econometric",
            "panel data",
            "difference-in-difference",
            "regression discontinuity",
            "synthetic control",
        },
        None,
        "econometrics",
    ),
    (
        {
            "embedding",
            "word2vec",
            "sentence embedding",
            "contrastive learning",
            "representation learning",
        },
        None,
        "deep_learning",
    ),
    (
        {
            "statistical",
            "hypothesis testing",
            "bayesian",
            "posterior",
            "likelihood",
            "nonparametric",
        },
        None,
        "statistics",
    ),
    (
        {"recommender", "collaborative filtering", "matrix factorization"},
        None,
        "recommender_systems",
    ),
    (
        {"software engineering", "code generation", "program synthesis"},
        None,
        "software_engineering",
    ),
]

# Fallback: map S2 fields of study to domains
S2_FIELD_MAP = {
    "Computer Science": "machine_learning",
    "Mathematics": "mathematics",
    "Economics": "econometrics",
    "Physics": "physics",
    "Biology": "biology_neuroscience",
    "Medicine": "biology_neuroscience",
    "Engineering": "software_engineering",
}


def classify_paper(title: str, abstract: str | None, s2_fields: list[dict] | None) -> str:
    """Classify a paper into a research-kb domain."""
    text = f"{title} {abstract or ''}".lower()

    # Try keyword rules
    for keywords, _, domain in DOMAIN_RULES:
        if any(kw.lower() in text for kw in keywords):
            return domain

    # Fallback to S2 fields
    if s2_fields:
        for field in s2_fields:
            category = field.get("category", "")
            if category in S2_FIELD_MAP:
                return S2_FIELD_MAP[category]

    # Default (overridden by --default-domain CLI flag)
    return None


ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


async def fetch_and_classify(
    source_dir: Path,
    dest_dir: Path,
    dry_run: bool = False,
    force: bool = False,
    default_domain: str = "machine_learning",
) -> None:
    """Fetch S2 metadata for arXiv papers and create sidecars."""
    configure_logging(level="WARNING")

    # Find all arXiv PDFs with valid arXiv ID filenames
    pdfs = sorted(source_dir.glob("*.pdf"))
    arxiv_ids = []
    skipped_invalid = 0
    for pdf in pdfs:
        arxiv_id = pdf.stem
        if not ARXIV_ID_PATTERN.match(arxiv_id):
            logger.warning("invalid_arxiv_filename", filename=pdf.name)
            skipped_invalid += 1
            continue
        arxiv_ids.append((arxiv_id, pdf))

    print(f"Found {len(arxiv_ids)} valid arXiv papers in {source_dir}")
    if skipped_invalid:
        print(f"  Skipped {skipped_invalid} files with non-arXiv filenames")

    # Batch fetch from S2 with retry on rate limit
    max_retries = 3
    papers = []
    async with S2Client() as client:
        s2_ids = [f"arXiv:{aid}" for aid, _ in arxiv_ids]

        print(f"Fetching metadata from Semantic Scholar ({len(s2_ids)} papers)...")
        for attempt in range(1, max_retries + 1):
            try:
                papers = await client.get_papers_batch(s2_ids)
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"  S2 API failed after {max_retries} attempts: {e}")
                    raise
                wait = 2**attempt
                print(
                    f"  S2 API error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)

    # Build lookup by arXiv ID
    paper_by_arxiv = {}
    for paper in papers:
        if paper and paper.external_ids:
            arxiv = paper.external_ids.get("ArXiv")
            if arxiv:
                paper_by_arxiv[arxiv] = paper

    found = len(paper_by_arxiv)
    print(f"S2 returned metadata for {found}/{len(arxiv_ids)} papers")

    # Classify and create sidecars
    results = []
    for arxiv_id, pdf_path in arxiv_ids:
        paper = paper_by_arxiv.get(arxiv_id)
        if paper:
            domain = (
                classify_paper(
                    paper.title or "",
                    paper.abstract,
                    paper.s2_fields_of_study,
                )
                or default_domain
            )
            sidecar = {
                "title": paper.title,
                "authors": [a.name for a in (paper.authors or [])],
                "year": paper.year,
                "arxiv_id": arxiv_id,
                "domain_id": domain,
                "source_type": "paper",
                "s2_paper_id": paper.paper_id,
                "citation_count": paper.citation_count,
                "venue": paper.venue,
                "abstract": (paper.abstract or "")[:500],
            }
        else:
            # No S2 data — use arXiv ID and fallback domain
            domain = default_domain
            sidecar = {
                "title": f"arXiv:{arxiv_id}",
                "authors": [],
                "year": int(arxiv_id[:2]) + 2000 if arxiv_id[:2].isdigit() else None,
                "arxiv_id": arxiv_id,
                "domain_id": domain,
                "source_type": "paper",
            }

        results.append((arxiv_id, pdf_path, domain, sidecar))

    # Print summary
    from collections import Counter

    domain_counts = Counter(d for _, _, d, _ in results)
    print(f"\n=== Domain Classification ===")
    for domain, count in domain_counts.most_common():
        print(f"  {domain:25s}: {count}")

    if dry_run:
        print(f"\n[DRY RUN] Would copy {len(results)} papers to {dest_dir}")
        for arxiv_id, _, domain, sidecar in results:
            title = sidecar.get("title", "?")[:60]
            print(f"  {arxiv_id:15s} → {domain:25s} {title}")
        return

    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy PDFs and create sidecars
    copied = 0
    sidecars_written = 0
    sidecars_skipped = 0
    for arxiv_id, pdf_path, domain, sidecar in results:
        dest_pdf = dest_dir / f"{arxiv_id}.pdf"
        dest_json = dest_dir / f"{arxiv_id}.json"

        if not dest_pdf.exists():
            shutil.copy2(pdf_path, dest_pdf)
            copied += 1

        # Write sidecar (skip existing unless --force)
        if dest_json.exists() and not force:
            sidecars_skipped += 1
            continue
        with open(dest_json, "w") as f:
            json.dump(sidecar, f, indent=2)
        sidecars_written += 1

    print(f"\nCopied {copied} new PDFs, wrote {sidecars_written} sidecars in {dest_dir}")
    if sidecars_skipped:
        print(f"  Skipped {sidecars_skipped} existing sidecars (use --force to overwrite)")


def main():
    parser = argparse.ArgumentParser(description="Classify arXiv papers via S2 and create sidecars")
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory with arXiv PDFs")
    parser.add_argument(
        "--dest-dir", type=Path, required=True, help="Destination for copies + sidecars"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying")
    parser.add_argument("--force", action="store_true", help="Overwrite existing sidecar files")
    parser.add_argument(
        "--default-domain",
        type=str,
        default="machine_learning",
        help="Fallback domain for papers without S2 metadata (default: machine_learning)",
    )
    args = parser.parse_args()

    asyncio.run(
        fetch_and_classify(
            args.source_dir,
            args.dest_dir,
            dry_run=args.dry_run,
            force=args.force,
            default_domain=args.default_domain,
        )
    )


if __name__ == "__main__":
    main()
