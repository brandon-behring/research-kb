#!/usr/bin/env python3
"""Manifest-driven batch ingestion for acquisition books.

Ingests ~28 books from ~/Downloads/acqusition_books/ with dual-extractor support:
Docling (default, 21 books) and MinerU (7 math-heavy books with formulas).

Designed for the RTX 2070 three-phase GPU workflow:

  Phase 1: python scripts/ingest_acquisition_batch.py              # Extraction (--no-embed default)
  Phase 2: python scripts/backfill_embeddings.py --batch-size 8    # Embed (embed_server on GPU)
  Phase 3: python scripts/build_citation_graph.py                  # Citation graph (CPU only)

Usage:
    python scripts/ingest_acquisition_batch.py                     # Extract all
    python scripts/ingest_acquisition_batch.py --dry-run           # Preview manifest
    python scripts/ingest_acquisition_batch.py --resume            # Resume from checkpoint
    python scripts/ingest_acquisition_batch.py --only 0,3,5        # Specific entries
    python scripts/ingest_acquisition_batch.py --extractor docling # Force all Docling
    python scripts/ingest_acquisition_batch.py --extractor mineru  # Force all MinerU
"""

import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_contracts import SourceType
from research_kb_storage import (
    ChunkStore,
    SourceStore,
    close_connection_pool,
    get_connection_pool,
)

logger = get_logger(__name__)

# ── Checkpoint ────────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path(__file__).parent.parent / "data" / "acquisition_batch_checkpoint.json"


def load_checkpoint() -> set[str]:
    """Load completed book keys from checkpoint file."""
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        return set(data.get("completed", []))
    return set()


def save_checkpoint(completed: set[str]) -> None:
    """Atomically save checkpoint."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"completed": sorted(completed)}, indent=2))
    tmp.rename(CHECKPOINT_PATH)


# ── Manifest ──────────────────────────────────────────────────────────────

ACQ = Path("/home/brandon_behring/Downloads/acqusition_books/acqusition_books")


def resolve_path(p: Path) -> Path:
    """Resolve a manifest path against the filesystem, handling NFC/NFD mismatches.

    Filenames on ext4 may use NFD (decomposed) Unicode while Python source uses NFC
    (precomposed). This function tries the path as-is, then NFD-normalized, then falls
    back to prefix matching within the parent directory.

    Parameters
    ----------
    p : Path
        The path to resolve.

    Returns
    -------
    Path
        The resolved path that exists on disk, or the original path if no match found.
    """
    if p.exists():
        return p

    # Try NFD normalization
    nfd_name = unicodedata.normalize("NFD", p.name)
    nfd_path = p.parent / nfd_name
    if nfd_path.exists():
        return nfd_path

    # Fallback: prefix match (first 30 ASCII chars are always unique)
    prefix = p.name[:30]
    nfd_prefix = unicodedata.normalize("NFD", prefix)
    try:
        for f in os.listdir(str(p.parent)):
            if f.startswith(prefix) or f.startswith(nfd_prefix):
                candidate = p.parent / f
                if candidate.exists():
                    return candidate
    except OSError:
        pass

    return p  # Return original if nothing found


MANIFEST: list[dict[str, Any]] = [
    # ── Causal Inference / Econometrics (4) — Docling ────────────────────
    {
        "key": "imbens_rubin_causal_inference",
        "title": "Causal Inference for Statistics, Social, and Biomedical Sciences",
        "authors": ["Guido W. Imbens", "Donald B. Rubin"],
        "year": 2015,
        "domain_id": "causal_inference",
        "extractor": "docling",
        "pdf": ACQ
        / "Causal inference for statistics, social, and biomedical -- Guido W_ Imbens, Donald B_ Rubin -- 1st, 2015 -- Cambridge University Press (Virtual -- 9780521885881 -- bde1ce8f355d75ab11147a1d5feada15 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "morgan_winship_counterfactuals",
        "title": "Counterfactuals and Causal Inference: Methods and Principles for Social Research",
        "authors": ["Stephen L. Morgan", "Christopher Winship"],
        "year": 2015,
        "domain_id": "causal_inference",
        "extractor": "docling",
        "pdf": ACQ
        / "Counterfactuals and Causal Inference_ Methods and Principles -- Stephen L_ Morgan, Christopher Winship -- Analytical Methods for Social Research, -- 9781107065079 -- 4c5aad916d4298de2b709e4dc5d61726 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "angrist_pischke_mastering_metrics",
        "title": "Mastering 'Metrics: The Path From Cause to Effect",
        "authors": ["Joshua D. Angrist", "J\u00f6rn-Steffen Pischke"],
        "year": 2015,
        "domain_id": "econometrics",
        "extractor": "docling",
        "pdf": ACQ
        / "Mastering 'Metrics_ The Path From Cause to Effect -- Joshua David Angrist & J\u00f6rn-Steffen Pischke -- 2015 -- PrincetonUP -- a2c77b9078db4c71cee3b7915ca6b060 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "rosenbaum_observational_studies",
        "title": "An Introduction to the Theory of Observational Studies",
        "authors": ["Paul R. Rosenbaum"],
        "year": 2025,
        "domain_id": "causal_inference",
        "extractor": "docling",
        "pdf": ACQ
        / "An Introduction to the Theory of Observational Studies -- Paul R_ Rosenbaum -- 2025 -- x -- 9783031904943 -- 04c873326fbc53b53921ab59088d5795 -- Anna\u2019s Archive.pdf",
    },
    # ── ML / Deep Learning / RL (4) — Docling ───────────────────────────
    {
        "key": "albada_ai_agents",
        "title": "Building Applications with AI Agents",
        "authors": ["Michael Albada"],
        "year": 2025,
        "domain_id": "rag_llm",
        "extractor": "docling",
        "pdf": ACQ
        / "Building Applications with AI Agents_ Designing and -- Michael Albada -- 2025 -- O'Reilly Media, Incorporated -- 9781098176501 -- 75c008c32d52fce6765e38c1b6407d95 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "lapan_deep_rl_3e",
        "title": "Deep Reinforcement Learning Hands-On",
        "authors": ["Maxim Lapan"],
        "year": 2024,
        "domain_id": "deep_learning",
        "extractor": "docling",
        "pdf": ACQ
        / "Deep Reinforcement Learning Hands-On_ A practical and -- Maxim Lapan -- 3, 2024 -- Packt -- 04eac3a9d8105611453091a5f505d3b0 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "albrecht_multi_agent_rl",
        "title": "Multi-Agent Reinforcement Learning: Foundations and Modern Approaches",
        "authors": ["Stefano V. Albrecht", "Filippos Christianos", "Lukas Sch\u00e4fer"],
        "year": 2024,
        "domain_id": "machine_learning",
        "extractor": "docling",
        "pdf": ACQ
        / "Multi-Agent Reinforcement Learning_ Foundations and Modern -- Stefano V_ Albrecht, Filippos Christianos, Lukas Sch\u00e4fer -- 0a1eb0bc49df5494e6fb1eea8c6afa60 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "hilpisch_rl_finance",
        "title": "Reinforcement Learning for Finance",
        "authors": ["Yves J. Hilpisch"],
        "year": 2024,
        "domain_id": "machine_learning",
        "extractor": "docling",
        "pdf": ACQ
        / "Reinforcement Learning for Finance_ A Python-Based -- Yves J_ Hilpisch -- 1, 2024 -- O'Reilly Media, Incorporated -- 9781098169145 -- 01065ef05b3a9e58bade4428417683bf -- Anna\u2019s Archive.pdf",
    },
    # ── Data Engineering / SE (5) — Docling ──────────────────────────────
    {
        "key": "reis_housley_data_engineering",
        "title": "Fundamentals of Data Engineering",
        "authors": ["Joe Reis", "Matt Housley"],
        "year": 2022,
        "domain_id": "software_engineering",
        "extractor": "docling",
        "pdf": ACQ
        / "Fundamentals of data engineering _ plan and build robust -- Joe Reis, Matt Housley -- 1 _ converted, 2022 -- O'Reilly Media, Incorporated -- 9781098108274 -- 154c987de8f3d58b451e2479b54661e7 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "nelson_se_data_scientists",
        "title": "Software Engineering for Data Scientists",
        "authors": ["Catherine Nelson"],
        "year": 2024,
        "domain_id": "software_engineering",
        "extractor": "docling",
        "pdf": ACQ
        / "Software Engineering for Data Scientists_ From Notebooks to -- Catherine Nelson -- 1 _ converted, 2024 -- O'Reilly Media, Incorporated -- 9781098136178 -- 195635a2e35f97b4e0f3e3a570a43393 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "petrov_database_internals",
        "title": "Database Internals: A Deep Dive Into How Distributed Data Systems Work",
        "authors": ["Alex Petrov"],
        "year": 2019,
        "domain_id": "software_engineering",
        "extractor": "docling",
        "pdf": ACQ
        / "Database Internals _ A Deep Dive Into How Distributed Data -- Alex Petrov -- O'REILLY, 1, 2019 -- O'Reilly Media, Incorporated -- 9781492040316 -- 05ef18aa9887e93cd3babf136e65001b -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "fowler_refactoring_2e",
        "title": "Refactoring: Improving the Design of Existing Code",
        "authors": ["Martin Fowler", "Kent Beck"],
        "year": 2018,
        "domain_id": "software_engineering",
        "extractor": "docling",
        "pdf": ACQ
        / "Refactoring_ Improving the Design of Existing Code -- Martin Fowler, Kent Beck -- Addison-Wesley Signature Series (Fowler) Ser, 2nd ed, -- 9780134757599 -- f4fc75eb1eee5537400908e6f59db631 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "slatkin_effective_python_3e",
        "title": "Effective Python: 125 Specific Ways to Write Better Python",
        "authors": ["Brett Slatkin"],
        "year": 2024,
        "domain_id": "software_engineering",
        "extractor": "docling",
        # 670-page Pearson edition (true 3rd ed), not the 1164-page anomalous copy
        "pdf": ACQ
        / "Effective Python_ 125 Specific Ways to Write Better Python_ -- Brett Slatkin -- 3, 2024 nov 30 -- Pearson Education, Limited -- 9780138172183 -- 859e2b0eab74efb9626b6b79a353f72a -- Anna\u2019s Archive.pdf",
    },
    # ── Python / SQL (3) — Docling ───────────────────────────────────────
    {
        "key": "ramalho_fluent_python_2e",
        "title": "Fluent Python: Clear, Concise, and Effective Programming",
        "authors": ["Luciano Ramalho"],
        "year": 2022,
        "domain_id": "software_engineering",
        "extractor": "docling",
        "pdf": ACQ
        / "Fluent Python_ Clear, Concise, and Effective Programming, -- Luciano Ramalho -- 2nd, 2022 -- Beijing _ O'Reilly Media, Inc -- 9781492056355 -- 6b8f1e751c6d6b82a49cc155099f9949 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "beaulieu_learning_sql",
        "title": "Learning SQL: Generate, Manipulate, and Retrieve Data",
        "authors": ["Alan Beaulieu"],
        "year": 2020,
        "domain_id": "software_engineering",
        "extractor": "docling",
        "pdf": ACQ
        / "Learning SQL_ Generate, Manipulate, and Retrieve Data -- Alan Beaulieu -- 2020 -- O'Reilly Media, Incorporated -- 9781492057611 -- 089bb3c2d0f555be4c292251f45c4863 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "molinaro_sql_cookbook_2e",
        "title": "SQL Cookbook: Query Solutions and Techniques for All SQL Users",
        "authors": ["Anthony Molinaro", "Robert W. De Graaf"],
        "year": 2020,
        "domain_id": "software_engineering",
        "extractor": "docling",
        "pdf": ACQ
        / "SQL cookbook _ query solutions and techniques for all SQL -- Anthony Molinaro, database developer; Robert W De Graaf -- Second edition, Sebastopol, -- 9781492077442 -- f26030bcee656c6a942752267fba1cba -- Anna\u2019s Archive.pdf",
    },
    # ── Recommender Systems / Advertising (3) — Docling ──────────────────
    {
        "key": "falk_practical_recsys",
        "title": "Practical Recommender Systems",
        "authors": ["Kim Falk"],
        "year": 2018,
        "domain_id": "machine_learning",
        "extractor": "docling",
        "pdf": ACQ
        / "Practical Recommender Systems -- Kim Falk -- 2018 -- 2e2ed031f9005b45d39f2cab37c6259f -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "ricci_recsys_handbook_3e",
        "title": "Recommender Systems Handbook",
        "authors": ["Francesco Ricci", "Lior Rokach", "Bracha Shapira"],
        "year": 2022,
        "domain_id": "machine_learning",
        "extractor": "docling",
        "pdf": ACQ
        / "Recommender Systems Handbook -- Francesco Ricci (editor), Lior Rokach (editor), Bracha -- 3rd ed_ 2022, New York, NY, 2022 -- Springer US _ Imprint_ -- 9781071621967 -- aae9c870d761525c9123aae2abdb0d37 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "liu_wang_computational_advertising",
        "title": "Computational Advertising: Market and Technologies for Internet Commercial Monetization",
        "authors": ["Peng Liu", "Chao Wang"],
        "year": 2020,
        "domain_id": "machine_learning",
        "extractor": "docling",
        "pdf": ACQ
        / "Computational Advertising_ Market and Technologies for the -- Peng Liu (Author); Chao Wang (Author) -- 2, 2020 may 12 -- Taylor & Francis Group; CRC -- 9780367206383 -- 7e99156b75a7c11af78b90152e633aa8 -- Anna\u2019s Archive.pdf",
    },
    # ── Cloud / GenAI (2) — Docling ──────────────────────────────────────
    {
        "key": "adedeji_genai_google_cloud",
        "title": "GenAI on Google Cloud",
        "authors": ["Ayo Adedeji", "Lavi Nigam", "Sarita Joshi", "Stephanie Gervasi"],
        "year": 2025,
        "domain_id": "rag_llm",
        "extractor": "docling",
        "pdf": ACQ
        / "GenAI on Google Cloud (for Duc Ka) -- Ayo Adedeji, Lavi Nigam, Sarita Joshi & Stephanie Gervasi -- 2025 -- O'Reilly Media, Inc_ -- c5984b54d19fa5d310166960a6d19060 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "kumar_vertex_ai",
        "title": "Learning Google Cloud Vertex AI",
        "authors": ["Hemanth Kumar K"],
        "year": 2024,
        "domain_id": "rag_llm",
        "extractor": "docling",
        "pdf": ACQ
        / "Learning Google Cloud Vertex AI_ Build, deploy, and manage -- Hemanth Kumar K -- 2024 -- BPB Publications -- 9789355515353 -- 9a2727cc40768ed005fecb9e92c51784 -- Anna\u2019s Archive.pdf",
    },
    # ── Mathematics (7) — MinerU (formula-heavy) ─────────────────────────
    {
        "key": "horn_johnson_matrix_analysis_2e",
        "title": "Matrix Analysis",
        "authors": ["Roger A. Horn", "Charles R. Johnson"],
        "year": 2013,
        "domain_id": "mathematics",
        "extractor": "mineru",
        "pdf": ACQ
        / "Matrix Analysis_ Second Edition -- Johnson, Charles R_;Horn, Roger A -- 2nd edition, 2013 -- Cambridge University Press (Virtual Publishing) -- 9780521548236 -- 1f92754e944bf039dbad5aca93106e9d -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "von_zur_gathen_modern_computer_algebra",
        "title": "Modern Computer Algebra",
        "authors": ["Joachim von zur Gathen", "J\u00fcrgen Gerhard"],
        "year": 2013,
        "domain_id": "mathematics",
        "extractor": "mineru",
        "pdf": ACQ
        / "Modern Computer Algebra -- Joachim von zur Gathen, J\u00fcrgen Gerhards -- EBL-Schweitzer, 3rd ed, Cambridge, U_K, 2013 -- Cambridge University Press -- 9781107039032 -- 1582d237b4f838eb5c0474487647b867 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "becker_grobner_bases",
        "title": "Gr\u00f6bner Bases: A Computational Approach to Commutative Algebra",
        "authors": ["Thomas Becker", "Volker Weispfenning", "H. Kredel"],
        "year": 1993,
        "domain_id": "mathematics",
        "extractor": "mineru",
        "pdf": ACQ
        / "Gr\u00f6bner Bases _ A Computational Approach to Commutative -- Thomas Becker, Volker Weispfenning, H_ Kredel -- Graduate Texts in Mathematics, 141, 1993 -- 9780387979717 -- d556abcb8625c0c9466b3923b1d0a1eb -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "baader_nipkow_term_rewriting",
        "title": "Term Rewriting and All That",
        "authors": ["Franz Baader", "Tobias Nipkow"],
        "year": 1998,
        "domain_id": "mathematics",
        "extractor": "mineru",
        "pdf": ACQ
        / "Term Rewriting and All That -- Franz Baader and Tobias Nipkow -- 1, 1998 -- Cambridge University Press (Virtual Publishing) -- 9780521455206 -- 51f787d9f0b736c8b1f84d70a256fbd0 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "winkler_polynomial_algorithms",
        "title": "Polynomial Algorithms in Computer Algebra",
        "authors": ["Franz Winkler"],
        "year": 1996,
        "domain_id": "mathematics",
        "extractor": "mineru",
        # Copy without (1) suffix — both identical
        "pdf": ACQ
        / "[Texts and Monographs in Symbolic Computation] Polynomial -- Dipl_-Ing_ Dr_ Franz Winkler (auth_) -- 10_1007_97, 1996 -- Springer Vienna _ Imprint _ -- 9783211827598 -- 0a14fa52ae58094410c4c04d3139cd31 -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "hairer_geometric_numerical_integration",
        "title": "Geometric Numerical Integration: Structure-Preserving Algorithms for ODEs",
        "authors": ["Ernst Hairer", "Christian Lubich", "Gerhard Wanner"],
        "year": 2006,
        "domain_id": "mathematics",
        "extractor": "mineru",
        "pdf": ACQ
        / "Geometric Numerical Integration_ Structure-Preserving -- Ernst Hairer, Christian Lubich, Gerhard Wanner -- Springer Series in Computational -- 9783540306634 -- 4d11600f27efa1b63e1ef81e95f3df8e -- Anna\u2019s Archive.pdf",
    },
    {
        "key": "hairer_solving_odes_i",
        "title": "Solving Ordinary Differential Equations I: Nonstiff Problems",
        "authors": ["Ernst Hairer", "Gerhard Wanner", "Syvert P. N\u00f8rsett"],
        "year": 1993,
        "domain_id": "mathematics",
        "extractor": "mineru",
        "pdf": ACQ
        / "Solving Ordinary Differential Equations I_ Nonstiff Problems -- Ernst Hairer, Gerhard Wanner, Syvert P_ N\u00f8rsett (auth_) -- Springer Series in -- 9780387566702 -- 87d38520393f83b83d229c27b78f0dd4 -- Anna\u2019s Archive.pdf",
    },
    # NOTE: Solving ODEs II EXCLUDED — corrupted PDF (0 pages, MuPDF syntax error)
    # NOTE: Hamilton Time Series EXCLUDED — same book already ingested from fixtures
    # NOTE: Huyen Designing ML Systems EXCLUDED — identical file hash to fixture
]

# Resolve all paths against filesystem (handles NFC/NFD Unicode mismatches)
for _entry in MANIFEST:
    _entry["pdf"] = resolve_path(_entry["pdf"])


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _get_extractor(name: str):
    """Lazy-import and return (extract_and_chunk, method_name) for the given extractor.

    Parameters
    ----------
    name : str
        Either "docling" or "mineru".

    Returns
    -------
    tuple[Callable, str]
        (extract_and_chunk function, extraction_method string)
    """
    if name == "mineru":
        from research_kb_pdf.mineru_extractor import extract_and_chunk

        return extract_and_chunk, "mineru"
    else:
        from research_kb_pdf.docling_extractor import extract_and_chunk

        return extract_and_chunk, "docling"


async def ingest_one(
    entry: dict[str, Any],
    extractor_override: str | None = None,
    quiet: bool = False,
) -> tuple[str, int]:
    """Extract one PDF and store in DB.

    Parameters
    ----------
    entry : dict
        Manifest entry with key, title, authors, year, domain_id, extractor, pdf.
    extractor_override : str | None
        If set, overrides the per-entry extractor choice.
    quiet : bool
        Suppress verbose logging.

    Returns
    -------
    tuple[str, int]
        (source_id, chunk_count)
    """
    pdf_path = str(entry["pdf"])
    title = entry["title"]
    authors = entry["authors"]
    year = entry["year"]
    domain_id = entry["domain_id"]
    extractor_name = extractor_override or entry.get("extractor", "docling")

    extract_fn, method_name = _get_extractor(extractor_name)

    if not quiet:
        logger.info("extracting_pdf", path=pdf_path, title=title, extractor=method_name)

    t0 = time.time()
    extraction_result, chunks = extract_fn(pdf_path, max_tokens=300)
    elapsed = time.time() - t0

    metadata = {
        "extraction_method": method_name,
        "total_pages": extraction_result.total_pages,
        "total_chars": extraction_result.total_chars,
        "total_headings": extraction_result.heading_count,
        "total_chunks": len(chunks),
        "has_equations": extraction_result.has_equations,
        "extraction_time_s": round(elapsed, 1),
        "domain": domain_id,
    }

    if not quiet:
        logger.info(
            "extraction_complete",
            title=title,
            extractor=method_name,
            chunks=len(chunks),
            pages=extraction_result.total_pages,
            time_s=round(elapsed, 1),
        )

    file_hash = compute_file_hash(pdf_path)

    source = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,
        title=title,
        file_hash=file_hash,
        domain_id=domain_id,
        authors=authors,
        year=year,
        file_path=pdf_path,
        metadata=metadata,
    )

    # Batch insert chunks
    chunks_data = []
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        chunks_data.append(
            {
                "source_id": source.id,
                "content": chunk.content,
                "content_hash": content_hash,
                "page_start": chunk.start_page,
                "page_end": chunk.end_page,
                "embedding": None,  # Phase 2
                "domain_id": domain_id,
                "metadata": {
                    "section_header": chunk.metadata.get("section", ""),
                    "chunk_index": i,
                },
            }
        )

    BATCH_SIZE = 100
    chunks_created = 0
    for i in range(0, len(chunks_data), BATCH_SIZE):
        batch = chunks_data[i : i + BATCH_SIZE]
        await ChunkStore.batch_create(batch)
        chunks_created += len(batch)

    if not quiet:
        logger.info(
            "ingestion_complete",
            source_id=str(source.id),
            title=title,
            chunks=chunks_created,
        )

    return str(source.id), chunks_created


async def run(args: argparse.Namespace) -> None:
    """Main entry point."""
    # Filter manifest
    manifest = MANIFEST
    if args.only:
        indices = {int(x.strip()) for x in args.only.split(",")}
        manifest = [m for i, m in enumerate(MANIFEST) if i in indices]

    # Dry run
    if args.dry_run:
        print(f"\n{'#':>3s}  {'EXTRACTOR':10s}  {'DOMAIN':22s}  {'KEY':48s}  PDF EXISTS?")
        print("-" * 120)
        for i, entry in enumerate(MANIFEST):
            if args.only:
                indices = {int(x.strip()) for x in args.only.split(",")}
                if i not in indices:
                    continue
            exists = entry["pdf"].exists()
            flag = "OK" if exists else "MISSING"
            ext = args.extractor or entry.get("extractor", "docling")
            print(f"{i:3d}  {ext:10s}  {entry['domain_id']:22s}  {entry['key']:48s}  {flag}")
            if not exists:
                print(f"     -> {entry['pdf']}")
        missing = [e for e in manifest if not e["pdf"].exists()]
        print(f"\nTotal: {len(manifest)} books, {len(missing)} PDFs missing")
        return

    # Verify all PDFs exist before starting
    missing = [e for e in manifest if not e["pdf"].exists()]
    if missing:
        print("ERROR: Missing PDFs:")
        for e in missing:
            print(f"  {e['key']}: {e['pdf']}")
        sys.exit(1)

    # Load checkpoint
    completed = load_checkpoint() if args.resume else set()

    # Initialize DB pool
    await get_connection_pool()

    total_chunks = 0
    total_ingested = 0
    failed: list[tuple[str, str]] = []
    prev_extractor: str | None = None

    for i, entry in enumerate(manifest):
        key = entry["key"]

        if key in completed:
            print(f"[{i + 1}/{len(manifest)}] SKIP (checkpoint): {key}")
            continue

        # Determine extractor for this entry
        current_extractor = args.extractor or entry.get("extractor", "docling")

        # Free Docling VRAM before switching to MinerU
        if prev_extractor == "docling" and current_extractor == "mineru":
            import gc

            from research_kb_pdf.docling_extractor import reset_converter

            reset_converter()  # Deletes model + torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)  # Let CUDA fully release memory to OS
            logger.info("switched_extractor", from_="docling", to="mineru")
            print("  [Switched extractor: docling -> mineru, freed VRAM]")
            prev_extractor = "mineru"  # Prevent re-triggering on subsequent entries

        print(f"\n[{i + 1}/{len(manifest)}] {current_extractor.upper()}: {entry['title']}")

        try:
            source_id, chunk_count = await ingest_one(
                entry, extractor_override=args.extractor, quiet=args.quiet
            )
            total_chunks += chunk_count
            total_ingested += 1

            print(f"  OK: {chunk_count} chunks, source_id={source_id[:12]}")

            # Checkpoint
            completed.add(key)
            save_checkpoint(completed)

            prev_extractor = current_extractor

        except Exception as e:
            print(f"  FAILED: {e}")
            logger.error("book_failed", key=key, error=str(e))
            failed.append((key, str(e)))

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"  Ingested: {total_ingested}/{len(manifest)} books")
    print(f"  Chunks:   {total_chunks}")
    print(f"  Failed:   {len(failed)}")
    if failed:
        for key, err in failed:
            print(f"    {key}: {err[:80]}")
    print(f"{'=' * 60}")

    await close_connection_pool()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manifest-driven batch ingestion for acquisition books (dual-extractor)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview manifest without executing")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated manifest indices to process (e.g., '0,3,5')",
    )
    parser.add_argument(
        "--extractor",
        choices=["docling", "mineru"],
        default=None,
        help="Force extractor for all entries (overrides per-entry setting)",
    )
    args = parser.parse_args()

    import asyncio

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
