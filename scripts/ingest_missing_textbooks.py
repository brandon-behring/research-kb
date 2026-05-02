#!/usr/bin/env python3
"""Ingest missing textbooks from fixtures/textbooks/ not in database.

Three-phase GPU workflow (RTX 2070, 8GB VRAM):
  Phase 1: python scripts/ingest_missing_textbooks.py          # Extract only (Docling on GPU)
  Phase 2: python scripts/backfill_embeddings.py --batch-size 8 # Embed only (embed_server on GPU)
  Phase 3: python scripts/build_citation_graph.py               # Citation graph (CPU only)

Phases 1 and 2 CANNOT share the GPU. This script defaults to --no-embed
(extraction only). Use --with-embed only if embed_server is NOT running.

Usage:
    python scripts/ingest_missing_textbooks.py          # Extract only (default)
    python scripts/ingest_missing_textbooks.py --quiet  # Errors + summary only
    python scripts/ingest_missing_textbooks.py --with-embed  # Single-phase (unsafe on shared GPU)
"""

import argparse
import asyncio
import hashlib
import re
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_contracts import SourceType
from research_kb_pdf import (
    EmbeddingClient,
    extract_and_chunk,
)
from research_kb_storage import (
    ChunkStore,
    DatabaseConfig,
    SourceStore,
    close_connection_pool,
    get_connection_pool,
)

logger = get_logger(__name__)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_sidecar_metadata(pdf_path: Path) -> dict | None:
    """Load metadata from JSON sidecar if it exists."""
    json_path = pdf_path.with_suffix(".json")
    if json_path.exists():
        import json

        try:
            with open(json_path) as f:
                data = json.load(f)
            # Clean up title from sidecar (remove underscores)
            title = data.get("title", "").replace("_", " ").strip()
            return {
                "title": title,
                "authors": data.get("authors", []),
                "year": data.get("year"),
                "source_db": data.get("source_db"),
                "domain": data.get("domain_id") or data.get("domain"),
            }
        except Exception:
            return None
    return None


def parse_filename_for_metadata(filename: str) -> dict:
    """Extract metadata from textbook filename patterns."""
    name = filename.replace(".pdf", "")

    # Handle tier-prefixed files (tier1_01_title_year.pdf)
    tier_match = re.match(r"^tier\d+_\d+_(.+)_(\d{4})$", name)
    if tier_match:
        title = tier_match.group(1).replace("_", " ").title()
        year = int(tier_match.group(2))
        return {"title": title, "authors": [], "year": year}

    # Handle Train Discrete Choice chapters
    if "train_discrete_choice" in name.lower():
        return {
            "title": name.replace("_", " ").title(),
            "authors": ["Train, Kenneth E."],
            "year": 2009,
        }

    # Try standard patterns
    year_match = re.search(r"(\d{4})", name)
    year = int(year_match.group(1)) if year_match else None

    # Handle "Applied Bayesian..." long format
    if "Applied Bayesian" in name or "Rubin" in name:
        return {
            "title": "Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives",
            "authors": ["Gelman, Andrew", "Rubin, Donald B.", "Meng, Xiao-Li"],
            "year": 2004,
        }

    title = name.replace("_", " ").title()
    return {"title": title, "authors": [], "year": year}


async def ingest_textbook(
    pdf_path: str,
    title: str,
    authors: list[str],
    year: int | None,
    domain_id: str,
    metadata: dict | None = None,
    quiet: bool = False,
    no_embed: bool = False,
) -> tuple[str, int, int]:
    """Ingest a single textbook PDF.

    Returns: (source_id, chunks_created, headings_found)
    """
    if not quiet:
        logger.info("extracting_pdf", path=pdf_path)

    # Extract and chunk with Docling
    extraction_result, chunks = extract_and_chunk(pdf_path, max_tokens=300)

    metadata = metadata or {}
    metadata["extraction_method"] = "docling"
    metadata["total_pages"] = extraction_result.total_pages
    metadata["total_chars"] = extraction_result.total_chars
    metadata["total_headings"] = extraction_result.heading_count
    metadata["total_chunks"] = len(chunks)
    metadata["has_equations"] = extraction_result.has_equations

    if not quiet:
        logger.info("extraction_complete", path=pdf_path, chunks=len(chunks))

    # Compute file hash
    file_hash = compute_file_hash(pdf_path)

    # Create source record as TEXTBOOK
    if not quiet:
        logger.info("creating_source", title=title)
    metadata["domain"] = domain_id
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

    # Generate embeddings (unless --no-embed for two-phase GPU workflow)
    if no_embed:
        if not quiet:
            logger.info("skipping_embeddings", chunks=len(chunks))
        embeddings = [None] * len(chunks)
    else:
        embedding_client = EmbeddingClient()
        texts = [chunk.content for chunk in chunks]

        if not quiet:
            logger.info("generating_embeddings", chunks=len(chunks))

        embeddings = embedding_client.embed_batch(texts, batch_size=32)

    # Prepare batch data for insertion
    chunks_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        chunks_data.append(
            {
                "source_id": source.id,
                "content": chunk.content,
                "content_hash": content_hash,
                "page_start": chunk.start_page,
                "page_end": chunk.end_page,
                "embedding": embedding,
                "domain_id": domain_id,
                "metadata": {
                    "section_header": chunk.metadata.get("section", ""),
                    "chunk_index": i,
                },
            }
        )

    # Batch insert in groups of 100
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
            chunks=chunks_created,
            headings=extraction_result.heading_count,
        )

    return str(source.id), chunks_created, extraction_result.heading_count


# Skip list - auxiliary files without content
SKIP_PATTERNS = [
    "Ch0_Front",
    "Index_p",
    "Refs_p",
    "_Glossary",
    "Quicksheet",
]

# Filename-to-domain mapping for books without sidecar metadata.
# Keys are case-insensitive substrings matched against filename.
DOMAIN_KEYWORDS = {
    "causal": "causal_inference",
    "instrumental_variable": "causal_inference",
    "angrist": "causal_inference",
    "mostly_harmless": "causal_inference",
    "rubin": "causal_inference",
    "rag": "rag_llm",
    "llm": "rag_llm",
    "knowledge_graph": "rag_llm",
    "generative_ai": "rag_llm",
    "retrieval_augmented": "rag_llm",
    "ai_agents": "rag_llm",
    "deep_learning": "deep_learning",
    "pytorch": "deep_learning",
    "gnn": "deep_learning",
    "graph_neural": "deep_learning",
    "neural_network": "deep_learning",
    "machine_learning": "machine_learning",
    "grokking_algorithm": "algorithms",
    "time_series": "time_series",
    "forecasting": "forecasting",
    "box_jenkins": "time_series",
    "econometric": "econometrics",
    "statistics": "statistics",
    "statistical_method": "statistics",
    "scala": "functional_programming",
    "haskell": "functional_programming",
    "software_design": "software_engineering",
    "software_engineering": "software_engineering",
    "feedback_driven": "software_engineering",
    "cfa": "finance",
    "portfolio": "portfolio_management",
    "data_scienc": "data_science",
    "fitness": "fitness",
    # Algorithms
    "algorithm": "algorithms",
    "data_structure": "algorithms",
    "roughgarden": "algorithms",
    "common_sense_guide": "algorithms",
    "problem_solving_with": "algorithms",
    # SQL & Databases
    "sql": "sql",
    "database_internals": "sql",
    # Recommender Systems
    "recommender": "recommender_systems",
    "collaborative_filtering": "recommender_systems",
    "recommendation_system": "recommender_systems",
    # AdTech
    "ads_adtech": "adtech",
    "adtech": "adtech",
    # Forecasting
    "demand_forecasting": "forecasting",
    "practical_time_series": "forecasting",
    # Tier 1: core math
    "linear_algebra": "linear_algebra",
    "matrix_theory": "linear_algebra",
    "optimization": "optimization",
    "convex_optim": "optimization",
    "probability_theory": "probability_theory",
    "measure_theory": "probability_theory",
    "stochastic_process": "probability_theory",
    "real_analysis": "analysis",
    "complex_analysis": "analysis",
    "functional_analysis": "analysis",
    "harmonic_analysis": "analysis",
    "fourier_analysis": "analysis",
    "operator_theory": "analysis",
    "spectral_theor": "analysis",
    "ergodic": "analysis",
    "banach": "analysis",
    "hilbert_space": "analysis",
    "topology": "topology_geometry",
    "geometr": "topology_geometry",
    "manifold": "topology_geometry",
    "differential_geometry": "topology_geometry",
    "algebraic_topology": "topology_geometry",
    "homotopy": "topology_geometry",
    "homology": "topology_geometry",
    "cohomology": "topology_geometry",
    "knot_theory": "topology_geometry",
    "lie_group": "topology_geometry",
    "symplectic": "topology_geometry",
    "riemannian": "topology_geometry",
    "calculus": "mathematics",
    "differential_equation": "mathematics",
    "variation": "mathematics",
    "numerical_method": "mathematics",
    "numerical_analysis": "mathematics",
    "number_theory": "mathematics",
    "abstract_algebra": "mathematics",
    "commutative_algebra": "mathematics",
    "set_theory": "mathematics",
    "combinator": "mathematics",
    "graph_theory": "mathematics",
    "dynamical_system": "mathematics",
    "mathematical_proof": "mathematics",
    # Tier 3: physics & engineering
    "quantum_mechanic": "physics",
    "quantum_theory": "physics",
    "quantum_field": "physics",
    "classical_mechanic": "physics",
    "classical_dynamic": "physics",
    "general_relativit": "physics",
    "special_relativit": "physics",
    "statistical_mechanic": "physics",
    "thermodynamic": "physics",
    "electrodynamic": "physics",
    "feynman_lecture": "physics",
    "cohen_tannoudji": "physics",
    "control_theory": "engineering",
    "control_system": "engineering",
    "signal_processing": "engineering",
    "information_theory": "information_theory",
}


# Map legacy sidecar domain values to valid domain IDs.
# The migrated/ directory uses non-standard labels.
LEGACY_DOMAIN_MAP = {
    "programming": "software_engineering",
    "other": "general",  # was "machine_learning" — silent ML over-tagging path; "general" is honest
    "ml_stats": "statistics",
    "math": "mathematics",
    "nlp": "nlp",
    "causal": "causal_inference",
    "dynamical_systems": "mathematics",
    "numerical_methods": "mathematics",
    "algebra": "mathematics",
    "signal_processing": "engineering",
    "biology_neuroscience": "neuroscience",
}


# Maps fixtures/library_books/<dir>/ → canonical domain (audit script taxonomy).
# Used by infer_domain when filename keywords don't match: the parent directory
# in library_books is the curated, ground-truth domain assignment.
LIBRARY_DIR_TO_DOMAIN = {
    "algebra": "mathematics",
    "algorithms": "algorithms",
    "analysis": "analysis",
    "biology_neuroscience": "neuroscience",
    "causal_inference": "causal_inference",
    "data_science": "data_science",
    "deep_learning": "deep_learning",
    "dynamical_systems": "mathematics",
    "finance": "finance",
    "fitness": "fitness",
    "machine_learning": "machine_learning",
    "mathematics": "mathematics",
    "numerical_methods": "mathematics",
    "optimization": "optimization",
    "physics": "physics",
    "probability_theory": "probability_theory",
    "rag_llm": "rag_llm",
    "reinforcement_learning": "reinforcement_learning",
    "signal_processing": "engineering",
    "software_engineering": "software_engineering",
    "statistics": "statistics",
    "time_series": "time_series",
    "topology_geometry": "topology_geometry",
}


# Path under which unclassified-source records are appended for follow-up triage.
UNCLASSIFIED_DLQ = Path(__file__).parent.parent / "data" / "dlq" / "unclassified_sources.jsonl"


def _record_unclassified(pdf_path: Path, sidecar_meta: dict | None) -> None:
    """Append an unclassified-source record to the DLQ for later batch triage.

    Captures filename, full path, and sidecar keys so a follow-up session can
    decide on a canonical domain (or extend DOMAIN_KEYWORDS / LIBRARY_DIR_TO_DOMAIN).
    """
    import json
    from datetime import datetime, timezone

    UNCLASSIFIED_DLQ.parent.mkdir(parents=True, exist_ok=True)
    sidecar_keys = list(sidecar_meta.keys()) if sidecar_meta else []
    sidecar_domain = (sidecar_meta or {}).get("domain")
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "filename": pdf_path.name,
        "pdf_path": str(pdf_path),
        "sidecar_keys": sidecar_keys,
        "sidecar_domain_raw": sidecar_domain,
        "fallback_assigned": "general",
    }
    with UNCLASSIFIED_DLQ.open("a") as f:
        f.write(json.dumps(record) + "\n")


VALID_DOMAINS = {
    # Tier 1 (audit_formula_gaps.py)
    "mathematics",
    "linear_algebra",
    "optimization",
    "probability_theory",
    "analysis",
    "topology_geometry",
    # Tier 2
    "statistics",
    "machine_learning",
    "reinforcement_learning",
    "deep_learning",
    "causal_inference",
    "time_series",
    "econometrics",
    # Tier 3
    "physics",
    "engineering",
    "finance",
    "nlp",
    "information_theory",
    "rag_llm",
    "algorithms",
    "computer_science",
    # Tier 4
    "software_engineering",
    "functional_programming",
    "business",
    "economics",
    "neuroscience",
    "interview_prep",
    "general",
    # Other canonical (not tier-classified)
    "adtech",
    "data_science",
    "fitness",
    "forecasting",
    "healthcare",
    "ml_engineering",
    "portfolio_management",
    "recommender_systems",
    "sql",
}


def infer_domain(
    filename: str,
    sidecar_meta: dict | None = None,
    full_path: Path | None = None,
) -> str:
    """Infer domain_id from sidecar metadata, filesystem path, or filename keywords.

    Returns a valid domain string. Falls back to 'general' if no match,
    appending the source to data/dlq/unclassified_sources.jsonl for later triage
    and logging a warning so silent drift can be audited.

    Priority:
      1. Sidecar (if value already canonical) — highest trust.
      2. Library_books filesystem path — `fixtures/library_books/<DIR>/file.pdf`
         is the curated ground-truth domain assignment.
      3. Filename keyword match (specific books).
      4. Legacy sidecar mapping (non-canonical → canonical).
      5. Fallback: 'general' + DLQ + warning.
    """
    # 1. Sidecar metadata — use if it's already a valid canonical domain
    if sidecar_meta and sidecar_meta.get("domain"):
        raw = sidecar_meta["domain"]
        if raw in VALID_DOMAINS:
            return raw

    # 2. Filesystem path: library_books/<dir>/ encodes curated domain
    if full_path is not None:
        parts = full_path.parts
        for i, part in enumerate(parts):
            if part == "library_books" and i + 1 < len(parts):
                dir_name = parts[i + 1]
                mapped = LIBRARY_DIR_TO_DOMAIN.get(dir_name)
                if mapped:
                    return mapped
                break

    # 3. Filename keyword inference (most reliable for specific books)
    name_lower = filename.lower().replace(" ", "_").replace("-", "_")
    for keyword, domain in sorted(DOMAIN_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if keyword in name_lower:
            return domain

    # 4. Legacy sidecar mapping (non-canonical → canonical)
    if sidecar_meta and sidecar_meta.get("domain"):
        mapped = LEGACY_DOMAIN_MAP.get(sidecar_meta["domain"])
        if mapped:
            return mapped

    # 5. Fallback: 'general' + DLQ + warning. Was 'machine_learning' historically;
    # the silent ML default tagged ~260 math/physics books incorrectly (see issue #9
    # post-2026-04-25 follow-up). 'general' is honest — surfaces sources that need
    # manual classification rather than hiding them under a wrong canonical domain.
    fallback_path = full_path if full_path is not None else Path(filename)
    _record_unclassified(fallback_path, sidecar_meta)
    logger.warning(
        "domain_classifier_fallback",
        filename=filename,
        path=str(fallback_path) if full_path is not None else None,
        sidecar_domain_raw=(sidecar_meta or {}).get("domain"),
        assigned="general",
        note="appended to data/dlq/unclassified_sources.jsonl",
    )
    return "general"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest missing textbooks from fixtures/textbooks/."
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output (one line per PDF + final summary)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output summary as JSON (for programmatic parsing)",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        default=True,
        help="Skip embedding generation (default: True). Use backfill_embeddings.py after.",
    )
    parser.add_argument(
        "--with-embed",
        action="store_true",
        help="Enable embedding during ingestion (single-phase, unsafe on shared GPU).",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Custom PDF directory to scan (default: fixtures/textbooks/). "
        "Use for library_books or other collections with sidecar JSONs.",
    )
    parser.add_argument(
        "--auto-stop-services",
        action="store_true",
        help=(
            "Stop research_kb_rerank.service if VRAM is below the Docling floor, "
            "then restart on exit. Default: abort with a systemctl hint."
        ),
    )
    return parser.parse_args()


async def main():
    """Ingest all missing textbooks from fixtures/textbooks/."""
    import json as json_module
    import time

    from research_kb_common import EmbeddingError, StorageError, configure_logging

    args = parse_args()

    # Resolve embed mode: --with-embed overrides --no-embed
    no_embed = not args.with_embed
    quiet = args.quiet
    json_output = args.json

    # Safety check: abort if embed_server is running during extraction
    if not no_embed:
        from research_kb_common.gpu_guard import abort_if_embed_server_running

        abort_if_embed_server_running()

    # Docling VRAM preflight — always on. Orthogonal to the embed_server
    # socket check above: that one guards against our own embed_server;
    # this one guards against any other VRAM consumer (rerank_server etc.).
    from research_kb_common.gpu_guard import (
        DEFAULT_DOCLING_VRAM_MIN_MB,
        MINERU_MANAGED_SERVICES,
        abort_if_vram_insufficient,
        restart_services,
    )

    _stopped_services = abort_if_vram_insufficient(
        min_mib=DEFAULT_DOCLING_VRAM_MIN_MB,
        managed_services=MINERU_MANAGED_SERVICES,
        auto_stop_services=args.auto_stop_services,
        caller_label="docling preflight (ingest_missing_textbooks)",
    )

    # Configure logging based on quiet mode
    if quiet:
        configure_logging(level="ERROR")

    if args.directory:
        textbooks_dir = Path(args.directory)
    else:
        textbooks_dir = Path(__file__).parent.parent / "fixtures" / "textbooks"

    if not textbooks_dir.exists():
        if json_output:
            print(json_module.dumps({"error": f"{textbooks_dir} does not exist"}))
        else:
            print(f"Error: {textbooks_dir} does not exist")
        return

    # Get all PDFs recursively
    all_pdfs = list(textbooks_dir.rglob("*.pdf"))
    if not quiet:
        print(f"Found {len(all_pdfs)} PDFs in {textbooks_dir}")

    # Initialize database connection pool
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Check which are already ingested and filter skipped
    to_ingest = []
    already_ingested = 0
    skipped = 0

    for pdf_path in all_pdfs:
        # Skip auxiliary files
        if any(skip in pdf_path.name for skip in SKIP_PATTERNS):
            skipped += 1
            if not quiet:
                logger.info("skipping_auxiliary", path=pdf_path.name)
            continue

        file_hash = compute_file_hash(str(pdf_path))
        existing = await SourceStore.get_by_file_hash(file_hash)

        if existing:
            already_ingested += 1
            if not quiet:
                logger.info("already_ingested", path=pdf_path.name, title=existing.title)
        else:
            to_ingest.append(pdf_path)

    if not quiet:
        print(f"Already ingested: {already_ingested}")
        print(f"Skipped (auxiliary): {skipped}")
        print(f"To ingest: {len(to_ingest)}")

    if not to_ingest:
        if json_output:
            print(
                json_module.dumps(
                    {
                        "success_count": 0,
                        "failed_count": 0,
                        "total_chunks": 0,
                        "already_ingested": already_ingested,
                        "skipped": skipped,
                        "failed_files": [],
                    }
                )
            )
        elif not quiet:
            print("Nothing to ingest!")
        return

    # Ingest missing textbooks
    results = {"success": [], "failed": []}

    for i, pdf_path in enumerate(to_ingest):
        if not quiet and not json_output:
            print(f"\n[{i+1}/{len(to_ingest)}] Processing: {pdf_path.name}")

        # Try sidecar metadata first, fall back to filename parsing
        meta = load_sidecar_metadata(pdf_path)
        if not meta or not meta.get("title"):
            meta = parse_filename_for_metadata(pdf_path.name)

        domain_id = infer_domain(pdf_path.name, meta, full_path=pdf_path)
        start_time = time.time()

        try:
            source_id, num_chunks, num_headings = await ingest_textbook(
                pdf_path=str(pdf_path),
                title=meta["title"],
                authors=meta["authors"],
                year=meta["year"],
                domain_id=domain_id,
                metadata={"auto_ingested": True, "source": "missing_textbooks_script"},
                quiet=quiet,
                no_embed=no_embed,
            )

            elapsed = time.time() - start_time

            results["success"].append(
                {
                    "file": pdf_path.name,
                    "title": meta["title"],
                    "chunks": num_chunks,
                    "elapsed_seconds": round(elapsed, 1),
                }
            )

            # Single-line progress in quiet mode
            if quiet and not json_output:
                print(f"✓ {pdf_path.name}: {num_chunks} chunks ({elapsed:.0f}s)")
            elif not quiet and not json_output:
                print(f"  ✓ {num_chunks} chunks created")

        except (MemoryError, OSError) as e:
            # System-level errors: not recoverable
            error_type = "memory_exhausted" if isinstance(e, MemoryError) else "file_io_error"
            if not json_output:
                logger.error(
                    "system_error",
                    file=pdf_path.name,
                    error_type=error_type,
                    error=str(e),
                )
            results["failed"].append(
                {
                    "file": pdf_path.name,
                    "error": f"{error_type}: {str(e)[:100]}",
                    "recoverable": False,
                }
            )
            if quiet and not json_output:
                print(f"✗ {pdf_path.name}: {error_type}")
            elif not json_output:
                print(f"  ✗ System error: {error_type}")

        except (EmbeddingError, ConnectionError) as e:
            # Embedding service errors: recoverable (retry later)
            if not json_output:
                logger.error("embedding_service_failure", file=pdf_path.name, error=str(e))
            results["failed"].append(
                {
                    "file": pdf_path.name,
                    "error": "Embedding service failure (retries exhausted)",
                    "recoverable": True,
                }
            )
            if quiet and not json_output:
                print(f"✗ {pdf_path.name}: embedding service failure (recoverable)")
            elif not json_output:
                print("  ✗ Embedding service failure (retry ingestion later)")

        except StorageError as e:
            # Database errors: recoverable
            if not json_output:
                logger.error("storage_error", file=pdf_path.name, error=str(e))
            results["failed"].append(
                {
                    "file": pdf_path.name,
                    "error": f"Database error: {str(e)[:100]}",
                    "recoverable": True,
                }
            )
            if quiet and not json_output:
                print(f"✗ {pdf_path.name}: database error (recoverable)")
            elif not json_output:
                print("  ✗ Database error")

        except Exception as e:
            # Unknown errors: not recoverable
            if not json_output:
                logger.error("ingestion_failed", file=pdf_path.name, error=str(e), exc_info=True)
            results["failed"].append(
                {
                    "file": pdf_path.name,
                    "error": str(e)[:100],
                    "recoverable": False,
                }
            )
            if quiet and not json_output:
                print(f"✗ {pdf_path.name}: {str(e)[:60]}")
            elif not json_output:
                print(f"  ✗ Failed: {e}")

    # Summary
    total_chunks = sum(r["chunks"] for r in results["success"])

    if json_output:
        # JSON summary for programmatic parsing
        summary = {
            "success_count": len(results["success"]),
            "failed_count": len(results["failed"]),
            "total_chunks": total_chunks,
            "already_ingested": already_ingested,
            "skipped": skipped,
            "failed_files": [
                {
                    "file": r["file"],
                    "error": r["error"],
                    "recoverable": r.get("recoverable", False),
                }
                for r in results["failed"]
            ],
        }
        print(json_module.dumps(summary, indent=2))
    elif quiet:
        # Minimal summary (already printed single-line per PDF)
        print(f"\nIngested: {len(results['success'])} textbooks | {total_chunks} chunks")
        if results["failed"]:
            recoverable = sum(1 for r in results["failed"] if r.get("recoverable", False))
            print(f"Failed: {len(results['failed'])} ({recoverable} recoverable)")
    else:
        # Full summary
        print("\n" + "=" * 70)
        print("INGESTION SUMMARY")
        print("=" * 70)
        print(f"Success: {len(results['success'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Total new chunks: {total_chunks}")

        if results["failed"]:
            print("\nFailed files:")
            for r in results["failed"]:
                status = "(recoverable)" if r.get("recoverable", False) else "(not recoverable)"
                print(f"  - {r['file']}: {r['error'][:60]} {status}")

    # Restart any services the VRAM preflight stopped at startup.
    if _stopped_services:
        if not quiet:
            print(f"  [preflight] Restarting: {', '.join(_stopped_services)}")
        restart_services(_stopped_services)

    await close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())
