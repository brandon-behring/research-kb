#!/usr/bin/env python3
"""Ingest interview prep cards into research-kb.

Ingests YAML cards from interview_prep_series into the interview_prep domain.
Creates one source per volume/directory, with each card becoming a chunk.

Usage:
    # Register domain first (one-time)
    python scripts/register_interview_prep_domain.py

    # Ingest all volumes
    python scripts/ingest_interview_prep.py --all

    # Ingest specific volume
    python scripts/ingest_interview_prep.py --volume vol2_causal_inference

    # Quiet mode for monitoring
    python scripts/ingest_interview_prep.py --all --quiet
"""

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml

# Add packages to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_contracts import SourceType
from research_kb_pdf import EmbeddingClient
from research_kb_storage import (
    ChunkStore,
    DatabaseConfig,
    SourceStore,
    get_connection_pool,
    close_connection_pool,
)

logger = get_logger(__name__)

# Default location of interview prep series
INTERVIEW_PREP_DIR = Path.home() / "Claude" / "interview_prep_series"
DOMAIN_ID = "interview_prep"


def sha256_content(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_dir(cards_dir: Path) -> str:
    """Compute SHA-256 hash of all YAML files in a directory."""
    hasher = hashlib.sha256()
    for yml_file in sorted(cards_dir.glob("*.yml")):
        with open(yml_file, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def load_cards_from_yaml(yml_path: Path) -> list[dict[str, Any]]:
    """Load cards from a YAML file."""
    try:
        with open(yml_path) as f:
            data = yaml.safe_load(f)
        if data and "cards" in data:
            return data["cards"]
        return []
    except Exception as e:
        logger.warning("yaml_load_failed", file=str(yml_path), error=str(e))
        return []


def format_card_content(card: dict[str, Any]) -> str:
    """Format a card's front/back as searchable content.

    Format: "Q: {front}\n\nA: {back}"
    This creates good embedding representation for Q&A retrieval.
    """
    front = card.get("front", "").strip()
    back = card.get("back", "").strip()
    return f"Q: {front}\n\nA: {back}"


async def ingest_volume(
    volume_dir: Path,
    embedding_client: EmbeddingClient,
    quiet: bool = False,
) -> tuple[str | None, int]:
    """Ingest all cards from one volume.

    Args:
        volume_dir: Path to volume directory (must have cards/ subdirectory)
        embedding_client: Client for generating embeddings
        quiet: If True, minimal output

    Returns:
        Tuple of (source_id, card_count) or (None, 0) if already ingested/failed
    """
    volume_name = volume_dir.name
    cards_dir = volume_dir / "cards"

    if not cards_dir.exists():
        if not quiet:
            logger.warning("no_cards_dir", volume=volume_name)
        return None, 0

    # Collect all cards from YAML files
    card_files = sorted(cards_dir.glob("*.yml"))
    if not card_files:
        if not quiet:
            logger.warning("no_card_files", volume=volume_name)
        return None, 0

    all_cards = []
    for card_file in card_files:
        cards = load_cards_from_yaml(card_file)
        all_cards.extend(cards)

    if not all_cards:
        if not quiet:
            logger.warning("no_cards_loaded", volume=volume_name, files=len(card_files))
        return None, 0

    # Compute directory hash for deduplication
    dir_hash = sha256_dir(cards_dir)

    # Check if already ingested
    existing = await SourceStore.get_by_file_hash(dir_hash)
    if existing:
        if not quiet:
            logger.info("already_ingested", volume=volume_name, source_id=str(existing.id))
        return None, 0

    if not quiet:
        logger.info(
            "ingesting_volume",
            volume=volume_name,
            cards=len(all_cards),
            files=len(card_files),
        )

    # Create source record
    source = await SourceStore.create(
        source_type=SourceType.CODE_REPO,  # Using CODE_REPO for YAML content
        title=f"Interview Prep: {volume_name}",
        file_hash=dir_hash,
        file_path=str(cards_dir),
        domain_id=DOMAIN_ID,  # Explicit domain assignment
        metadata={
            "domain": DOMAIN_ID,
            "volume": volume_name,
            "card_count": len(all_cards),
            "card_files": len(card_files),
            "source_type": "interview_cards",
        },
    )

    # Create chunks from cards
    chunks_created = 0
    for i, card in enumerate(all_cards):
        content = format_card_content(card)

        # Skip empty cards
        if len(content.strip()) < 10:
            continue

        # Sanitize content
        sanitized = content.replace("\x00", "").replace("\uFFFD", "")

        # Generate embedding
        try:
            embedding = embedding_client.embed(sanitized)
        except Exception as e:
            logger.warning("embedding_failed", card_id=card.get("id", "unknown"), error=str(e))
            continue

        # Calculate content hash
        content_hash = sha256_content(sanitized)

        # Extract metadata
        metadata = {
            "card_id": card.get("id"),
            "card_type": card.get("type"),
            "los_id": card.get("los_id"),
            "company_tags": card.get("company_tags", []),
            "tags": card.get("tags", []),
            "source_file": card.get("source"),
        }

        # Create chunk record
        await ChunkStore.create(
            source_id=source.id,
            content=sanitized,
            content_hash=content_hash,
            embedding=embedding,
            domain_id=DOMAIN_ID,
            metadata=metadata,
        )
        chunks_created += 1

        # Progress logging
        if not quiet and (i + 1) % 100 == 0:
            logger.info("progress", volume=volume_name, processed=i + 1, total=len(all_cards))

    if not quiet:
        logger.info("volume_complete", volume=volume_name, chunks=chunks_created)

    return str(source.id), chunks_created


def find_volume_dirs(base_dir: Path) -> list[Path]:
    """Find all directories containing cards/ subdirectory."""
    volumes = []

    # Check direct volume directories (vol*, behavioral, company_prep)
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and (item / "cards").exists():
            volumes.append(item)

    return volumes


async def main():
    parser = argparse.ArgumentParser(description="Ingest interview prep cards into research-kb")
    parser.add_argument("--all", action="store_true", help="Ingest all volumes")
    parser.add_argument("--volume", type=str, help="Ingest specific volume by name")
    parser.add_argument("--base-dir", type=Path, default=INTERVIEW_PREP_DIR, help="Base directory")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--json", action="store_true", help="JSON output format")
    args = parser.parse_args()

    if not args.all and not args.volume:
        parser.error("Must specify --all or --volume")

    if not args.base_dir.exists():
        logger.error("base_dir_not_found", path=str(args.base_dir))
        sys.exit(1)

    # Initialize database
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Initialize embedding client
    embedding_client = EmbeddingClient()

    results = []

    try:
        if args.all:
            volumes = find_volume_dirs(args.base_dir)
            if not volumes:
                logger.error("no_volumes_found", base_dir=str(args.base_dir))
                sys.exit(1)

            if not args.quiet:
                logger.info("found_volumes", count=len(volumes))

            total_cards = 0
            for volume_dir in volumes:
                source_id, card_count = await ingest_volume(
                    volume_dir, embedding_client, quiet=args.quiet
                )
                if source_id:
                    results.append(
                        {
                            "volume": volume_dir.name,
                            "source_id": source_id,
                            "cards": card_count,
                        }
                    )
                    total_cards += card_count

            if args.json:
                import json

                print(json.dumps({"volumes": results, "total_cards": total_cards}, indent=2))
            elif not args.quiet:
                print(f"\n{'='*60}")
                print(f"Ingestion complete: {len(results)} volumes, {total_cards} cards")
                for r in results:
                    print(f"  {r['volume']}: {r['cards']} cards")

        else:
            volume_dir = args.base_dir / args.volume
            if not volume_dir.exists():
                logger.error("volume_not_found", volume=args.volume)
                sys.exit(1)

            source_id, card_count = await ingest_volume(
                volume_dir, embedding_client, quiet=args.quiet
            )

            if args.json:
                import json

                print(
                    json.dumps(
                        {
                            "volume": args.volume,
                            "source_id": source_id,
                            "cards": card_count,
                        }
                    )
                )
            elif not args.quiet:
                if source_id:
                    print(f"Ingested {args.volume}: {card_count} cards (source: {source_id})")
                else:
                    print(f"Volume {args.volume}: already ingested or no cards")

    finally:
        await close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())
