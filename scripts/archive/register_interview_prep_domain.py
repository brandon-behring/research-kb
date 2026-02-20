#!/usr/bin/env python3
"""Register the interview_prep domain in research-kb.

Run this once before ingesting interview prep content.

Usage:
    python scripts/register_interview_prep_domain.py
"""

import asyncio
import sys
from pathlib import Path

# Add packages to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_storage import DatabaseConfig, get_connection_pool, close_connection_pool
from research_kb_storage.domain_store import DomainStore

logger = get_logger(__name__)


async def register_domain() -> None:
    """Register the interview_prep domain."""
    config = DatabaseConfig()
    await get_connection_pool(config)

    try:
        # Check if domain already exists
        existing = await DomainStore.get_by_id("interview_prep")
        if existing:
            logger.info("domain_already_exists", domain_id="interview_prep")
            print("Domain 'interview_prep' already registered.")
            return

        # Register new domain
        domain = await DomainStore.create(
            domain_id="interview_prep",
            name="Interview Preparation",
            description="ML/DS interview questions, solutions, and study materials from interview_prep_series",
            concept_types=[
                "problem",
                "solution",
                "definition",
                "technique",
                "redflag",
                "principle",
            ],
            relationship_types=["REQUIRES", "USES", "RELATED_TO", "EXTENDS"],
            default_fts_weight=0.4,
            default_vector_weight=0.6,
            default_graph_weight=0.1,
            default_citation_weight=0.0,  # No citations for interview content
        )

        logger.info("domain_registered", domain_id=domain.id, name=domain.name)
        print(f"Domain '{domain.id}' registered successfully.")
        print(f"  Name: {domain.name}")
        print(f"  Description: {domain.description}")
        print(f"  Concept types: {domain.concept_types}")

    finally:
        await close_connection_pool()


if __name__ == "__main__":
    asyncio.run(register_domain())
