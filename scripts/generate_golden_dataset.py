#!/usr/bin/env python3
"""Generate candidate test cases for golden dataset using LLM.

Phase 2: Hybrid approach for golden dataset expansion.
- Step 1: Query concepts from database
- Step 2: Generate test case candidates using Ollama
- Step 3: User reviews and approves candidates
- Step 4: Merge approved cases into retrieval_test_cases.yaml

Usage:
    python scripts/generate_golden_dataset.py --limit 20 --output candidates.yaml
    python scripts/generate_golden_dataset.py --source-type textbook --limit 30
    python scripts/generate_golden_dataset.py --concepts-only  # Just show concepts
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))

from research_kb_storage import DatabaseConfig, get_connection_pool


@dataclass
class ConceptCandidate:
    """A concept that could become a test case."""

    name: str
    concept_type: str
    source_title: str
    source_type: str
    frequency: int  # How many chunks mention this concept


@dataclass
class TestCaseCandidate:
    """A candidate test case for review."""

    query: str
    expected_source_pattern: str
    expected_in_top_k: int
    expected_concepts: list[str]
    relevance_grade: int
    tags: list[str]
    notes: str


async def get_top_concepts(
    limit: int = 50,
    source_type: Optional[str] = None,
    concept_types: Optional[list[str]] = None,
) -> list[ConceptCandidate]:
    """Get most frequently mentioned concepts from the database.

    Args:
        limit: Maximum number of concepts to return
        source_type: Filter by source type (paper, textbook)
        concept_types: Filter by concept types (METHOD, ASSUMPTION, etc.)

    Returns:
        List of ConceptCandidate objects
    """
    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    # Build query with optional filters
    where_clauses = []
    params = []
    param_idx = 1

    if source_type:
        where_clauses.append(f"s.source_type = ${param_idx}")
        params.append(source_type)
        param_idx += 1

    if concept_types:
        where_clauses.append(f"c.concept_type = ANY(${param_idx}::text[])")
        params.append(concept_types)
        param_idx += 1

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
        WITH concept_stats AS (
            SELECT
                c.name,
                c.concept_type,
                s.title as source_title,
                s.source_type,
                COUNT(DISTINCT cc.chunk_id) as frequency
            FROM concepts c
            JOIN chunk_concepts cc ON c.id = cc.concept_id
            JOIN chunks ch ON cc.chunk_id = ch.id
            JOIN sources s ON ch.source_id = s.id
            WHERE {where_sql}
            GROUP BY c.id, c.name, c.concept_type, s.title, s.source_type
        )
        SELECT name, concept_type, source_title, source_type, frequency
        FROM concept_stats
        WHERE frequency >= 3  -- Minimum mentions
        ORDER BY frequency DESC
        LIMIT ${param_idx}
    """
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [
        ConceptCandidate(
            name=row["name"],
            concept_type=row["concept_type"],
            source_title=row["source_title"],
            source_type=row["source_type"],
            frequency=row["frequency"],
        )
        for row in rows
    ]


async def generate_test_case_with_ollama(
    concept: ConceptCandidate,
    ollama_url: str = "http://localhost:11434",
) -> Optional[TestCaseCandidate]:
    """Generate a test case candidate using Ollama.

    Args:
        concept: The concept to generate a test case for
        ollama_url: Ollama API endpoint

    Returns:
        TestCaseCandidate or None if generation fails
    """
    import httpx

    prompt = f"""Generate a search test case for a causal inference knowledge base.

Concept: {concept.name}
Type: {concept.concept_type}
Primary Source: {concept.source_title}

Create a JSON object with:
1. "query": A natural search query someone would use to find info about this concept (5-15 words)
2. "expected_concepts": 3-5 related concepts that should appear in results
3. "tags": 2-4 relevant tags (lowercase, use underscores)

Focus on practical, real-world search queries a researcher might type.

Respond with ONLY valid JSON, no explanation:
{{"query": "...", "expected_concepts": [...], "tags": [...]}}"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            response.raise_for_status()
            result = response.json()
            generated = json.loads(result.get("response", "{}"))

            # Build source pattern from source title
            title_words = concept.source_title.lower().split()[:3]
            source_pattern = "|".join(w for w in title_words if len(w) > 3)

            return TestCaseCandidate(
                query=generated.get("query", concept.name),
                expected_source_pattern=source_pattern or concept.name.lower(),
                expected_in_top_k=5,
                expected_concepts=generated.get("expected_concepts", [concept.name]),
                relevance_grade=3,
                tags=generated.get("tags", ["generated"]),
                notes=f"Generated from concept: {concept.name} ({concept.source_title})",
            )

    except Exception as e:
        print(f"  Warning: Failed to generate for {concept.name}: {e}")
        return None


def candidates_to_yaml(candidates: list[TestCaseCandidate]) -> str:
    """Convert test case candidates to YAML format."""
    test_cases = []
    for c in candidates:
        test_cases.append(
            {
                "query": c.query,
                "expected_source_pattern": c.expected_source_pattern,
                "expected_in_top_k": c.expected_in_top_k,
                "expected_concepts": c.expected_concepts,
                "relevance_grade": c.relevance_grade,
                "tags": c.tags,
                "notes": c.notes,
            }
        )

    return yaml.dump(
        {"test_cases": test_cases},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


async def main():
    parser = argparse.ArgumentParser(description="Generate candidate test cases for golden dataset")
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=20,
        help="Number of concepts to process (default: 20)",
    )
    parser.add_argument(
        "--source-type",
        "-s",
        choices=["paper", "textbook"],
        help="Filter by source type",
    )
    parser.add_argument(
        "--concept-type",
        "-c",
        action="append",
        choices=["METHOD", "ASSUMPTION", "PROBLEM", "DEFINITION", "THEOREM"],
        help="Filter by concept type (can specify multiple)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="candidates.yaml",
        help="Output file for candidates (default: candidates.yaml)",
    )
    parser.add_argument(
        "--concepts-only",
        action="store_true",
        help="Just list concepts without generating test cases",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
    args = parser.parse_args()

    print(f"Fetching top {args.limit} concepts from database...")

    concepts = await get_top_concepts(
        limit=args.limit,
        source_type=args.source_type,
        concept_types=args.concept_type,
    )

    if not concepts:
        print("No concepts found matching criteria.")
        return

    print(f"Found {len(concepts)} concepts\n")

    if args.concepts_only:
        print("Top Concepts (frequency = chunk mentions):\n")
        for c in concepts[:50]:
            print(f"  {c.name} ({c.concept_type})")
            print(f"    Source: {c.source_title} ({c.source_type})")
            print(f"    Mentions: {c.frequency}\n")
        return

    # Generate test cases
    print("Generating test case candidates with Ollama...")
    candidates = []

    for i, concept in enumerate(concepts):
        print(f"  [{i+1}/{len(concepts)}] {concept.name}...")
        candidate = await generate_test_case_with_ollama(concept, args.ollama_url)
        if candidate:
            candidates.append(candidate)
            print(f"    → {candidate.query}")

    if not candidates:
        print("\nNo candidates generated. Is Ollama running?")
        return

    # Write to file
    output_path = Path(args.output)
    yaml_content = candidates_to_yaml(candidates)

    with open(output_path, "w") as f:
        f.write("# Generated test case candidates\n")
        f.write("# Review and edit before merging into retrieval_test_cases.yaml\n")
        f.write(f"# Generated {len(candidates)} candidates from {len(concepts)} concepts\n\n")
        f.write(yaml_content)

    print(f"\n✓ Generated {len(candidates)} candidates")
    print(f"✓ Written to: {output_path}")
    print("\nNext steps:")
    print(f"  1. Review candidates in {output_path}")
    print("  2. Remove or edit low-quality entries")
    print("  3. Merge approved cases into fixtures/eval/retrieval_test_cases.yaml")


if __name__ == "__main__":
    asyncio.run(main())
