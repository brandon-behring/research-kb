#!/usr/bin/env python3
"""Process method/assumption enrichment batch results.

Fetches completed batch results and populates methods/assumptions tables
with inferred attributes and provenance data.

Usage:
    python scripts/process_method_enrichment.py
    python scripts/process_method_enrichment.py --batch-id <id>
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

import asyncpg

from research_kb_common import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("<PROJECT_ROOT>/batch_results/enrichment")


def parse_llm_response(content: str) -> Optional[dict]:
    """Parse JSON response from LLM, handling markdown code blocks."""
    try:
        # Try direct JSON parse
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end > start:
            try:
                return json.loads(content[start:end].strip())
            except json.JSONDecodeError:
                pass

    if "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        if end > start:
            try:
                return json.loads(content[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Try finding JSON object in content
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

    return None


async def process_method_result(
    conn,
    concept_id: UUID,
    data: dict,
    evidence_chunk_id: Optional[str],
) -> bool:
    """Insert or update method record with enriched attributes."""
    try:
        evidence_ids = [UUID(evidence_chunk_id)] if evidence_chunk_id else []
        confidence = data.get("confidence", 0.5)

        # Check if method already exists
        existing = await conn.fetchval(
            "SELECT id FROM methods WHERE concept_id = $1",
            concept_id,
        )

        if existing:
            # Update existing
            await conn.execute(
                """
                UPDATE methods SET
                    required_assumptions = $2,
                    problem_types = $3,
                    common_estimators = $4,
                    evidence_chunk_ids = $5,
                    inference_confidence = $6,
                    enriched_at = $7
                WHERE concept_id = $1
            """,
                concept_id,
                data.get("required_assumptions", []),
                data.get("problem_types", []),
                data.get("common_estimators", []),
                evidence_ids,
                confidence,
                datetime.now(timezone.utc),
            )
        else:
            # Insert new
            await conn.execute(
                """
                INSERT INTO methods (
                    concept_id, required_assumptions, problem_types,
                    common_estimators, evidence_chunk_ids,
                    inference_confidence, enriched_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                concept_id,
                data.get("required_assumptions", []),
                data.get("problem_types", []),
                data.get("common_estimators", []),
                evidence_ids,
                confidence,
                datetime.now(timezone.utc),
            )

        return True
    except Exception as e:
        logger.error("method_insert_failed", concept_id=str(concept_id), error=str(e))
        return False


async def process_assumption_result(
    conn,
    concept_id: UUID,
    data: dict,
    evidence_chunk_id: Optional[str],
) -> bool:
    """Insert or update assumption record with enriched attributes."""
    try:
        evidence_ids = [UUID(evidence_chunk_id)] if evidence_chunk_id else []
        confidence = data.get("confidence", 0.5)

        # Handle violation_consequences - LLM sometimes returns list, schema expects TEXT
        violation = data.get("violation_consequences")
        if isinstance(violation, list):
            violation = "; ".join(str(v) for v in violation)

        # Check if assumption already exists
        existing = await conn.fetchval(
            "SELECT id FROM assumptions WHERE concept_id = $1",
            concept_id,
        )

        if existing:
            # Update existing
            await conn.execute(
                """
                UPDATE assumptions SET
                    mathematical_statement = $2,
                    is_testable = $3,
                    common_tests = $4,
                    violation_consequences = $5,
                    evidence_chunk_ids = $6,
                    inference_confidence = $7,
                    enriched_at = $8
                WHERE concept_id = $1
            """,
                concept_id,
                data.get("mathematical_statement"),
                data.get("is_testable"),
                data.get("common_tests", []),
                violation,
                evidence_ids,
                confidence,
                datetime.now(timezone.utc),
            )
        else:
            # Insert new
            await conn.execute(
                """
                INSERT INTO assumptions (
                    concept_id, mathematical_statement, is_testable,
                    common_tests, violation_consequences, evidence_chunk_ids,
                    inference_confidence, enriched_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                concept_id,
                data.get("mathematical_statement"),
                data.get("is_testable"),
                data.get("common_tests", []),
                violation,
                evidence_ids,
                confidence,
                datetime.now(timezone.utc),
            )

        return True
    except Exception as e:
        logger.error("assumption_insert_failed", concept_id=str(concept_id), error=str(e))
        return False


async def fetch_batch_results(client, batch_id: str) -> list[dict]:
    """Fetch results from a completed batch."""
    results = []
    for result in client.messages.batches.results(batch_id):
        results.append(
            {
                "custom_id": result.custom_id,
                "result": result.result if hasattr(result, "result") else None,
            }
        )
    return results


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process enrichment batch results")
    parser.add_argument("--batch-id", help="Specific batch ID to process")
    parser.add_argument("--info-file", help="Info file with batch IDs")
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Get batch IDs to process
    batch_ids = []
    if args.batch_id:
        batch_ids = [args.batch_id]
    elif args.info_file:
        with open(args.info_file) as f:
            info = json.load(f)
            batch_ids = info.get("batch_ids", [])
    else:
        # Find most recent info file
        info_files = sorted(OUTPUT_DIR.glob("enrichment_*_info.json"))
        if info_files:
            with open(info_files[-1]) as f:
                info = json.load(f)
                batch_ids = info.get("batch_ids", [])
                print(f"Using batch IDs from: {info_files[-1]}")

    if not batch_ids:
        print("ERROR: No batch IDs found. Use --batch-id or --info-file")
        return

    # Connect to database
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        database="research_kb",
    )

    print("=" * 60)
    print("Process Method/Assumption Enrichment Results")
    print("=" * 60)

    total_methods = 0
    total_assumptions = 0
    total_errors = 0

    for batch_id in batch_ids:
        print(f"\nProcessing batch: {batch_id}")

        # Check batch status
        batch = client.messages.batches.retrieve(batch_id)
        print(f"  Status: {batch.processing_status}")

        if batch.processing_status != "ended":
            print("  ⏳ Batch not complete yet")
            continue

        print(f"  Succeeded: {batch.request_counts.succeeded}")
        print(f"  Failed: {batch.request_counts.errored}")

        # Fetch and process results
        method_count = 0
        assumption_count = 0
        error_count = 0

        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id

            # Parse concept type and ID (format: type_uuid_with_underscores)
            if "_" not in custom_id:
                continue

            # Split on first underscore to get type, rest is UUID with underscores
            concept_type, uuid_part = custom_id.split("_", 1)
            # Convert underscores back to hyphens for UUID
            concept_id_str = uuid_part.replace("_", "-")
            try:
                concept_id = UUID(concept_id_str)
            except ValueError:
                error_count += 1
                continue

            # Get the message content
            if result.result.type != "succeeded":
                error_count += 1
                continue

            message = result.result.message
            if not message.content:
                error_count += 1
                continue

            # Extract text content
            text_content = ""
            for block in message.content:
                if hasattr(block, "text"):
                    text_content = block.text
                    break

            # Parse JSON response
            data = parse_llm_response(text_content)
            if not data:
                logger.warning("json_parse_failed", custom_id=custom_id)
                error_count += 1
                continue

            # Get evidence chunk ID from concept
            evidence_chunk_id = await conn.fetchval(
                """
                SELECT chunk_id::text
                FROM chunk_concepts
                WHERE concept_id = $1
                ORDER BY relevance_score DESC NULLS LAST
                LIMIT 1
            """,
                concept_id,
            )

            # Process based on type
            if concept_type == "method":
                if await process_method_result(conn, concept_id, data, evidence_chunk_id):
                    method_count += 1
                else:
                    error_count += 1
            elif concept_type == "assumption":
                if await process_assumption_result(conn, concept_id, data, evidence_chunk_id):
                    assumption_count += 1
                else:
                    error_count += 1

        print(f"  Methods enriched: {method_count}")
        print(f"  Assumptions enriched: {assumption_count}")
        print(f"  Errors: {error_count}")

        total_methods += method_count
        total_assumptions += assumption_count
        total_errors += error_count

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total methods enriched: {total_methods}")
    print(f"Total assumptions enriched: {total_assumptions}")
    print(f"Total errors: {total_errors}")

    # Verify in database
    method_count = await conn.fetchval("SELECT COUNT(*) FROM methods")
    assumption_count = await conn.fetchval("SELECT COUNT(*) FROM assumptions")
    print("\nDatabase state:")
    print(f"  Methods table: {method_count} rows")
    print(f"  Assumptions table: {assumption_count} rows")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
