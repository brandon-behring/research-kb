#!/usr/bin/env python3
"""Compare extraction quality across Haiku model versions.

Runs the same 100 chunks through Haiku 3, 3.5, and 4.5 to compare:
- Concept count
- Relationship count
- Failure rate
- Qualitative differences
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "extraction" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

import asyncpg


@dataclass
class ModelResults:
    """Results for a single model."""
    model: str
    total_chunks: int = 0
    successful: int = 0
    failed: int = 0
    total_concepts: int = 0
    total_relationships: int = 0
    latencies: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    extractions: dict = field(default_factory=dict)  # chunk_id -> extraction

    @property
    def avg_concepts(self) -> float:
        return self.total_concepts / max(self.successful, 1)

    @property
    def avg_relationships(self) -> float:
        return self.total_relationships / max(self.successful, 1)

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / max(len(self.latencies), 1)

    @property
    def failure_rate(self) -> float:
        return self.failed / max(self.total_chunks, 1) * 100


async def load_test_chunks(chunk_ids_file: str) -> list[tuple[str, str]]:
    """Load chunk IDs and content from database."""
    with open(chunk_ids_file) as f:
        chunk_ids = [line.strip() for line in f if line.strip()]

    conn = await asyncpg.connect(
        'postgresql://postgres:postgres@localhost:5432/research_kb'
    )

    chunks = []
    for chunk_id in chunk_ids:
        row = await conn.fetchrow(
            "SELECT id, content FROM chunks WHERE id = $1",
            chunk_id
        )
        if row:
            chunks.append((str(row['id']), row['content']))

    await conn.close()
    return chunks


async def run_extraction(model: str, chunks: list[tuple[str, str]], results: ModelResults):
    """Run extraction for a single model."""
    from research_kb_extraction.anthropic_client import AnthropicClient

    client = AnthropicClient(model=model)

    for chunk_id, content in chunks:
        results.total_chunks += 1
        start = time.time()

        try:
            extraction = await client.extract_concepts(content)
            latency = time.time() - start

            results.successful += 1
            results.total_concepts += extraction.concept_count
            results.total_relationships += extraction.relationship_count
            results.latencies.append(latency)
            results.extractions[chunk_id] = {
                'concepts': [c.model_dump() for c in extraction.concepts],
                'relationships': [r.model_dump() for r in extraction.relationships],
            }

            # Progress indicator
            if results.total_chunks % 10 == 0:
                print(f"  {model}: {results.total_chunks}/{len(chunks)} chunks processed")

        except Exception as e:
            results.failed += 1
            results.errors.append(str(e))
            print(f"  {model}: Error on chunk {chunk_id[:8]}: {e}")

    await client.close()


def print_comparison(all_results: dict[str, ModelResults]):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)

    # Summary table
    print("\n### Summary Statistics\n")
    print(f"{'Model':<12} {'Success':<10} {'Fail%':<8} {'Concepts':<12} {'Rels':<10} {'Latency':<10}")
    print("-" * 62)

    for model, results in all_results.items():
        print(f"{model:<12} {results.successful:<10} {results.failure_rate:<8.1f} "
              f"{results.avg_concepts:<12.1f} {results.avg_relationships:<10.1f} "
              f"{results.avg_latency:<10.2f}s")

    # Concept count comparison
    print("\n### Per-Chunk Comparison (first 10 chunks)\n")
    chunk_ids = list(list(all_results.values())[0].extractions.keys())[:10]

    print(f"{'Chunk':<10} ", end="")
    for model in all_results.keys():
        print(f"{model:<15} ", end="")
    print()
    print("-" * 60)

    for chunk_id in chunk_ids:
        print(f"{chunk_id[:8]:<10} ", end="")
        for model, results in all_results.items():
            if chunk_id in results.extractions:
                ext = results.extractions[chunk_id]
                c = len(ext['concepts'])
                r = len(ext['relationships'])
                print(f"{c}c/{r}r{'':>8} ", end="")
            else:
                print(f"{'FAIL':<15} ", end="")
        print()

    # Sample extraction comparison
    print("\n### Sample Extraction (first chunk with concepts)\n")
    for chunk_id in chunk_ids:
        has_concepts = False
        for results in all_results.values():
            if chunk_id in results.extractions:
                if len(results.extractions[chunk_id]['concepts']) > 0:
                    has_concepts = True
                    break

        if has_concepts:
            print(f"Chunk: {chunk_id}")
            for model, results in all_results.items():
                if chunk_id in results.extractions:
                    concepts = results.extractions[chunk_id]['concepts']
                    concept_names = [c['name'] for c in concepts[:5]]
                    print(f"  {model}: {concept_names}")
            break


async def main():
    """Run comparison test."""
    print("=" * 70)
    print("HAIKU MODEL COMPARISON TEST")
    print("=" * 70)

    # Load chunks
    print("\nLoading test chunks...")
    chunks = await load_test_chunks("/tmp/test_chunks.txt")
    print(f"Loaded {len(chunks)} chunks")

    # Models to test
    models = ["haiku", "haiku-3.5", "haiku-4.5"]
    all_results = {}

    for model in models:
        print(f"\n--- Testing {model} ---")
        results = ModelResults(model=model)
        await run_extraction(model, chunks, results)
        all_results[model] = results

        # Save intermediate results
        with open(f"/tmp/haiku_test_{model.replace('.', '_')}.json", 'w') as f:
            json.dump({
                'model': model,
                'successful': results.successful,
                'failed': results.failed,
                'total_concepts': results.total_concepts,
                'total_relationships': results.total_relationships,
                'avg_latency': results.avg_latency,
                'extractions': results.extractions,
            }, f, indent=2)

    # Print comparison
    print_comparison(all_results)

    # Save final comparison
    with open("/tmp/haiku_comparison.json", 'w') as f:
        summary = {
            model: {
                'successful': r.successful,
                'failed': r.failed,
                'failure_rate': r.failure_rate,
                'avg_concepts': r.avg_concepts,
                'avg_relationships': r.avg_relationships,
                'avg_latency': r.avg_latency,
            }
            for model, r in all_results.items()
        }
        json.dump(summary, f, indent=2)

    print("\n✓ Results saved to /tmp/haiku_comparison.json")


if __name__ == "__main__":
    asyncio.run(main())
