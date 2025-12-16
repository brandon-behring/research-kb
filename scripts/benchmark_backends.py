#!/usr/bin/env python3
"""Benchmark comparison between extraction backends.

Compares Ollama vs LlamaCpp on the same set of chunks to measure:
- Throughput (chunks/min)
- Latency (p50, p95)
- Quality (concepts extracted, validation failures)

Usage:
    # Compare all backends on 20 random chunks
    python scripts/benchmark_backends.py --chunks 20

    # Compare specific backends
    python scripts/benchmark_backends.py --backends ollama llamacpp --chunks 20

    # Save results
    python scripts/benchmark_backends.py --output benchmark_results.json
"""

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "extraction" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_contracts import Chunk
from research_kb_extraction import ExtractionMetrics, get_llm_client
from research_kb_extraction.prompts import SYSTEM_PROMPT, format_extraction_prompt
from research_kb_storage import ChunkStore, get_connection_pool


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single backend."""
    backend: str
    chunks_processed: int
    successful: int
    failed: int
    empty: int
    total_concepts: int
    total_relationships: int
    duration_seconds: float
    throughput_per_min: float
    latency_p50_ms: float
    latency_p95_ms: float
    latencies_ms: list[float]


async def benchmark_backend(
    backend: str,
    chunks: list[Chunk],
    model: Optional[str] = None,
) -> BenchmarkResult:
    """Benchmark a single backend on given chunks."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {backend}")
    print(f"{'='*60}")

    # Initialize client
    kwargs = {"temperature": 0.1}
    if backend == "ollama":
        kwargs["num_ctx"] = 2048
    elif backend == "llamacpp":
        kwargs["n_ctx"] = 2048
        kwargs["n_gpu_layers"] = 20  # Fits in 8GB VRAM with desktop overhead

    try:
        client = get_llm_client(backend=backend, model=model, **kwargs)
    except Exception as e:
        print(f"  ✗ Failed to create client: {e}")
        return BenchmarkResult(
            backend=backend,
            chunks_processed=0,
            successful=0,
            failed=len(chunks),
            empty=0,
            total_concepts=0,
            total_relationships=0,
            duration_seconds=0,
            throughput_per_min=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latencies_ms=[],
        )

    # Check availability
    if not await client.is_available():
        print(f"  ✗ Backend not available")
        return BenchmarkResult(
            backend=backend,
            chunks_processed=0,
            successful=0,
            failed=len(chunks),
            empty=0,
            total_concepts=0,
            total_relationships=0,
            duration_seconds=0,
            throughput_per_min=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latencies_ms=[],
        )

    metrics = ExtractionMetrics(backend=client.extraction_method)
    start_time = time.time()

    try:
        for i, chunk in enumerate(chunks):
            chunk_start = time.perf_counter()
            print(f"\r  Processing chunk {i+1}/{len(chunks)}...", end="", flush=True)

            try:
                extraction = await client.extract_concepts(chunk.content)
                latency_ms = (time.perf_counter() - chunk_start) * 1000

                if extraction.concept_count == 0:
                    metrics.record_empty(latency_ms=latency_ms)
                else:
                    metrics.record_success(
                        concepts=extraction.concept_count,
                        relationships=extraction.relationship_count,
                        latency_ms=latency_ms,
                    )
            except json.JSONDecodeError:
                latency_ms = (time.perf_counter() - chunk_start) * 1000
                metrics.record_json_failure(latency_ms=latency_ms)
            except Exception as e:
                latency_ms = (time.perf_counter() - chunk_start) * 1000
                metrics.record_validation_failure(latency_ms=latency_ms)
                print(f"\n  ✗ Chunk {i+1} failed: {e}")

        print()  # Newline after progress

    finally:
        await client.close()

    duration = time.time() - start_time

    # Calculate results
    latencies = metrics.latencies_ms
    p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0
    p95_idx = int(len(latencies) * 0.95) if latencies else 0
    p95 = sorted(latencies)[p95_idx] if latencies else 0

    result = BenchmarkResult(
        backend=client.extraction_method,
        chunks_processed=metrics.total_chunks,
        successful=metrics.successful,
        failed=metrics.validation_failures + metrics.json_parse_failures,
        empty=metrics.empty_extractions,
        total_concepts=metrics.total_concepts,
        total_relationships=metrics.total_relationships,
        duration_seconds=duration,
        throughput_per_min=(metrics.total_chunks / duration) * 60 if duration > 0 else 0,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latencies_ms=latencies,
    )

    # Print summary
    print(f"\n  Results for {result.backend}:")
    print(f"    Throughput: {result.throughput_per_min:.2f} chunks/min")
    print(f"    Latency p50: {result.latency_p50_ms:.0f}ms, p95: {result.latency_p95_ms:.0f}ms")
    print(f"    Success: {result.successful}/{result.chunks_processed} ({100*result.successful/max(1, result.chunks_processed):.0f}%)")
    print(f"    Concepts: {result.total_concepts}, Relationships: {result.total_relationships}")

    return result


async def main():
    parser = argparse.ArgumentParser(description="Benchmark extraction backends")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["ollama", "instructor"],
        choices=["ollama", "instructor", "llamacpp", "anthropic"],
        help="Backends to benchmark"
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=20,
        help="Number of chunks to benchmark (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chunk selection"
    )

    args = parser.parse_args()

    # Initialize database
    await get_connection_pool()

    # Get random chunks
    print(f"Loading {args.chunks} random chunks...")
    random.seed(args.seed)
    all_chunks = await ChunkStore.list_all(limit=10000)
    valid_chunks = [c for c in all_chunks if len(c.content) >= 200]
    selected = random.sample(valid_chunks, min(args.chunks, len(valid_chunks)))
    print(f"Selected {len(selected)} chunks (seed={args.seed})")

    # Run benchmarks
    results = []
    for backend in args.backends:
        result = await benchmark_backend(backend, selected)
        results.append(result)

    # Print comparison
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"{'Backend':<30} {'Throughput':<15} {'Latency p50':<12} {'Success':<10}")
    print("-" * 60)

    for r in results:
        if r.chunks_processed > 0:
            success_pct = f"{100*r.successful/r.chunks_processed:.0f}%"
            print(f"{r.backend:<30} {r.throughput_per_min:.2f}/min       {r.latency_p50_ms:.0f}ms        {success_pct}")
        else:
            print(f"{r.backend:<30} {'N/A':<15} {'N/A':<12} {'0%':<10}")

    # Find fastest
    valid_results = [r for r in results if r.throughput_per_min > 0]
    if len(valid_results) >= 2:
        sorted_results = sorted(valid_results, key=lambda r: r.throughput_per_min, reverse=True)
        fastest = sorted_results[0]
        second = sorted_results[1]
        speedup = fastest.throughput_per_min / second.throughput_per_min if second.throughput_per_min > 0 else 0
        print(f"\n★ Fastest: {fastest.backend} ({speedup:.2f}x faster than {second.backend})")

    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "chunks_tested": len(selected),
            "seed": args.seed,
            "results": [
                {k: v for k, v in asdict(r).items() if k != "latencies_ms"}
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
