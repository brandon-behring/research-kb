#!/usr/bin/env python3
"""Batch concept extraction for the SSM subset (30 papers).

Runs the ExtractionPipeline over a fixed list of source UUIDs using the
Anthropic Haiku backend for speed. Prompt domain is "rag_llm" — closest
match for LLM-architecture papers in the built-in prompt set.

Written 2026-04-10 as B3 of the post_transformers KB bootstrap.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "extraction" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # so we can import extract_concepts

from extract_concepts import ExtractionPipeline  # noqa: E402
from research_kb_storage import ConceptStore, RelationshipStore  # noqa: E402

# (arxiv_id, source_id, nickname) — verified 2026-04-10
SSM_SUBSET: list[tuple[str, str, str]] = [
    ("2008.07669", "07760009-79d0-4533-bb1c-e45b1cbf1937", "HiPPO"),
    ("2111.00396", "4bf3e4db-f9a3-4c47-b109-75b95ef7507d", "S4"),
    ("2206.11893", "4819bb29-845f-4609-aedc-c6897423c69c", "S4D"),
    ("2208.04933", "6cba123e-fa14-44ea-9e6b-a9fb2c81a6ea", "S5"),
    ("2312.00752", "2bd124de-5fdc-4aaf-b496-a9c574402621", "Mamba"),
    ("2405.21060", "c8799f1d-4845-4936-9d2c-7c6b897b106c", "Mamba-2"),
    ("2603.15569", "1613a156-4ee7-4ddb-80d3-dcbb7ad8f829", "Mamba-3"),
    ("2302.10866", "a049ba0f-d8af-426e-8fa2-0de46eff7c27", "Hyena"),
    ("2305.13048", "24da587e-e4ab-4687-9f35-5f0189dfe454", "RWKV"),
    ("2503.14456", "5248cb47-3d4d-4dc0-b1bc-afb7dd363c50", "RWKV-7"),
    ("2503.13427", "c862cdd8-cd58-4893-ab40-d171659f0b82", "xLSTM-7B"),
    ("2006.16236", "73c67b46-6ef1-4a0b-b172-b4a4fc886149", "LinearTransformer"),
    ("2406.06484", "a1e7f7d0-8031-4921-aaff-f45b7da14bb8", "DeltaNet"),
    ("2412.06464", "1ba75828-57ab-4ba6-afc6-cf945935f6cb", "GatedDeltaNet"),
    ("2501.00663", "e14ed023-c516-4b71-98b9-36097fc9d85d", "Titans"),
    ("2312.06635", "2fc83b17-07d9-44b0-ac24-2668a1908a48", "GLA"),
    ("2407.14207", "2f883497-8c14-49dc-9492-87906dc4d2a0", "Longhorn"),
    ("2212.14052", "89b34e0b-ffd5-42e8-84b4-633f43d03309", "H3"),
    ("2307.08621", "06cc5815-572c-42fd-9898-49d76dbada47", "RetNet"),
    ("2405.04517", "be04b6ff-7cc1-4d8c-8d55-b0799872acf9", "xLSTM"),
    ("2501.12352", "7a8204ee-83c6-4190-8c9e-971067cc9f43", "TTR"),
    ("2406.00209", "6b4e0937-2d84-411b-9303-403423f3a387", "MambaLyapunov"),
    ("2504.19561", "55ac6cf3-9b32-4ea9-bb72-ece72a76d8d3", "ESS"),
    ("2511.17970", "8d02866d-df49-4691-a275-c54f31ac3f0d", "Controllability"),
    ("2512.16723", "b493c9cc-1c51-4d6a-8b56-af9513354b77", "KOSS"),
    ("2312.04927", "218dfe85-fed2-4435-84a7-6aecfc9d5f2e", "Zoology"),
    ("2404.06654", "fb8a7870-bee6-4126-a563-bd80dea633af", "RULER"),
    ("2110.13985", "a7d5a810-1b39-42e8-9c4b-8454a4a63fc9", "LSSL"),
    ("2402.18668", "1c100aad-68fc-4b38-9426-f3b2e519e210", "Based"),
    ("2303.06349", "d9bb56c8-73e2-4608-a7a5-ba0e74d92f71", "LRU"),
]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Extract concepts from the SSM subset")
    parser.add_argument(
        "--limit-papers",
        type=int,
        default=None,
        help="Process only the first N papers (for testing)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from paper at this index (for resume)",
    )
    args = parser.parse_args()

    subset = SSM_SUBSET[args.start_index :]
    if args.limit_papers:
        subset = subset[: args.limit_papers]

    # Build pipeline ONCE — it handles LLM client, dedup, concept store lifetime.
    pipeline = ExtractionPipeline(
        backend="anthropic",
        model="haiku-4.5",
        confidence_threshold=0.7,
        batch_size=20,
        concurrency=10,
        dry_run=False,
        skip_backup=True,  # individual sources; full backup not needed per call
        num_ctx=4096,
        domain_id="deep_learning",  # proper domain with RNN/transformer/SSM vocabulary
        timeout=180.0,
    )

    totals = {
        "sources_done": 0,
        "sources_failed": 0,
        "chunks_processed": 0,
        "concepts_new": 0,
        "relationships_stored": 0,
    }

    try:
        for i, (arxiv_id, source_id, nick) in enumerate(subset, 1):
            print(f"\n{'='*70}")
            print(f"[{i}/{len(subset)}] {arxiv_id} {nick}  (source_id={source_id[:8]})")
            print("=" * 70)
            try:
                # resume=False is critical: resume mode ignores source_id filter
                # and would pull ALL unprocessed chunks across the whole KB.
                stats = await pipeline.run(
                    source_id=UUID(source_id),
                    resume=False,
                )
                totals["sources_done"] += 1
                totals["chunks_processed"] += stats.chunks_processed
                totals["concepts_new"] += stats.concepts_new
                totals["relationships_stored"] += stats.relationships_stored
                print(
                    f"  ✓ {stats.chunks_processed} chunks, "
                    f"{stats.concepts_new} new concepts, "
                    f"{stats.relationships_stored} relationships "
                    f"(duration: {stats.duration_seconds:.1f}s)"
                )
            except Exception as e:
                totals["sources_failed"] += 1
                print(f"  ✗ failed: {e}")
    finally:
        try:
            await pipeline.close()
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    for k, v in totals.items():
        print(f"  {k}: {v}")

    total_concepts = await ConceptStore.count()
    total_rels = await RelationshipStore.count()
    print(f"\nDatabase totals: {total_concepts} concepts, {total_rels} relationships")


if __name__ == "__main__":
    asyncio.run(main())
