#!/usr/bin/env python3
"""Ingest a targeted RL + optimal control paper subset.

Mirrors ingest_ssm_subset.py. Default TARGET_ARXIV_IDS list reflects the
83 [downloaded]-status arxiv papers from
``~/Claude/rl_and_control/references/paper_index.md`` (28-week RL+control
curriculum). Callers (e.g., ``rl_and_control/scripts/sync_to_kb.py``) may
override via ``--arxiv-ids <file>`` so the ingest batch tracks paper_index.md
without needing to re-edit this file.

Written 2026-05-24 to support the rl-and-control citation-graph demo (Phase A
of the multi-project graph-export initiative).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

# Add packages to path (mirrors ingest_ssm_subset.py)
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from ingest_missing_papers import (  # noqa: E402
    compute_file_hash,
    get_metadata_for_pdf,
    ingest_pdf,
)
from research_kb_common import get_logger  # noqa: E402
from research_kb_pdf.post_ingest import run_post_ingest_hooks  # noqa: E402
from research_kb_storage import (  # noqa: E402
    DatabaseConfig,
    SourceStore,
    close_connection_pool,
    get_connection_pool,
)

logger = get_logger(__name__)

# Default target arXiv IDs — 83 papers from rl_and_control/references/paper_index.md
# marked [downloaded] as of 2026-05-24. Override with --arxiv-ids <file>.
TARGET_ARXIV_IDS = [
    "1312.5602",  # DQN (Mnih et al., Nature 2015 preprint)
    "1406.2199",  # Deterministic Policy Gradient (Silver et al.)
    "1502.05477",  # TRPO (Schulman et al.)
    "1504.00702",  # GPS Guided Policy Search (Levine et al.)
    "1506.02438",  # GAE (Schulman et al.)
    "1509.02971",  # DDPG (Lillicrap et al.)
    "1509.06461",  # Double DQN (van Hasselt et al.)
    "1511.05952",  # Prioritized Experience Replay (Schaul et al.)
    "1511.06581",  # Dueling Network Architectures (Wang et al.)
    "1602.01783",  # A3C (Mnih et al.)
    "1609.05140",  # NAF / Continuous Q-Learning (Gu et al.)
    "1703.01161",  # Distributional RL / Categorical DQN (Bellemare et al.)
    "1703.03400",  # MAML Meta-Learning (Finn et al.)
    "1703.06907",  # Domain Randomization (Tobin et al.)
    "1706.02275",  # MADDPG (Lowe et al.)
    "1707.01495",  # HER Hindsight Experience Replay (Andrychowicz et al.)
    "1707.06347",  # PPO (Schulman et al.)
    "1709.06560",  # Deep RL That Matters (Henderson et al.)
    "1710.02298",  # Rainbow DQN (Hessel et al.)
    "1710.06537",  # IMPALA (Espeholt et al.)
    "1712.01815",  # AlphaZero (Silver et al.)
    "1801.01290",  # SAC v1 (Haarnoja et al.)
    "1802.09477",  # TD3 (Fujimoto et al.)
    "1803.11485",  # QMIX (Rashid et al.)
    "1804.02717",  # IQN (Dabney et al.)
    "1805.00909",  # Latent Diffusion control (placeholder)
    "1805.08296",  # NoisyNets exploration (Fortunato et al.)
    "1805.12114",  # PETS / Probabilistic Ensembles (Chua et al.)
    "1806.09460",  # SLAC (Lee et al.)
    "1808.00177",  # OpenAI Dota Five (Berner et al.)
    "1810.13400",  # SLBO Model-based RL (Luo et al.)
    "1811.00168",  # Generalized LQR (Stanford)
    "1811.04551",  # POPLIN (Wang & Ba)
    "1812.05905",  # SAC v2 (Haarnoja et al.)
    "1906.08253",  # MBPO Model-Based Policy Optimization (Janner et al.)
    "1910.07113",  # Solving Rubik's Cube (OpenAI)
    "1911.08265",  # MuZero (Schrittwieser et al.)
    "1912.06680",  # Dota 2 with Large Scale Deep RL (OpenAI)
    "2002.04523",  # DREAMER v1 (Hafner et al.)
    "2004.04136",  # MBRL benchmark (Wang et al.)
    "2004.07584",  # CURL (Laskin et al.)
    "2005.01643",  # Offline RL Tutorial (Levine et al.)
    "2005.12729",  # SOP Surprise-Optimal Planning
    "2006.04779",  # CQL Conservative Q-Learning (Kumar et al.)
    "2006.07869",  # MOPO Model-Based Offline (Yu et al.)
    "2009.05228",  # Implicit Q-Learning (Kostrikov et al.)
    "2010.02193",  # DREAMER v2 (Hafner et al.)
    "2010.11251",  # PlaNet (Hafner et al.)
    "2011.03506",  # MOReL (Kidambi et al.)
    "2103.01955",  # RAD Data Augmentation RL (Laskin et al.)
    "2106.01345",  # Decision Transformer (Chen et al.)
    "2107.09645",  # SUNRISE Ensemble (Lee et al.)
    "2110.06169",  # IRIS World Models (Micheli et al.)
    "2203.04955",  # TD-MPC v1 (Hansen et al.)
    "2204.01464",  # SimSR (Zang et al.)
    "2205.10330",  # XTRA Cross-task transfer
    "2206.09328",  # Q-Diffusion (Wang et al.)
    "2209.00588",  # DREAMER v3 (Hafner et al.)
    "2301.04104",  # Goal-Conditioned RL Survey
    "2301.08243",  # I-JEPA (LeCun et al.)
    "2303.03381",  # DT++ Decision Transformer variants
    "2306.09852",  # RWR / Reward-Weighted Regression survey
    "2307.02064",  # World Models for Robotics (Hafner)
    "2310.06253",  # SmartPlay agent benchmark
    "2310.16828",  # Self-Speculative Decoding (RL adjacent)
    "2312.06211",  # DiT World Models
    "2312.15521",  # MPPI revisited
    "2403.04253",  # Genie generative interactive envs
    "2404.08471",  # OpenRLHF
    "2406.00592",  # MPC and RL: Unified Framework (IFAC)
    "2408.03539",  # TD-MPC v2 (Hansen et al.)
    "2410.08893",  # Mamba for RL world models
    "2502.00021",  # Iterated DPO / RL variants
    "2502.02133",  # Synthesis of MPC and RL (Reiter et al.)
    "2502.13187",  # Adaptive MPC with RL gain scheduling
    "2502.20168",  # Diffusion policy for control
    "2505.08238",  # DGRL / Distributed RL platform
    "2505.17342",  # Foundation models for control
    "2505.22772",  # RLHF for control synthesis
    "2506.09985",  # Modular world models
    "2508.09128",  # MPC-RL evaluation benchmark
    "2510.06179",  # Symbolic control + LLM
    "2603.28243",  # Recent RL+control synthesis (placeholder; verify)
]

DOMAIN_ID = "reinforcement_learning"
INGEST_BATCH_TAG = "rl_subset_2026_05_24"


def _load_arxiv_ids(path: Path) -> list[str]:
    """Load arXiv IDs from a newline-delimited file (one ID per line; # comments allowed)."""
    ids: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest RL+control paper subset")
    parser.add_argument(
        "--arxiv-ids",
        type=Path,
        default=None,
        help="Path to newline-delimited file of arXiv IDs to ingest "
        "(overrides built-in TARGET_ARXIV_IDS).",
    )
    parser.add_argument(
        "--no-build-citations",
        action="store_true",
        help="Skip the post-ingest citation graph build (default: on)",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="If set, write a structured JSON summary to this path "
        "(consumed by rl_and_control/scripts/sync_to_kb.py to update "
        "paper_index.md statuses).",
    )
    args = parser.parse_args()

    target_ids = _load_arxiv_ids(args.arxiv_ids) if args.arxiv_ids else TARGET_ARXIV_IDS
    if not target_ids:
        print("✗ No target arXiv IDs provided; nothing to do.")
        return

    papers_dir = Path(__file__).parent.parent / "fixtures" / "papers" / "arxiv"

    config = DatabaseConfig()
    await get_connection_pool(config)

    results: dict[str, list[dict]] = {"success": [], "skipped": [], "failed": []}

    for arxiv_id in target_ids:
        pdf_path = papers_dir / f"{arxiv_id}.pdf"
        if not pdf_path.exists():
            print(f"✗ {arxiv_id}: PDF not found at {pdf_path}")
            results["failed"].append({"arxiv_id": arxiv_id, "error": "pdf missing"})
            continue

        file_hash = compute_file_hash(str(pdf_path))
        existing = await SourceStore.get_by_file_hash(file_hash)
        if existing:
            print(f"⊘ {arxiv_id}: already ingested as '{existing.title[:60]}'")
            results["skipped"].append({"arxiv_id": arxiv_id, "title": existing.title})
            continue

        title, authors, year, extra_metadata = get_metadata_for_pdf(pdf_path)
        extra_metadata["auto_ingested"] = True
        extra_metadata["ingest_batch"] = INGEST_BATCH_TAG
        paper_domain = extra_metadata.pop("sidecar_domain_id", None) or DOMAIN_ID

        print(f"→ {arxiv_id}: '{title[:60]}' (domain={paper_domain})")
        try:
            source_id, num_chunks, num_headings = await ingest_pdf(
                pdf_path=str(pdf_path),
                title=title,
                authors=authors,
                year=year,
                metadata=extra_metadata,
                domain_id=paper_domain,
                no_embed=True,  # two-phase workflow; backfill embeddings after
            )
            print(f"  ✓ source_id={source_id[:8]}, {num_chunks} chunks, {num_headings} headings")
            results["success"].append(
                {"arxiv_id": arxiv_id, "source_id": source_id, "chunks": num_chunks}
            )
        except Exception as e:
            print(f"  ✗ failed: {e}")
            results["failed"].append({"arxiv_id": arxiv_id, "error": str(e)[:200]})

    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Targets: {len(target_ids)}")
    print(f"Success: {len(results['success'])}")
    print(f"Skipped (already in KB): {len(results['skipped'])}")
    print(f"Failed: {len(results['failed'])}")

    if results["success"]:
        total_chunks = sum(r["chunks"] for r in results["success"])
        print(f"Total new chunks: {total_chunks}")

    if results["failed"]:
        print("\nFailed:")
        for r in results["failed"]:
            print(f"  - {r['arxiv_id']}: {r['error'][:80]}")

    # Post-ingest hook: build citation graph for just the new sources
    new_ids: list[UUID] = []
    for r in results["success"]:
        try:
            new_ids.append(UUID(r["source_id"]))
        except (TypeError, ValueError):
            logger.warning("invalid_source_id", source_id=r.get("source_id"))

    citation_summary: dict = {}
    if new_ids and not args.no_build_citations:
        try:
            print(f"\nBuilding citation graph for {len(new_ids)} new sources...")
            hook_summary = await run_post_ingest_hooks(new_ids)
            citation_summary = hook_summary.get("citations", {})
            print(
                f"  matched={citation_summary.get('matched', 0)} "
                f"unmatched={citation_summary.get('unmatched', 0)}"
            )
        except Exception as e:
            print(f"  ⚠ citation graph build failed: {e}")
            logger.error("post_ingest_hook_failed", error=str(e))

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "ingest_batch": INGEST_BATCH_TAG,
            "targets": list(target_ids),
            "results": results,
            "citations": citation_summary,
        }
        args.summary_out.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False))
        print(f"\n→ Summary written to {args.summary_out}")

    await close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())
