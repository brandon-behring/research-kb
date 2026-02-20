"""Add missing domains to the domains table.

Must be run before sync_chunk_domains.py to satisfy FK constraints.

Usage:
    python scripts/add_missing_domains.py
"""

import asyncio
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for pkg in ("storage", "contracts", "common"):
    sys.path.insert(0, str(_root / "packages" / pkg / "src"))

from research_kb_storage import DatabaseConfig, get_connection_pool

# New domains to add (id, display_name, description)
NEW_DOMAINS = [
    (
        "deep_learning",
        "Deep Learning",
        "Neural networks, architectures, training techniques",
    ),
    ("econometrics", "Econometrics", "Statistical methods for economic data analysis"),
    ("finance", "Finance", "Financial instruments, pricing, risk management"),
    (
        "machine_learning",
        "Machine Learning",
        "Classical and modern ML algorithms and methods",
    ),
    (
        "mathematics",
        "Mathematics",
        "Linear algebra, calculus, differential equations, physics",
    ),
    (
        "software_engineering",
        "Software Engineering",
        "API design, DevOps, testing, architecture",
    ),
    (
        "statistics",
        "Statistics",
        "Bayesian, frequentist, hypothesis testing, estimation",
    ),
    (
        "data_science",
        "Data Science",
        "Applied data analysis, complexity science, networks",
    ),
    ("ml_engineering", "ML Engineering", "ML pipelines, MLOps, deployment"),
    ("algorithms", "Algorithms", "Data structures, graph algorithms, optimization"),
    ("functional_programming", "Functional Programming", "Haskell, Scala, FP patterns"),
    ("portfolio_management", "Portfolio Management", "CFA materials, portfolio theory"),
    ("fitness", "Fitness", "Strength training, exercise science"),
    ("economics", "Economics", "Macroeconomics, microeconomics"),
    ("forecasting", "Forecasting", "General forecasting methods"),
]


async def main():
    pool = await get_connection_pool(DatabaseConfig())

    existing = await pool.fetch("SELECT id FROM domains")
    existing_ids = {r["id"] for r in existing}

    added = 0
    for domain_id, name, description in NEW_DOMAINS:
        if domain_id not in existing_ids:
            await pool.execute(
                "INSERT INTO domains (id, name, description) VALUES ($1, $2, $3)",
                domain_id,
                name,
                description,
            )
            print(f"  Added: {domain_id} ({name})")
            added += 1

    print(f"\nAdded {added} new domain(s). Total: {len(existing_ids) + added}")
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
