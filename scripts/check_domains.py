"""Quick check: list current domain values and counts."""

import asyncio
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for pkg in ("storage", "contracts", "common"):
    sys.path.insert(0, str(_root / "packages" / pkg / "src"))

from research_kb_storage import DatabaseConfig, get_connection_pool


async def main():
    pool = await get_connection_pool(DatabaseConfig())
    rows = await pool.fetch(
        "SELECT DISTINCT metadata->>'domain' as domain, COUNT(*) as cnt "
        "FROM sources GROUP BY 1 ORDER BY 2 DESC"
    )
    print("Current domain distribution:")
    for r in rows:
        d = r["domain"] or "(null)"
        print(f"  {d:30s} {r['cnt']}")
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
