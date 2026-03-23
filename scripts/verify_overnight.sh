#!/usr/bin/env bash
# Verify overnight pipeline results (embedding backfill + citation extraction)
#
# Usage: bash scripts/verify_overnight.sh

set -euo pipefail

VENV=".venv/bin/python"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=== Research-KB Overnight Pipeline Verification ==="
echo "$(date)"
echo ""

# --- Process status ---
echo "--- Process Status ---"
for name in backfill citation; do
    pidfile="/tmp/${name}_pid.txt"
    if [[ -f "$pidfile" ]]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}[$name] PID $pid still running${NC}"
        else
            echo -e "${GREEN}[$name] PID $pid finished${NC}"
        fi
    else
        echo -e "${YELLOW}[$name] No PID file found at $pidfile${NC}"
    fi
done
echo ""

# --- Database checks ---
echo "--- Database Metrics ---"
$VENV -c "
import asyncio, asyncpg

async def verify():
    conn = await asyncpg.connect(user='postgres', password='postgres',
                                  host='localhost', port=5432, database='research_kb')

    # Embedding coverage
    total_chunks = await conn.fetchval('SELECT COUNT(*) FROM chunks')
    null_emb = await conn.fetchval('SELECT COUNT(*) FROM chunks WHERE embedding IS NULL')
    emb_pct = ((total_chunks - null_emb) / total_chunks * 100) if total_chunks > 0 else 0

    # Citation network
    cit_edges = await conn.fetchval('SELECT COUNT(*) FROM source_citations')
    cit_auth = await conn.fetchval('SELECT COUNT(*) FROM sources WHERE citation_authority > 0')
    raw_cit = await conn.fetchval('SELECT COUNT(*) FROM citations')

    # Sources
    total_src = await conn.fetchval('SELECT COUNT(*) FROM sources')
    src_with_cit = await conn.fetchval('''
        SELECT COUNT(DISTINCT source_id) FROM citations
    ''')

    await conn.close()

    print(f'Total chunks:           {total_chunks:>10,}')
    print(f'Null embeddings:        {null_emb:>10,}  (target: 0)')
    if null_emb == 0:
        print('  ✓ PASS: All chunks have embeddings')
    else:
        print(f'  ✗ FAIL: {null_emb:,} chunks still missing embeddings ({100-emb_pct:.1f}%)')
    print()

    print(f'Raw citations:          {raw_cit:>10,}')
    print(f'Source citation edges:   {cit_edges:>10,}  (baseline: 5,028)')
    if cit_edges > 5028:
        print(f'  ✓ PASS: {cit_edges - 5028:,} new edges added')
    else:
        print(f'  ✗ FAIL: No new citation edges (still {cit_edges:,})')
    print()

    print(f'Sources with authority:  {cit_auth:>10,}  (baseline: 132)')
    if cit_auth > 132:
        print(f'  ✓ PASS: {cit_auth - 132} new sources have authority scores')
    else:
        print(f'  ✗ FAIL: No new authority scores')
    print()

    print(f'Sources total:          {total_src:>10,}')
    print(f'Sources with citations: {src_with_cit:>10,}  (of {total_src})')

asyncio.run(verify())
"

echo ""

# --- Log file checks ---
echo "--- Log Files ---"
for logfile in /tmp/backfill_report.json /tmp/citation_report.log /tmp/citation_graph.log; do
    if [[ -f "$logfile" ]]; then
        lines=$(wc -l < "$logfile")
        size=$(du -h "$logfile" | cut -f1)
        last_line=$(tail -1 "$logfile")
        echo "  $logfile: $lines lines, $size"
        echo "    Last: $last_line"
    else
        echo "  $logfile: NOT FOUND"
    fi
done

echo ""
echo "--- Next Steps ---"
echo "If all checks pass:"
echo "  1. Run post-backfill eval:"
echo "     .venv/bin/python scripts/eval_retrieval.py --per-domain --output /tmp/post_backfill_eval.json"
echo "  2. Run citation eval:"
echo "     .venv/bin/python scripts/eval_retrieval.py --per-domain --use-citations --citation-weight 0.15 --output /tmp/citation_eval.json"
echo "  3. Optimize weights:"
echo "     .venv/bin/python scripts/optimize_weights.py --no-graph --cross-validate"
