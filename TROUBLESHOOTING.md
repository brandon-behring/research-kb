# Troubleshooting

Common issues and solutions for research-kb.

## GROBID

### GROBID takes 60+ seconds to start

This is expected. The Docker healthcheck has a 60-second `start_period`. Wait for the container to report healthy:

```bash
docker compose ps  # Wait for grobid to show "healthy"
```

### GROBID returns empty extractions

Ensure GROBID is fully started (check health). Some PDFs may not be extractable — check the PDF is not image-only (scanned).

## Embedding Server

### `embed_batch() timeout` on large batches

The CPU-based embedding server has a 60-second timeout. Batches over 100 texts will exceed this. The client automatically chunks into batches of 100, but if you see timeouts:

```bash
# Check embed server is running
ls /tmp/research_kb_embed.sock

# Restart if needed
# The embed server starts with the daemon or can be run standalone
```

### Slow embedding (minutes per document)

CPU-based embedding is inherently slow. For large corpora:
- Use GPU acceleration if available (`CUDA_VISIBLE_DEVICES=0`)
- Expect ~60-90 minutes for 13 textbooks on CPU
- Run ingestion overnight with `nohup`

## KuzuDB

### `KuzuDB lock: only one process can access at a time`

KuzuDB uses file-level locking. If the daemon holds the lock, other processes (scripts, tests) cannot access it.

**Solutions:**
- Stop the daemon before running sync scripts: `systemctl --user stop research-kb-daemon`
- Tests use `tmp_path` for isolated KuzuDB instances (no conflict)
- The daemon pre-warms KuzuDB on startup — cold start takes 5-15 seconds

### Graph queries returning empty results

1. Check KuzuDB has been synced: `research-kb stats` (look for concept/relationship counts)
2. If empty, sync from PostgreSQL: `python scripts/sync_kuzu.py`
3. Verify KuzuDB path: `ls ~/.research_kb/kuzu/research_kb.kuzu/`

## Docker Compose

### NEVER use `docker compose down -v`

The `-v` flag **destroys all PostgreSQL data** including your corpus. Use the safe wrapper:

```bash
# Safe: uses confirmation prompt
./scripts/docker-safe.sh down -v

# Or just stop containers without removing volumes
docker compose stop
docker compose down  # (without -v) removes containers but keeps data
```

### Database connection refused

```bash
# Check PostgreSQL is running
docker compose ps

# Check connection
psql postgresql://research_kb:research_kb@localhost:5432/research_kb -c "SELECT 1"

# Restart if needed
docker compose restart postgres
```

## Concept Extraction

### Ollama: only partial GPU offload

On GPUs with limited VRAM (8GB), the llama3.1:8b model may only offload 9/33 layers to GPU. This makes it CPU-bound at ~1 chunk/min.

**Solutions:**
- Use the Anthropic backend instead (~200 chunks/min): `--backend anthropic`
- Use a smaller model with Ollama
- Allocate more VRAM (larger GPU)

### Haiku JSON output wrapped in code fences

The Anthropic Haiku model sometimes wraps JSON in markdown code fences even when instructed not to. The extraction client automatically strips these. If you see parsing errors, check for this pattern.

### Haiku max_tokens too low

For methods with 8+ assumptions, 1500 tokens is insufficient for the structured JSON output. The default is set to 4096 to handle verbose outputs.

## Daemon

### `nc: unix connect failed`

The daemon is not running:

```bash
# Start manually
research-kb-daemon &

# Or via systemd
systemctl --user start research-kb-daemon

# Check socket exists
ls /tmp/research_kb_daemon_$USER.sock
```

### First query slow (>5 seconds)

The daemon pre-warms KuzuDB and loads the embedding model on startup. The first query after start may be slow. Subsequent queries should be <100ms.

### Socket permission denied

Check ownership of the socket file:

```bash
ls -la /tmp/research_kb_daemon_$USER.sock
# Should be owned by your user
```

## Search

### Graph search falls back to FTS+vector

This is expected behavior when concepts haven't been extracted for the queried topic. Run concept extraction first:

```bash
python scripts/extract_concepts.py --limit 1000
python scripts/sync_kuzu.py
```

### Empty search results

1. Check database has data: `research-kb stats`
2. If empty, ingest the demo corpus: `python scripts/setup_demo.py`
3. Check embedding server is running (vector search needs it)
