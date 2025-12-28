# Research KB Daemon

Low-latency query service for research-kb via Unix domain socket.

## Overview

The daemon provides sub-100ms query responses by:
- Maintaining a persistent database connection pool
- Pre-loading the embedding model
- Using Unix socket for minimal IPC overhead
- Supporting JSON-RPC 2.0 protocol

## Installation

```bash
pip install -e packages/daemon
```

## Usage

### Starting the Daemon

```bash
# Start with default socket path
research-kb-daemon

# Start with custom socket path
research-kb-daemon --socket /tmp/custom.sock
```

### Systemd Service

```bash
# Install as user service
./scripts/install_daemon.sh install

# Start service
systemctl --user start research-kb-daemon

# Enable auto-start on login
systemctl --user enable research-kb-daemon

# Check status
systemctl --user status research-kb-daemon
```

## Protocol

JSON-RPC 2.0 over Unix domain socket.

### Methods

#### search

Execute hybrid search.

```json
{
  "jsonrpc": "2.0",
  "method": "search",
  "params": {
    "query": "instrumental variables",
    "limit": 10,
    "context_type": "balanced",
    "use_graph": true,
    "use_citations": false
  },
  "id": 1
}
```

**Parameters:**
- `query` (string, required): Search query text
- `limit` (int, default: 10): Maximum results
- `context_type` (string): Weight preset - "building", "auditing", "balanced"
- `use_graph` (bool): Enable graph boosting
- `use_citations` (bool): Enable citation authority

#### health

Check system health.

```json
{"jsonrpc": "2.0", "method": "health", "id": 1}
```

Returns:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "status": "healthy",
    "uptime_seconds": 3600.5,
    "database": "healthy",
    "embed_server": {"status": "healthy", "device": "cuda"}
  },
  "id": 1
}
```

#### stats

Get database statistics.

```json
{"jsonrpc": "2.0", "method": "stats", "id": 1}
```

### Testing with nc

```bash
echo '{"jsonrpc":"2.0","method":"health","id":1}' | nc -U /tmp/research_kb_daemon_$USER.sock
```

## Dependencies

- research-kb-storage: Database operations
- research-kb-contracts: Data models
- research-kb-common: Logging
- httpx: Async HTTP client
- jsonrpcserver: JSON-RPC handling

## Architecture

```
┌─────────────────────────────────────────┐
│  research-kb-daemon                     │
├─────────────────────────────────────────┤
│  Unix Socket Server (asyncio)           │
│  JSON-RPC 2.0 Protocol                  │
├─────────────────────────────────────────┤
│  Connection Pools:                      │
│  - asyncpg (2-10 database connections)  │
│  - embed_server (Unix socket client)    │
└─────────────────────────────────────────┘
```
