#!/usr/bin/env bash
# Hook integration for lever_of_archimedes
#
# Usage: source this file from hooks/lib/research_kb.sh
#
# Example integration (in lever_of_archimedes/hooks/lib/research_kb.sh):
#   source /path/to/research-kb/services/research_kb_daemon/hook_integration.sh
#   results=$(search_research_kb "$prompt" 3)
#   if [ -n "$results" ]; then
#       echo "📚 Research KB Context:"
#       echo "$results"
#   fi

SOCKET_PATH="${RESEARCH_KB_SOCKET_PATH:-/tmp/research_kb_daemon.sock}"
CLIENT_PATH="${RESEARCH_KB_CLIENT:-$HOME/Claude/research-kb/services/research_kb_daemon/client.py}"

# Check if daemon is available
is_daemon_available() {
    [ -S "$SOCKET_PATH" ] && python3 -c "
import socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(0.1)
try:
    s.connect('$SOCKET_PATH')
    s.close()
    exit(0)
except:
    exit(1)
" 2>/dev/null
}

# Search via daemon (fast path)
search_via_daemon() {
    local query="$1"
    local limit="${2:-3}"

    python3 "$CLIENT_PATH" "$query" --limit "$limit" 2>/dev/null
}

# Search via CLI (fallback)
search_via_cli() {
    local query="$1"
    local limit="${2:-3}"

    research-kb query "$query" --limit "$limit" --no-graph --timeout 5 2>/dev/null
}

# Main search function - tries daemon first, falls back to CLI
search_research_kb() {
    local query="$1"
    local limit="${2:-3}"

    # Try daemon first (faster)
    if is_daemon_available; then
        search_via_daemon "$query" "$limit"
        return $?
    fi

    # Fall back to CLI
    search_via_cli "$query" "$limit"
}

# Quick check if query is causal inference related
is_causal_query() {
    local query="$1"
    local keywords="causal|IV|instrumental|DML|double machine learning|treatment effect|ATE|ATT|confound|endogen|counterfactual|DAG|SCM|propensity|regression discontinuity|DiD|difference.in.difference|synthetic control|CATE|heterogeneous treatment"

    echo "$query" | grep -iE "$keywords" >/dev/null 2>&1
}
