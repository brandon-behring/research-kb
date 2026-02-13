#!/usr/bin/env bash
# Install the Research-KB reranker as a systemd user service.
#
# Usage:
#   ./scripts/install_reranker.sh install   # Install + enable + start
#   ./scripts/install_reranker.sh status    # Check service status
#   ./scripts/install_reranker.sh uninstall # Stop + disable + remove
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="${HOME}/.config/systemd/user"
SERVICE_FILE="research-kb-reranker.service"

case "${1:-install}" in
    install)
        mkdir -p "$SERVICE_DIR"
        cp "$SCRIPT_DIR/systemd/$SERVICE_FILE" "$SERVICE_DIR/$SERVICE_FILE"
        systemctl --user daemon-reload
        systemctl --user enable "$SERVICE_FILE"
        systemctl --user start "$SERVICE_FILE"
        echo "Reranker service installed and started."
        echo "Check status: systemctl --user status $SERVICE_FILE"
        echo "Socket: /tmp/research_kb_rerank.sock"
        ;;
    status)
        systemctl --user status "$SERVICE_FILE" || true
        if [ -S /tmp/research_kb_rerank.sock ]; then
            echo "Socket: /tmp/research_kb_rerank.sock (exists)"
        else
            echo "Socket: /tmp/research_kb_rerank.sock (missing)"
        fi
        ;;
    uninstall)
        systemctl --user stop "$SERVICE_FILE" || true
        systemctl --user disable "$SERVICE_FILE" || true
        rm -f "$SERVICE_DIR/$SERVICE_FILE"
        systemctl --user daemon-reload
        echo "Reranker service uninstalled."
        ;;
    *)
        echo "Usage: $0 {install|status|uninstall}"
        exit 1
        ;;
esac
