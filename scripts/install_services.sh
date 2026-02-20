#!/bin/bash
# Install systemd user services for research-kb
# Replaces <PROJECT_ROOT> with actual project path at install time.
#
# Usage: ./scripts/install_services.sh [install|uninstall]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEMD_DIR="$HOME/.config/systemd/user"

install() {
    mkdir -p "$SYSTEMD_DIR"

    echo "Installing research-kb systemd services..."
    echo "  Project root: $PROJECT_ROOT"
    echo "  Systemd dir:  $SYSTEMD_DIR"

    # Daemon service
    sed "s|<PROJECT_ROOT>|$PROJECT_ROOT|g" \
        "$PROJECT_ROOT/services/research-kb-daemon.service" \
        > "$SYSTEMD_DIR/research-kb-daemon.service"
    echo "  Installed: research-kb-daemon.service"

    # KuzuDB sync service + timer
    sed "s|<PROJECT_ROOT>|$PROJECT_ROOT|g" \
        "$PROJECT_ROOT/scripts/systemd/research-kb-sync.service" \
        > "$SYSTEMD_DIR/research-kb-sync.service"
    cp "$PROJECT_ROOT/scripts/systemd/research-kb-sync.timer" \
        "$SYSTEMD_DIR/research-kb-sync.timer"
    echo "  Installed: research-kb-sync.{service,timer}"

    # Reranker service
    sed "s|<PROJECT_ROOT>|$PROJECT_ROOT|g" \
        "$PROJECT_ROOT/scripts/systemd/research-kb-reranker.service" \
        > "$SYSTEMD_DIR/research-kb-reranker.service"
    echo "  Installed: research-kb-reranker.service"

    # S2 discovery service + timer
    sed "s|<PROJECT_ROOT>|$PROJECT_ROOT|g" \
        "$PROJECT_ROOT/services/s2-discovery.service" \
        > "$SYSTEMD_DIR/s2-discovery.service"
    cp "$PROJECT_ROOT/services/s2-discovery.timer" \
        "$SYSTEMD_DIR/s2-discovery.timer"
    echo "  Installed: s2-discovery.{service,timer}"

    systemctl --user daemon-reload
    echo ""
    echo "Services installed. Enable with:"
    echo "  systemctl --user enable --now research-kb-daemon"
    echo "  systemctl --user enable --now research-kb-sync.timer"
    echo "  systemctl --user enable --now s2-discovery.timer"
}

uninstall() {
    echo "Removing research-kb systemd services..."

    local services=(
        research-kb-daemon.service
        research-kb-sync.service
        research-kb-sync.timer
        research-kb-reranker.service
        s2-discovery.service
        s2-discovery.timer
    )

    for svc in "${services[@]}"; do
        systemctl --user stop "$svc" 2>/dev/null || true
        systemctl --user disable "$svc" 2>/dev/null || true
        rm -f "$SYSTEMD_DIR/$svc"
        echo "  Removed: $svc"
    done

    systemctl --user daemon-reload
    echo "Done."
}

case "${1:-install}" in
    install)   install ;;
    uninstall) uninstall ;;
    *)         echo "Usage: $0 [install|uninstall]"; exit 1 ;;
esac
