#!/bin/bash
# Install s2-discovery timer as a user systemd service
#
# This timer runs weekly S2 paper discovery, growing the corpus automatically.
#
# Usage:
#   ./scripts/install_s2_discovery.sh [install|uninstall|status|run-now]
#
# After installation:
#   systemctl --user list-timers           # See scheduled timers
#   journalctl --user -u s2-discovery -f   # View logs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="s2-discovery"
SERVICE_FILE="$PROJECT_DIR/services/${SERVICE_NAME}.service"
TIMER_FILE="$PROJECT_DIR/services/${SERVICE_NAME}.timer"
USER_SERVICE_DIR="$HOME/.config/systemd/user"

usage() {
    echo "Usage: $0 [install|uninstall|status|run-now]"
    echo
    echo "Commands:"
    echo "  install    Install and enable the discovery timer"
    echo "  uninstall  Stop and remove the timer/service"
    echo "  status     Show timer and service status"
    echo "  run-now    Trigger discovery immediately"
    exit 1
}

install_timer() {
    echo "Installing $SERVICE_NAME timer..."

    # Create user systemd directory
    mkdir -p "$USER_SERVICE_DIR"

    # Symlink service and timer files
    echo "Linking service files..."
    ln -sf "$SERVICE_FILE" "$USER_SERVICE_DIR/${SERVICE_NAME}.service"
    ln -sf "$TIMER_FILE" "$USER_SERVICE_DIR/${SERVICE_NAME}.timer"

    # Reload systemd
    echo "Reloading systemd..."
    systemctl --user daemon-reload

    # Enable timer (auto-start on login)
    echo "Enabling timer..."
    systemctl --user enable "${SERVICE_NAME}.timer"

    # Start timer now
    echo "Starting timer..."
    systemctl --user start "${SERVICE_NAME}.timer"

    echo
    echo "Installation complete!"
    echo
    echo "Timer is now scheduled. Next run:"
    systemctl --user list-timers "${SERVICE_NAME}.timer" --no-pager 2>/dev/null || true
    echo
    echo "To trigger discovery now:"
    echo "  systemctl --user start ${SERVICE_NAME}.service"
    echo
    echo "To view logs:"
    echo "  journalctl --user -u ${SERVICE_NAME} -f"
}

uninstall_timer() {
    echo "Uninstalling $SERVICE_NAME..."

    # Stop timer if running
    if systemctl --user is-active --quiet "${SERVICE_NAME}.timer" 2>/dev/null; then
        echo "Stopping timer..."
        systemctl --user stop "${SERVICE_NAME}.timer"
    fi

    # Stop service if running
    if systemctl --user is-active --quiet "${SERVICE_NAME}.service" 2>/dev/null; then
        echo "Stopping service..."
        systemctl --user stop "${SERVICE_NAME}.service"
    fi

    # Disable timer
    if systemctl --user is-enabled --quiet "${SERVICE_NAME}.timer" 2>/dev/null; then
        echo "Disabling timer..."
        systemctl --user disable "${SERVICE_NAME}.timer"
    fi

    # Remove symlinks
    if [ -L "$USER_SERVICE_DIR/${SERVICE_NAME}.service" ]; then
        echo "Removing service file..."
        rm "$USER_SERVICE_DIR/${SERVICE_NAME}.service"
    fi
    if [ -L "$USER_SERVICE_DIR/${SERVICE_NAME}.timer" ]; then
        echo "Removing timer file..."
        rm "$USER_SERVICE_DIR/${SERVICE_NAME}.timer"
    fi

    # Reload systemd
    systemctl --user daemon-reload

    echo "Uninstallation complete!"
}

show_status() {
    echo "=== S2 Discovery Timer Status ==="
    echo

    echo "--- Timer Status ---"
    if systemctl --user is-active --quiet "${SERVICE_NAME}.timer" 2>/dev/null; then
        echo "Timer: ACTIVE"
    else
        echo "Timer: INACTIVE"
    fi
    echo
    systemctl --user status "${SERVICE_NAME}.timer" --no-pager 2>/dev/null || echo "(timer not installed)"
    echo

    echo "--- Scheduled Runs ---"
    systemctl --user list-timers "${SERVICE_NAME}.timer" --no-pager 2>/dev/null || echo "(no timer scheduled)"
    echo

    echo "--- Last Service Run ---"
    if systemctl --user is-active --quiet "${SERVICE_NAME}.service" 2>/dev/null; then
        echo "Service: RUNNING"
    else
        echo "Service: STOPPED (normal for oneshot)"
    fi
    echo
    systemctl --user status "${SERVICE_NAME}.service" --no-pager 2>/dev/null || echo "(service not installed)"
    echo

    echo "--- Recent Logs ---"
    journalctl --user -u "${SERVICE_NAME}" --no-pager -n 20 2>/dev/null || echo "(no logs)"
}

run_now() {
    echo "Triggering S2 discovery now..."
    systemctl --user start "${SERVICE_NAME}.service"
    echo
    echo "Discovery started. Follow logs with:"
    echo "  journalctl --user -u ${SERVICE_NAME} -f"
}

# Main
case "${1:-}" in
    install)
        install_timer
        ;;
    uninstall)
        uninstall_timer
        ;;
    status)
        show_status
        ;;
    run-now)
        run_now
        ;;
    *)
        usage
        ;;
esac
