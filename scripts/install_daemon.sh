#!/bin/bash
# Install research-kb-daemon as a user systemd service
#
# Usage:
#   ./scripts/install_daemon.sh [install|uninstall|status]
#
# After installation:
#   systemctl --user start research-kb-daemon
#   systemctl --user enable research-kb-daemon  # Auto-start on login

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="research-kb-daemon"
SERVICE_FILE="$PROJECT_DIR/services/${SERVICE_NAME}.service"
USER_SERVICE_DIR="$HOME/.config/systemd/user"

usage() {
    echo "Usage: $0 [install|uninstall|status]"
    echo
    echo "Commands:"
    echo "  install    Install and enable the daemon service"
    echo "  uninstall  Stop and remove the daemon service"
    echo "  status     Show service status"
    exit 1
}

install_service() {
    echo "Installing $SERVICE_NAME..."

    # Create user systemd directory
    mkdir -p "$USER_SERVICE_DIR"

    # Install package first
    echo "Installing daemon package..."
    pip install -e "$PROJECT_DIR/packages/daemon"

    # Symlink service file
    echo "Linking service file..."
    ln -sf "$SERVICE_FILE" "$USER_SERVICE_DIR/${SERVICE_NAME}.service"

    # Reload systemd
    echo "Reloading systemd..."
    systemctl --user daemon-reload

    # Enable service (auto-start on login)
    echo "Enabling service..."
    systemctl --user enable "$SERVICE_NAME"

    echo
    echo "Installation complete!"
    echo
    echo "To start the daemon now:"
    echo "  systemctl --user start $SERVICE_NAME"
    echo
    echo "To check status:"
    echo "  systemctl --user status $SERVICE_NAME"
    echo
    echo "To view logs:"
    echo "  journalctl --user -u $SERVICE_NAME -f"
}

uninstall_service() {
    echo "Uninstalling $SERVICE_NAME..."

    # Stop service if running
    if systemctl --user is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        echo "Stopping service..."
        systemctl --user stop "$SERVICE_NAME"
    fi

    # Disable service
    if systemctl --user is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        echo "Disabling service..."
        systemctl --user disable "$SERVICE_NAME"
    fi

    # Remove symlink
    if [ -L "$USER_SERVICE_DIR/${SERVICE_NAME}.service" ]; then
        echo "Removing service file..."
        rm "$USER_SERVICE_DIR/${SERVICE_NAME}.service"
    fi

    # Reload systemd
    systemctl --user daemon-reload

    echo "Uninstallation complete!"
}

show_status() {
    echo "Service: $SERVICE_NAME"
    echo

    if systemctl --user is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        echo "Status: RUNNING"
    else
        echo "Status: STOPPED"
    fi

    echo
    systemctl --user status "$SERVICE_NAME" --no-pager 2>/dev/null || true
}

# Main
case "${1:-}" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    status)
        show_status
        ;;
    *)
        usage
        ;;
esac
