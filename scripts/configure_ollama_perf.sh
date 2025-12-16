#!/bin/bash
# Configure Ollama with performance flags for faster extraction
# Run with: sudo ./scripts/configure_ollama_perf.sh

set -e

echo "=== Configuring Ollama Performance Flags ==="

# Create override directory
mkdir -p /etc/systemd/system/ollama.service.d

# Write override config
cat > /etc/systemd/system/ollama.service.d/override.conf << 'OVERRIDE'
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
OVERRIDE

echo "✓ Created /etc/systemd/system/ollama.service.d/override.conf"

# Reload systemd
systemctl daemon-reload
echo "✓ Reloaded systemd"

# Restart Ollama
systemctl restart ollama
echo "✓ Restarted Ollama service"

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 3

# Verify service is running
if systemctl is-active --quiet ollama; then
    echo "✓ Ollama is running"
else
    echo "✗ Ollama failed to start"
    systemctl status ollama
    exit 1
fi

echo ""
echo "=== Configuration Complete ==="
echo "Environment variables set:"
echo "  OLLAMA_FLASH_ATTENTION=1"
echo "  OLLAMA_NUM_PARALLEL=2"
echo "  OLLAMA_KV_CACHE_TYPE=q8_0"
