#!/bin/bash
# Download GGUF model for llama.cpp inference
# Usage: ./scripts/download_gguf_model.sh

set -e

MODEL_DIR="<PROJECT_ROOT>/models"
MODEL_NAME="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

# Hugging Face model URL (bartowski's quantized version)
MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

echo "=== Downloading GGUF Model ==="
echo "Model: $MODEL_NAME"
echo "Destination: $MODEL_PATH"
echo ""

if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH"
    echo "Size: $(du -h "$MODEL_PATH" | cut -f1)"
    exit 0
fi

mkdir -p "$MODEL_DIR"

# Download with wget (resume support)
echo "Downloading (~4.9GB)..."
wget --continue --progress=bar:force:noscroll \
    -O "$MODEL_PATH" \
    "$MODEL_URL"

echo ""
echo "=== Download Complete ==="
echo "Model: $MODEL_PATH"
echo "Size: $(du -h "$MODEL_PATH" | cut -f1)"
