#!/bin/bash
# Build llama-cpp-python with CUDA support
# Run AFTER install_llama_cpp_cuda.sh (which installs CUDA toolkit)
# Usage: ./scripts/build_llama_cpp_cuda.sh

set -e

cd <PROJECT_ROOT>

echo "=== Building llama-cpp-python with CUDA ==="

# Activate venv
source venv/bin/activate

# Uninstall existing
pip uninstall -y llama-cpp-python 2>/dev/null || true

# Build with CUDA support
echo "Building from source with CUDA enabled (this takes ~5 minutes)..."
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir --force-reinstall

# Verify GPU support
echo ""
echo "=== Verifying GPU Support ==="
python3 -c "
from llama_cpp import Llama
print('llama-cpp-python version:', Llama.__module__)
# Check if GPU offload is supported
import llama_cpp.llama_cpp as llama_cpp_lib
print('supports_gpu_offload:', llama_cpp_lib.llama_supports_gpu_offload())
"

echo ""
echo "=== Build Complete ==="
