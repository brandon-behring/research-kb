#!/bin/bash
# Install llama-cpp-python with CUDA support
# Run with: sudo ./scripts/install_llama_cpp_cuda.sh
# Then activate venv and run: ./scripts/build_llama_cpp_cuda.sh

set -e

echo "=== Installing CUDA Toolkit for llama-cpp-python ==="
echo "This will install ~3GB of packages."
echo ""

# Install CUDA development toolkit
apt-get update
apt-get install -y nvidia-cuda-toolkit

echo ""
echo "✓ CUDA toolkit installed"
echo ""
echo "Now run as your regular user:"
echo "  ./scripts/build_llama_cpp_cuda.sh"
