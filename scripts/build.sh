#!/bin/bash
# scripts/build.sh

echo "Starting build process for CPU and GPU CGLS Solvers..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake and Make
cmake ..
make -j4

echo "========================================"
echo "Build complete! Executables are in the build/ directory."
echo "Run them using: ./build/main_cpu and ./build/main_cuda"