cmake_minimum_required(VERSION 3.18)

# Enable both C++ and CUDA languages
project(HeteroPoissonCGLS LANGUAGES CXX CUDA)

# Require modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------
# HARDWARE OPTIMIZATION
# -------------------------------------------------------------
# The NVIDIA MX230 uses the Pascal architecture (Compute Capability 6.1)
set(CMAKE_CUDA_ARCHITECTURES "61")

# Add the include folder so compilers can find our .hpp and .cuh headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# -------------------------------------------------------------
# PHASE 1: CPU EXECUTABLE
# -------------------------------------------------------------
add_executable(main_cpu 
    src/cpu/main_cpu.cpp 
    src/cpu/cgls_cpu.cpp
    src/common/utils.cpp   # <-- Utility file linked here
)

# Add CPU optimization flag (-O3) for maximum serial performance
target_compile_options(main_cpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)

# -------------------------------------------------------------
# PHASE 2: CUDA EXECUTABLE
# -------------------------------------------------------------
add_executable(main_cuda 
    src/cuda/main_cuda.cu 
    src/cuda/cgls_cuda.cu
    src/common/utils.cpp   # <-- Utility file linked here as well
)

# Note: Thrust is included automatically with the CUDA Toolkit.