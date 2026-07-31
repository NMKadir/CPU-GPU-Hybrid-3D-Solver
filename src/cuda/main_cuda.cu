#include "cgls_cuda.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include "utils.hpp" // Add this at the top


int main(int argc, char** argv) {
    // If a number is passed in the terminal, use it. Otherwise, default to 32.
    int N = (argc > 1) ? std::stoi(argv[1]) : 32;
    size_t total_points = N * N * N;

    std::vector<double> h_b(total_points, 1.0);
    std::vector<double> h_x(total_points, 0.0);

    // Initialize solver (allocates GPU memory)
    CGLSSolverCUDA solver(N, 1e-6, 1000);

    std::cout << "Starting CGLS Solve on GPU for grid size " << N << "x" << N << "x" << N << "...\n";
    
    // Time the solve process (including memory transfers if you want a strict application profile)
    auto start = std::chrono::high_resolution_clock::now();
    int iters = solver.solve(h_b, h_x);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "GPU Execution Time: " << elapsed.count() << " seconds\n";

    // ... inside main(), after the solver finishes:
    utils::save_vtk("solution_gpu.vtk", h_x, N);
    return 0;
}