#include "cgls_cpu.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include "utils.hpp" // Add this at the top


int main(int argc, char** argv) {
    // If a number is passed in the terminal, use it. Otherwise, default to 32.
    int N = (argc > 1) ? std::stoi(argv[1]) : 32;
    size_t total_points = N * N * N;

    // Right-hand side b (unit source term)
    std::vector<double> b(total_points, 1.0);

    // Initial guess x^(0) = 0
    std::vector<double> x(total_points, 0.0);

    CGLSSolverCPU solver(N, 1e-6, 1000);

    std::cout << "Starting CGLS Solve on CPU for grid size " << N << "x" << N << "x" << N << "...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    int iters = solver.solve(b, x);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution Time: " << elapsed.count() << " seconds\n";

    // ... inside main(), after the solver finishes:
    utils::save_vtk("solution_gpu.vtk", x, N);
    return 0;
}