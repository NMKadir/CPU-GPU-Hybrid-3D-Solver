#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <cmath>

namespace utils {

void save_vtk(const std::string& filename, const std::vector<double>& x, int N) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    // VTK Header for Structured Points
    out << "# vtk DataFile Version 3.0\n";
    out << "3D Poisson Solution\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_POINTS\n";
    
    // Define the grid dimensions and spacing
    out << "DIMENSIONS " << N << " " << N << " " << N << "\n";
    out << "ORIGIN 0 0 0\n";
    
    double h = 1.0 / (N + 1);
    out << "SPACING " << h << " " << h << " " << h << "\n";
    
    // Write the data points
    int total_points = N * N * N;
    out << "POINT_DATA " << total_points << "\n";
    out << "SCALARS potential double 1\n";
    out << "LOOKUP_TABLE default\n";

    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                size_t idx = i + N * (j + N * k);
                out << x[idx] << "\n";
            }
        }
    }

    out.close();
    std::cout << "Successfully saved 3D solution to " << filename << "\n";
}

bool compare_results(const std::vector<double>& cpu_x, const std::vector<double>& gpu_x, double tolerance) {
    if (cpu_x.size() != gpu_x.size()) {
        std::cerr << "Error: Vector sizes do not match!\n";
        return false;
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < cpu_x.size(); ++i) {
        double diff = std::abs(cpu_x[i] - gpu_x[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    if (max_diff > tolerance) {
        std::cerr << "[FAILED] GPU results differ from CPU. Max difference: " << max_diff << "\n";
        return false;
    }

    std::cout << "[PASSED] GPU results match CPU exactly! (Max diff: " << max_diff << ")\n";
    return true;
}

} // namespace utils