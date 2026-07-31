#pragma once

#include <vector>
#include <string>

namespace utils {
    // Exports the 1D solution vector into a 3D VTK file for visualization in ParaView
    void save_vtk(const std::string& filename, const std::vector<double>& x, int N);

    // Compares CPU and GPU outputs to ensure the CUDA math is perfectly accurate
    bool compare_results(const std::vector<double>& cpu_x, const std::vector<double>& gpu_x, double tolerance = 1e-5);
}