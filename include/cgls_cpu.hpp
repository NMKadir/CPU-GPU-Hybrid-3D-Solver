#pragma once

#include <vector>

class CGLSSolverCPU {
public:
    CGLSSolverCPU(int N, double tol = 1e-6, int max_iter = 1000);

    // Main solver
    int solve(const std::vector<double>& b, std::vector<double>& x);

private:
    int N_;           // Number of interior grid points along one dimension (N x N x N grid)
    size_t size_;     // Total grid points = N^3
    double tol_;      // Tolerance epsilon
    int max_iter_;    // Maximum iterations
    double inv_h2_;   // 1 / h^2 where h is grid spacing

    // 7-point 3D Poisson stencil operator: y = A * x
    void apply_A(const std::vector<double>& x, std::vector<double>& y) const;

    // Composite operator: y = H * x = A^T * A * x = A * A * x
    void apply_H(const std::vector<double>& x, std::vector<double>& y, std::vector<double>& temp) const;

    // Vector operations
    double dot(const std::vector<double>& u, const std::vector<double>& v) const;
};