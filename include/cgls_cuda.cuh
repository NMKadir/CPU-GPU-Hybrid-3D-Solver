#pragma once

#include <vector>
#include <cstddef>

class CGLSSolverCUDA {
public:
    CGLSSolverCUDA(int N, double tol = 1e-6, int max_iter = 1000);
    ~CGLSSolverCUDA(); // Destructor to free GPU memory

    // Main solver function
    int solve(const std::vector<double>& h_b, std::vector<double>& h_x);

private:
    int N_;
    size_t size_;
    double tol_;
    int max_iter_;
    double inv_h2_;

    // Device (GPU) memory pointers
    double *d_x, *d_b, *d_r, *d_p, *d_Hp, *d_temp, *d_f;

    // Helper functions
    void allocate_device_memory();
    void free_device_memory();
};