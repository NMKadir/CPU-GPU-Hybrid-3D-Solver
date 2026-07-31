#include "cgls_cuda.cuh"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

// ---------------------------------------------------------
// CUDA KERNELS
// ---------------------------------------------------------

// Applies the 7-point 3D Poisson Stencil
__global__ void apply_A_kernel(const double* x, double* y, int N, double inv_h2) {
    // Map thread/block indices to 3D grid coordinates
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < N && j < N && k < N) {
        size_t idx = i + N * (j + N * k);
        double val = 6.0 * x[idx];

        // Boundary-checked 7-point stencil
        if (i > 0)     val -= x[(i - 1) + N * (j + N * k)];
        if (i < N - 1) val -= x[(i + 1) + N * (j + N * k)];
        if (j > 0)     val -= x[i + N * ((j - 1) + N * k)];
        if (j < N - 1) val -= x[i + N * ((j + 1) + N * k)];
        if (k > 0)     val -= x[i + N * (j + N * (k - 1))];
        if (k < N - 1) val -= x[i + N * (j + N * (k + 1))];

        y[idx] = val * inv_h2;
    }
}

// Computes r_0 = f - Hp
__global__ void init_r_kernel(double* r, const double* f, const double* Hp, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) r[idx] = f[idx] - Hp[idx];
}

// Updates x and r (Lines 6 & 7)
__global__ void update_x_r_kernel(double* x, double* r, const double* p, const double* Hp, double alpha, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += alpha * p[idx];
        r[idx] -= alpha * Hp[idx];
    }
}

// Updates p (Line 12)
__global__ void update_p_kernel(double* p, const double* r, double beta, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        p[idx] = r[idx] + beta * p[idx];
    }
}

// ---------------------------------------------------------
// HOST CLASS IMPLEMENTATION
// ---------------------------------------------------------

CGLSSolverCUDA::CGLSSolverCUDA(int N, double tol, int max_iter)
    : N_(N), size_(static_cast<size_t>(N) * N * N), tol_(tol), max_iter_(max_iter) {
    double h = 1.0 / (N + 1);
    inv_h2_ = 1.0 / (h * h);
    allocate_device_memory();
}

CGLSSolverCUDA::~CGLSSolverCUDA() {
    free_device_memory();
}

void CGLSSolverCUDA::allocate_device_memory() {
    size_t bytes = size_ * sizeof(double);
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_r, bytes);
    cudaMalloc(&d_p, bytes);
    cudaMalloc(&d_Hp, bytes);
    cudaMalloc(&d_temp, bytes);
    cudaMalloc(&d_f, bytes);
}

void CGLSSolverCUDA::free_device_memory() {
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_r);
    cudaFree(d_p); cudaFree(d_Hp); cudaFree(d_temp); cudaFree(d_f);
}

int CGLSSolverCUDA::solve(const std::vector<double>& h_b, std::vector<double>& h_x) {
    size_t bytes = size_ * sizeof(double);

    // Copy initial data from Host (CPU) to Device (GPU)
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);

    // Set up kernel execution grids
    dim3 threads3D(8, 8, 8); // 512 threads per block (good for MX230)
    dim3 blocks3D((N_ + threads3D.x - 1) / threads3D.x,
                  (N_ + threads3D.y - 1) / threads3D.y,
                  (N_ + threads3D.z - 1) / threads3D.z);

    int threads1D = 256;
    int blocks1D = (size_ + threads1D - 1) / threads1D;

    // Thrust pointers for dot products
    thrust::device_ptr<double> ptr_r(d_r);
    thrust::device_ptr<double> ptr_p(d_p);
    thrust::device_ptr<double> ptr_Hp(d_Hp);

    // Line 2: f = A * b
    apply_A_kernel<<<blocks3D, threads3D>>>(d_b, d_f, N_, inv_h2_);
    
    // Line 2: Hp = H * x_0 (Apply A twice)
    apply_A_kernel<<<blocks3D, threads3D>>>(d_x, d_temp, N_, inv_h2_);
    apply_A_kernel<<<blocks3D, threads3D>>>(d_temp, d_Hp, N_, inv_h2_);

    // Line 2: r_0 = f - Hp
    init_r_kernel<<<blocks1D, threads1D>>>(d_r, d_f, d_Hp, size_);

    // Line 3: p_0 = r_0
    cudaMemcpy(d_p, d_r, bytes, cudaMemcpyDeviceToDevice);

    // Initial gamma
    double gamma = thrust::inner_product(ptr_r, ptr_r + size_, ptr_r, 0.0);

    for (int k = 0; k < max_iter_; ++k) {
        // Precompute Hp = H * p (Apply A twice)
        apply_A_kernel<<<blocks3D, threads3D>>>(d_p, d_temp, N_, inv_h2_);
        apply_A_kernel<<<blocks3D, threads3D>>>(d_temp, d_Hp, N_, inv_h2_);

        // Line 5: alpha
        double p_Hp = thrust::inner_product(ptr_p, ptr_p + size_, ptr_Hp, 0.0);
        double alpha = gamma / p_Hp;

        // Lines 6 & 7: Update x and r
        update_x_r_kernel<<<blocks1D, threads1D>>>(d_x, d_r, d_p, d_Hp, alpha, size_);

        // Line 8: Norm check
        double gamma_new = thrust::inner_product(ptr_r, ptr_r + size_, ptr_r, 0.0);
        double norm_r = std::sqrt(gamma_new);

        if (norm_r < tol_) {
            std::cout << "[CUDA CGLS] Converged at iteration " << k + 1 
                      << " with norm ||r|| = " << norm_r << "\n";
            // Copy final solution back to CPU
            cudaMemcpy(h_x.data(), d_x, bytes, cudaMemcpyDeviceToHost);
            return k + 1;
        }

        // Line 11: beta
        double beta = gamma_new / gamma;

        // Line 12: Update p
        update_p_kernel<<<blocks1D, threads1D>>>(d_p, d_r, beta, size_);

        gamma = gamma_new;
    }

    std::cout << "[CUDA CGLS] Reached max iterations.\n";
    cudaMemcpy(h_x.data(), d_x, bytes, cudaMemcpyDeviceToHost);
    return max_iter_;
}