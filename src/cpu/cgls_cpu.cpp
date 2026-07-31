#include "cgls_cpu.hpp"
#include <cmath>
#include <iostream>

CGLSSolverCPU::CGLSSolverCPU(int N, double tol, int max_iter)
    : N_(N), size_(static_cast<size_t>(N) * N * N), tol_(tol), max_iter_(max_iter) {
    double h = 1.0 / (N + 1);
    inv_h2_ = 1.0 / (h * h);
}

// Applies the 3D Poisson 7-point stencil: A * x
void CGLSSolverCPU::apply_A(const std::vector<double>& x, std::vector<double>& y) const {
    for (int k = 0; k < N_; ++k) {
        for (int j = 0; j < N_; ++j) {
            for (int i = 0; i < N_; ++i) {
                size_t idx = i + N_ * (j + N_ * k);
                double val = 6.0 * x[idx];

                // Boundary-checked 7-point stencil (Dirichlet boundary conditions)
                if (i > 0)      val -= x[(i - 1) + N_ * (j + N_ * k)];
                if (i < N_ - 1) val -= x[(i + 1) + N_ * (j + N_ * k)];
                if (j > 0)      val -= x[i + N_ * ((j - 1) + N_ * k)];
                if (j < N_ - 1) val -= x[i + N_ * ((j + 1) + N_ * k)];
                if (k > 0)      val -= x[i + N_ * (j + N_ * (k - 1))];
                if (k < N_ - 1) val -= x[i + N_ * (j + N_ * (k + 1))];

                y[idx] = val * inv_h2_;
            }
        }
    }
}

// Applies H = A^T * A. (Since Poisson matrix A is symmetric, H = A * A)
void CGLSSolverCPU::apply_H(const std::vector<double>& x, std::vector<double>& y, std::vector<double>& temp) const {
    apply_A(x, temp);
    apply_A(temp, y);
}

double CGLSSolverCPU::dot(const std::vector<double>& u, const std::vector<double>& v) const {
    double result = 0.0;
    for (size_t i = 0; i < size_; ++i) {
        result += u[i] * v[i];
    }
    return result;
}

int CGLSSolverCPU::solve(const std::vector<double>& b, std::vector<double>& x) {
    std::vector<double> temp(size_, 0.0);
    std::vector<double> f(size_, 0.0);
    std::vector<double> r(size_, 0.0);
    std::vector<double> p(size_, 0.0);
    std::vector<double> Hp(size_, 0.0);

    // Line 1: x^{(0)} is already passed in x (initialized by caller)

    // Line 2: f = A^T * b (Since A = A^T, f = A * b)
    apply_A(b, f);

    // Line 2: r^{(0)} = f - H * x^{(0)}
    apply_H(x, Hp, temp); // Temporary use of Hp to compute H * x^(0)
    for (size_t i = 0; i < size_; ++i) {
        r[i] = f[i] - Hp[i];
    }

    // Line 3: p^{(0)} = r^{(0)}
    p = r;

    // r^(k)T * r^(k)
    double gamma = dot(r, r);

    // Line 4: for k = 0, 1, 2, ...
    for (int k = 0; k < max_iter_; ++k) {
        // Precompute Hp^{(k)} = H * p^{(k)}
        apply_H(p, Hp, temp);

        // Line 5: alpha_k = (r^{(k)T} * r^{(k)}) / (p^{(k)T} * H * p^{(k)})
        double p_Hp = dot(p, Hp);
        double alpha = gamma / p_Hp;

        // Line 6 & 7: Update x and r
        for (size_t i = 0; i < size_; ++i) {
            x[i] += alpha * p[i];   // x^{(k+1)} = x^{(k)} + alpha_k * p^{(k)}
            r[i] -= alpha * Hp[i];  // r^{(k+1)} = r^{(k)} - alpha_k * H * p^{(k)}
        }

        // Line 8: Compute norm and check if ||r^{(k+1)}||_2 < epsilon
        double gamma_new = dot(r, r);
        double norm_r = std::sqrt(gamma_new);

        if (norm_r < tol_) {
            std::cout << "[CPU CGLS] Converged at iteration " << k + 1 
                      << " with norm ||r|| = " << norm_r << "\n";
            return k + 1; // Line 9: stop
        }

        // Line 11: beta_k = (r^{(k+1)T} * r^{(k+1)}) / (r^{(k)T} * r^{(k)})
        double beta = gamma_new / gamma;

        // Line 12: p^{(k+1)} = r^{(k+1)} + beta_k * p^{(k)}
        for (size_t i = 0; i < size_; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        // Update gamma for next iteration
        gamma = gamma_new;
    }

    std::cout << "[CPU CGLS] Reached maximum iterations (" << max_iter_ << ") without full convergence.\n";
    return max_iter_;
}