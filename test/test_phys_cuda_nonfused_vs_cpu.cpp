// Compare CPU reference vs CUDA non-fused residuals and backward
#include "phys.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace phys;

static inline size_t idx3(int x, int y, int z, int nx, int ny) {
    return static_cast<size_t>((z * ny + y) * nx + x);
}

static double rel_l2_err(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = double(a[i]) - double(b[i]);
        num += d * d;
        den += double(a[i]) * double(a[i]);
    }
    return std::sqrt(num / (den + 1e-30));
}

static double max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(double(a[i]) - double(b[i])));
    return m;
}

int main() {
    GridSpec g;
    g.nx = 64; g.ny = 64; g.nz = 32;
    const float PI = 3.14159265358979323846f;
    const float L = 2.0f * PI;
    g.hx = L / g.nx; g.hy = L / g.ny; g.hz = L / g.nz;
    g.dt = 1e-3f; g.periodic = true;

    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const size_t N = static_cast<size_t>(nx) * ny * nz;
    std::vector<float> sigma_tm1(N), sigma_t(N), sigma_tp1(N);
    std::vector<float> u_tm1(3 * N), u_t(3 * N), u_tp1(3 * N);

    auto off = [&](size_t i, int c) { return static_cast<size_t>(c) * N + i; };

    float t = 1.2345f;
    float tm1 = t - g.dt, tp1 = t + g.dt;

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                float xf = x * g.hx, yf = y * g.hy, zf = z * g.hz;
                size_t i = idx3(x, y, z, nx, ny);
                sigma_tm1[i] = std::sinf(xf + yf + zf - tm1);
                sigma_t[i]   = std::sinf(xf + yf + zf - t);
                sigma_tp1[i] = std::sinf(xf + yf + zf - tp1);
                u_tm1[off(i,0)] = 1.f; u_tm1[off(i,1)] = 1.f; u_tm1[off(i,2)] = 1.f;
                u_t  [off(i,0)] = 1.f; u_t  [off(i,1)] = 1.f; u_t  [off(i,2)] = 1.f;
                u_tp1[off(i,0)] = 1.f; u_tp1[off(i,1)] = 1.f; u_tp1[off(i,2)] = 1.f;
            }
        }
    }

    std::vector<float> R_sigma_cpu(N), R_ux_cpu(N), R_uy_cpu(N), R_uz_cpu(N);
    cpu_phys_residuals(g, sigma_tm1.data(), sigma_t.data(), sigma_tp1.data(),
                       u_tm1.data(), u_t.data(), u_tp1.data(),
                       R_sigma_cpu.data(), R_ux_cpu.data(), R_uy_cpu.data(), R_uz_cpu.data());

    std::vector<float> R_sigma_gpu(N), R_ux_gpu(N), R_uy_gpu(N), R_uz_gpu(N);
    cuda_phys_residuals_nonfused(g, sigma_tm1.data(), sigma_t.data(), sigma_tp1.data(),
                                 u_tm1.data(), u_t.data(), u_tp1.data(),
                                 R_sigma_gpu.data(), R_ux_gpu.data(), R_uy_gpu.data(), R_uz_gpu.data());

    double rel_s = rel_l2_err(R_sigma_cpu, R_sigma_gpu);
    double rel_ux = rel_l2_err(R_ux_cpu, R_ux_gpu);
    double rel_uy = rel_l2_err(R_uy_cpu, R_uy_gpu);
    double rel_uz = rel_l2_err(R_uz_cpu, R_uz_gpu);
    double max_s = max_abs_err(R_sigma_cpu, R_sigma_gpu);
    double max_ux = max_abs_err(R_ux_cpu, R_ux_gpu);
    double max_uy = max_abs_err(R_uy_cpu, R_uy_gpu);
    double max_uz = max_abs_err(R_uz_cpu, R_uz_gpu);

    std::cout << "rel_l2 R_sigma: " << rel_s << " max_abs: " << max_s << "\n";
    std::cout << "rel_l2 R_ux: " << rel_ux << " max_abs: " << max_ux << "\n";
    std::cout << "rel_l2 R_uy: " << rel_uy << " max_abs: " << max_uy << "\n";
    std::cout << "rel_l2 R_uz: " << rel_uz << " max_abs: " << max_uz << "\n";

    if (rel_s > 3e-4 || max_s > 1e-3 ||
        rel_ux > 1e-7 || max_ux > 1e-6 ||
        rel_uy > 1e-7 || max_uy > 1e-6 ||
        rel_uz > 1e-7 || max_uz > 1e-6) {
        std::cerr << "[FAIL] CUDA non-fused vs CPU residual mismatch\n";
        return 1;
    }

    // Backward parity (scale residuals)
    PhysWeights w; w.w_sigma = 1.3f; w.w_u = 0.7f;
    std::vector<float> gS_cpu(N), gUx_cpu(N), gUy_cpu(N), gUz_cpu(N);
    cpu_phys_loss_backward(g, w, R_sigma_cpu.data(), R_ux_cpu.data(), R_uy_cpu.data(), R_uz_cpu.data(),
                           gS_cpu.data(), gUx_cpu.data(), gUy_cpu.data(), gUz_cpu.data());

    std::vector<float> gS_gpu(N), gUx_gpu(N), gUy_gpu(N), gUz_gpu(N);
    cuda_phys_loss_backward_nonfused(g, w, R_sigma_gpu.data(), R_ux_gpu.data(), R_uy_gpu.data(), R_uz_gpu.data(),
                                     gS_gpu.data(), gUx_gpu.data(), gUy_gpu.data(), gUz_gpu.data());

    double rel_gs = rel_l2_err(gS_cpu, gS_gpu);
    double max_gs = max_abs_err(gS_cpu, gS_gpu);
    std::cout << "rel_l2 g_sigma: " << rel_gs << " max_abs: " << max_gs << "\n";
    if (rel_gs > 1e-7 || max_gs > 1e-6) {
        std::cerr << "[FAIL] CUDA non-fused vs CPU backward mismatch\n";
        return 2;
    }

    std::cout << "[PASS] test_phys_cuda_nonfused_vs_cpu\n";
    return 0;
}

