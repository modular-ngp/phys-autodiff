// Compare fused vs non-fused CUDA for residuals and backward
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
    g.nx = 96; g.ny = 64; g.nz = 48;
    const float PI = 3.14159265358979323846f;
    const float L = 2.0f * PI;
    g.hx = L / g.nx; g.hy = L / g.ny; g.hz = L / g.nz;
    g.dt = 1e-3f; g.periodic = true;

    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const size_t N = static_cast<size_t>(nx) * ny * nz;
    std::vector<float> sigma_tm1(N), sigma_t(N), sigma_tp1(N);
    std::vector<float> u_tm1(3 * N), u_t(3 * N), u_tp1(3 * N);
    auto off = [&](size_t i, int c) { return static_cast<size_t>(c) * N + i; };
    float t = 2.3456f; float tm1 = t - g.dt, tp1 = t + g.dt;
    for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
        float xf = x * g.hx, yf = y * g.hy, zf = z * g.hz; size_t i = idx3(x, y, z, nx, ny);
        sigma_tm1[i] = std::sinf(2*xf + 3*yf + 4*zf - tm1);
        sigma_t[i]   = std::sinf(2*xf + 3*yf + 4*zf - t);
        sigma_tp1[i] = std::sinf(2*xf + 3*yf + 4*zf - tp1);
        u_tm1[off(i,0)] = std::sinf(zf); u_tm1[off(i,1)] = std::cosf(xf); u_tm1[off(i,2)] = std::sinf(yf);
        u_t  [off(i,0)] = std::sinf(zf); u_t  [off(i,1)] = std::cosf(xf); u_t  [off(i,2)] = std::sinf(yf);
        u_tp1[off(i,0)] = std::sinf(zf); u_tp1[off(i,1)] = std::cosf(xf); u_tp1[off(i,2)] = std::sinf(yf);
    }

    std::vector<float> R_sigma_nf(N), R_ux_nf(N), R_uy_nf(N), R_uz_nf(N);
    cuda_phys_residuals_nonfused(g, sigma_tm1.data(), sigma_t.data(), sigma_tp1.data(),
                                 u_tm1.data(), u_t.data(), u_tp1.data(),
                                 R_sigma_nf.data(), R_ux_nf.data(), R_uy_nf.data(), R_uz_nf.data());

    std::vector<float> R_sigma_f(N), R_ux_f(N), R_uy_f(N), R_uz_f(N);
    cuda_phys_residuals_fused(g, sigma_tm1.data(), sigma_t.data(), sigma_tp1.data(),
                              u_tm1.data(), u_t.data(), u_tp1.data(),
                              R_sigma_f.data(), R_ux_f.data(), R_uy_f.data(), R_uz_f.data());

    double rel_s = rel_l2_err(R_sigma_nf, R_sigma_f);
    double rel_ux = rel_l2_err(R_ux_nf, R_ux_f);
    double rel_uy = rel_l2_err(R_uy_nf, R_uy_f);
    double rel_uz = rel_l2_err(R_uz_nf, R_uz_f);
    double max_s = max_abs_err(R_sigma_nf, R_sigma_f);
    double max_ux = max_abs_err(R_ux_nf, R_ux_f);
    double max_uy = max_abs_err(R_uy_nf, R_uy_f);
    double max_uz = max_abs_err(R_uz_nf, R_uz_f);

    std::cout << "residuals rel_l2: " << rel_s << ", " << rel_ux << ", " << rel_uy << ", " << rel_uz << "\n";
    std::cout << "residuals max_abs: " << max_s << ", " << max_ux << ", " << max_uy << ", " << max_uz << "\n";
    if (rel_s > 1e-7 || max_s > 1e-6 ||
        rel_ux > 1e-7 || max_ux > 1e-6 ||
        rel_uy > 1e-7 || max_uy > 1e-6 ||
        rel_uz > 1e-7 || max_uz > 1e-6) {
        std::cerr << "[FAIL] fused vs non-fused residual mismatch\n";
        return 1;
    }

    // Backward comparison
    PhysWeights w; w.w_sigma = 1.1f; w.w_u = 0.8f;
    std::vector<float> gS_nf(N), gUx_nf(N), gUy_nf(N), gUz_nf(N);
    cuda_phys_loss_backward_nonfused(g, w, R_sigma_nf.data(), R_ux_nf.data(), R_uy_nf.data(), R_uz_nf.data(),
                                     gS_nf.data(), gUx_nf.data(), gUy_nf.data(), gUz_nf.data());

    std::vector<float> gS_f(N), gUx_f(N), gUy_f(N), gUz_f(N);
    cuda_phys_loss_backward_fused(g, w, sigma_tm1.data(), sigma_t.data(), sigma_tp1.data(),
                                  u_tm1.data(), u_t.data(), u_tp1.data(),
                                  gS_f.data(), gUx_f.data(), gUy_f.data(), gUz_f.data());
    double rel_gs = rel_l2_err(gS_nf, gS_f);
    double rel_gx = rel_l2_err(gUx_nf, gUx_f);
    double rel_gy = rel_l2_err(gUy_nf, gUy_f);
    double rel_gz = rel_l2_err(gUz_nf, gUz_f);
    double max_gs = max_abs_err(gS_nf, gS_f);
    double max_gx = max_abs_err(gUx_nf, gUx_f);
    double max_gy = max_abs_err(gUy_nf, gUy_f);
    double max_gz = max_abs_err(gUz_nf, gUz_f);
    std::cout << "backward rel_l2: " << rel_gs << ", " << rel_gx << ", " << rel_gy << ", " << rel_gz << "\n";
    std::cout << "backward max_abs: " << max_gs << ", " << max_gx << ", " << max_gy << ", " << max_gz << "\n";
    if (rel_gs > 1e-7 || max_gs > 1e-6 ||
        rel_gx > 1e-7 || max_gx > 1e-6 ||
        rel_gy > 1e-7 || max_gy > 1e-6 ||
        rel_gz > 1e-7 || max_gz > 1e-6) {
        std::cerr << "[FAIL] fused vs non-fused backward mismatch\n";
        return 2;
    }

    std::cout << "[PASS] test_phys_cuda_fused_vs_nonfused\n";
    return 0;
}

