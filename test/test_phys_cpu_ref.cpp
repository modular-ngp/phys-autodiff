// CPU reference tests for physics residuals and loss using manufactured solutions
#include "phys.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace phys;

static inline size_t idx3(int x, int y, int z, int nx, int ny) {
    return static_cast<size_t>((z * ny + y) * nx + x);
}

int main() {
    GridSpec g;
    g.nx = 64; g.ny = 64; g.nz = 32;
    const float PI = 3.14159265358979323846f;
    const float L = 2.0f * PI;
    g.hx = L / g.nx; g.hy = L / g.ny; g.hz = L / g.nz;
    g.dt = 1e-3f; // small for accurate central time diff
    g.periodic = true;

    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const size_t N = static_cast<size_t>(nx) * ny * nz;
    std::vector<float> sigma_tm1(N), sigma_t(N), sigma_tp1(N);
    std::vector<float> u_tm1(3 * N), u_t(3 * N), u_tp1(3 * N);

    auto off = [&](size_t i, int c) { return static_cast<size_t>(c) * N + i; };

    float t = 1.2345f;
    float tm1 = t - g.dt, tp1 = t + g.dt;

    // Manufactured solution 1:
    // sigma = sin(x + y + z - t), u = (1,1,1)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                float xf = x * g.hx, yf = y * g.hy, zf = z * g.hz;
                size_t i = idx3(x, y, z, nx, ny);
                sigma_tm1[i] = std::sinf(xf + yf + zf - tm1);
                sigma_t[i]   = std::sinf(xf + yf + zf - t);
                sigma_tp1[i] = std::sinf(xf + yf + zf - tp1);
                // u constant in space/time
                u_tm1[off(i,0)] = 1.f; u_tm1[off(i,1)] = 1.f; u_tm1[off(i,2)] = 1.f;
                u_t  [off(i,0)] = 1.f; u_t  [off(i,1)] = 1.f; u_t  [off(i,2)] = 1.f;
                u_tp1[off(i,0)] = 1.f; u_tp1[off(i,1)] = 1.f; u_tp1[off(i,2)] = 1.f;
            }
        }
    }

    std::vector<float> R_sigma(N), R_ux(N), R_uy(N), R_uz(N);
    cpu_phys_residuals(g, sigma_tm1.data(), sigma_t.data(), sigma_tp1.data(),
                       u_tm1.data(), u_t.data(), u_tp1.data(),
                       R_sigma.data(), R_ux.data(), R_uy.data(), R_uz.data());

    // Expected (discrete central differences):
    // dt_sigma = -cos(phi) * sin(dt)/dt
    // u·∇sigma = cos(phi) * (sin(hx)/hx + sin(hy)/hy + sin(hz)/hz)
    // R_sigma = dt_sigma + u·∇sigma, R_u = 0 for constant u
    const float c_dt = std::sinf(g.dt) / g.dt;
    const float c_hx = std::sinf(g.hx) / g.hx;
    const float c_hy = std::sinf(g.hy) / g.hy;
    const float c_hz = std::sinf(g.hz) / g.hz;
    double num = 0.0, den = 0.0;
    double max_abs = 0.0;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                float xf = x * g.hx, yf = y * g.hy, zf = z * g.hz;
                size_t i = idx3(x, y, z, nx, ny);
                float phi = xf + yf + zf - t;
                float ref = -std::cosf(phi) * c_dt + std::cosf(phi) * (c_hx + c_hy + c_hz);
                float diff = R_sigma[i] - ref;
                num += double(diff) * diff;
                den += double(ref) * ref;
                max_abs = std::max(max_abs, std::abs(double(diff)));
                if (std::abs(R_ux[i]) > 1e-6f || std::abs(R_uy[i]) > 1e-6f || std::abs(R_uz[i]) > 1e-6f) {
                    std::cerr << "R_u not ~0 at i=" << i << "\n";
                    return 2;
                }
            }
        }
    }

    double rel_l2 = std::sqrt(num / (den + 1e-30));
    std::cout << "rel_l2(R_sigma) vs discrete-analytic: " << rel_l2 << " max_abs: " << max_abs << "\n";
    // Allow small mismatch due to float cancellation in central time differencing on precomputed float fields
    if (rel_l2 > 3e-4 || max_abs > 1e-3) {
        std::cerr << "[FAIL] residual mismatch exceeds tolerance\n";
        return 1;
    }

    // Also test loss and backward scaling
    PhysWeights w; w.w_sigma = 1.7f; w.w_u = 0.9f;
    float Ls = 0.f, Lu = 0.f;
    cpu_phys_loss_forward(g, w,
                          sigma_tm1.data(), sigma_t.data(), sigma_tp1.data(),
                          u_tm1.data(), u_t.data(), u_tp1.data(),
                          &Ls, &Lu,
                          nullptr, nullptr, nullptr, nullptr);
    std::vector<float> gS(N), gUx(N), gUy(N), gUz(N);
    cpu_phys_loss_backward(g, w,
                           R_sigma.data(), R_ux.data(), R_uy.data(), R_uz.data(),
                           gS.data(), gUx.data(), gUy.data(), gUz.data());
    // Check gS = 2*w_sigma/N * R_sigma; and gU = 0 here
    double rel_l2_g = 0.0, den_g = 0.0, max_abs_g = 0.0;
    double scale = 2.0 * w.w_sigma / double(N);
    for (size_t i = 0; i < N; ++i) {
        double ref = scale * R_sigma[i];
        double diff = double(gS[i]) - ref;
        rel_l2_g += diff * diff;
        den_g += ref * ref;
        max_abs_g = std::max(max_abs_g, std::abs(diff));
        if (std::abs(gUx[i]) > 1e-7 || std::abs(gUy[i]) > 1e-7 || std::abs(gUz[i]) > 1e-7) {
            std::cerr << "g_u not ~0 at i=" << i << "\n";
            return 3;
        }
    }
    rel_l2_g = std::sqrt(rel_l2_g / (den_g + 1e-30));
    std::cout << "rel_l2(g_sigma) vs ref: " << rel_l2_g << " max_abs: " << max_abs_g << "\n";
    if (rel_l2_g > 1e-7 || max_abs_g > 1e-6) {
        std::cerr << "[FAIL] g_sigma mismatch exceeds tolerance\n";
        return 4;
    }

    std::cout << "[PASS] test_phys_cpu_ref\n";
    return 0;
}
