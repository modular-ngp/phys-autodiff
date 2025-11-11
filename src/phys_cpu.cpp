#include "phys.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace phys {

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline int wrapi(int v, int n) {
    int r = v % n;
    return r < 0 ? r + n : r;
}

static inline size_t index3(int x, int y, int z, int nx, int ny, int nz) {
    return static_cast<size_t>((z * ny + y) * nx + x);
}

static inline size_t vec_off(size_t idx, size_t N, int c) {
    return static_cast<size_t>(c) * N + idx; // channel-major
}

void cpu_phys_residuals(const GridSpec& g,
                        const float* sigma_tm1,
                        const float* sigma_t,
                        const float* sigma_tp1,
                        const float* u_tm1,
                        const float* u_t,
                        const float* u_tp1,
                        float* R_sigma,
                        float* R_ux,
                        float* R_uy,
                        float* R_uz) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const size_t N = static_cast<size_t>(nx) * ny * nz;
    const double inv2dt = 1.0 / (2.0 * double(g.dt));
    const double inv2hx = 1.0 / (2.0 * double(g.hx));
    const double inv2hy = 1.0 / (2.0 * double(g.hy));
    const double inv2hz = 1.0 / (2.0 * double(g.hz));

    auto at = [&](const float* f, int x, int y, int z) -> float {
        if (g.periodic) {
            x = wrapi(x, nx); y = wrapi(y, ny); z = wrapi(z, nz);
        } else {
            x = clampi(x, 0, nx - 1);
            y = clampi(y, 0, ny - 1);
            z = clampi(z, 0, nz - 1);
        }
        return f[index3(x, y, z, nx, ny, nz)];
    };

    auto at_vec = [&](const float* v, int c, int x, int y, int z) -> float {
        if (g.periodic) {
            x = wrapi(x, nx); y = wrapi(y, ny); z = wrapi(z, nz);
        } else {
            x = clampi(x, 0, nx - 1);
            y = clampi(y, 0, ny - 1);
            z = clampi(z, 0, nz - 1);
        }
        size_t idx = index3(x, y, z, nx, ny, nz);
        return v[vec_off(idx, static_cast<size_t>(nx) * ny * nz, c)];
    };

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                size_t idx = index3(x, y, z, nx, ny, nz);
                // Time derivatives (central)
                double dt_sigma = (double(at(sigma_tp1, x, y, z)) - double(at(sigma_tm1, x, y, z))) * inv2dt;
                double ux_t = double(at_vec(u_t, 0, x, y, z));
                double uy_t = double(at_vec(u_t, 1, x, y, z));
                double uz_t = double(at_vec(u_t, 2, x, y, z));
                double dux_dt = (double(at_vec(u_tp1, 0, x, y, z)) - double(at_vec(u_tm1, 0, x, y, z))) * inv2dt;
                double duy_dt = (double(at_vec(u_tp1, 1, x, y, z)) - double(at_vec(u_tm1, 1, x, y, z))) * inv2dt;
                double duz_dt = (double(at_vec(u_tp1, 2, x, y, z)) - double(at_vec(u_tm1, 2, x, y, z))) * inv2dt;

                // Spatial gradients (central)
                double ds_dx = (double(at(sigma_t, x + 1, y, z)) - double(at(sigma_t, x - 1, y, z))) * inv2hx;
                double ds_dy = (double(at(sigma_t, x, y + 1, z)) - double(at(sigma_t, x, y - 1, z))) * inv2hy;
                double ds_dz = (double(at(sigma_t, x, y, z + 1)) - double(at(sigma_t, x, y, z - 1))) * inv2hz;

                double dux_dx = (double(at_vec(u_t, 0, x + 1, y, z)) - double(at_vec(u_t, 0, x - 1, y, z))) * inv2hx;
                double duy_dy = (double(at_vec(u_t, 1, x, y + 1, z)) - double(at_vec(u_t, 1, x, y - 1, z))) * inv2hy;
                double duz_dz = (double(at_vec(u_t, 2, x, y, z + 1)) - double(at_vec(u_t, 2, x, y, z - 1))) * inv2hz;

                double dux_dy = (double(at_vec(u_t, 0, x, y + 1, z)) - double(at_vec(u_t, 0, x, y - 1, z))) * inv2hy;
                double dux_dz = (double(at_vec(u_t, 0, x, y, z + 1)) - double(at_vec(u_t, 0, x, y, z - 1))) * inv2hz;
                double duy_dx = (double(at_vec(u_t, 1, x + 1, y, z)) - double(at_vec(u_t, 1, x - 1, y, z))) * inv2hx;
                double duy_dz = (double(at_vec(u_t, 1, x, y, z + 1)) - double(at_vec(u_t, 1, x, y, z - 1))) * inv2hz;
                double duz_dx = (double(at_vec(u_t, 2, x + 1, y, z)) - double(at_vec(u_t, 2, x - 1, y, z))) * inv2hx;
                double duz_dy = (double(at_vec(u_t, 2, x, y + 1, z)) - double(at_vec(u_t, 2, x, y - 1, z))) * inv2hy;

                // Divergence and advection
                double div_u = dux_dx + duy_dy + duz_dz;
                double adv_sigma = ux_t * ds_dx + uy_t * ds_dy + uz_t * ds_dz;
                // (u · ∇) u for each component
                double adv_ux = ux_t * dux_dx + uy_t * dux_dy + uz_t * dux_dz;
                double adv_uy = ux_t * duy_dx + uy_t * duy_dy + uz_t * duy_dz;
                double adv_uz = ux_t * duz_dx + uy_t * duz_dy + uz_t * duz_dz;

                R_sigma[idx] = float(dt_sigma + adv_sigma + double(at(sigma_t, x, y, z)) * div_u);
                R_ux[idx]    = float(dux_dt + adv_ux);
                R_uy[idx]    = float(duy_dt + adv_uy);
                R_uz[idx]    = float(duz_dt + adv_uz);
            }
        }
    }
}

void cpu_phys_loss_forward(const GridSpec& g,
                           const PhysWeights& w,
                           const float* sigma_tm1,
                           const float* sigma_t,
                           const float* sigma_tp1,
                           const float* u_tm1,
                           const float* u_t,
                           const float* u_tp1,
                           float* out_loss_sigma,
                           float* out_loss_u,
                           float* opt_R_sigma,
                           float* opt_R_ux,
                           float* opt_R_uy,
                           float* opt_R_uz) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const size_t N = static_cast<size_t>(nx) * ny * nz;
    std::vector<float> R_sigma, R_ux, R_uy, R_uz;
    float* r_s = opt_R_sigma;
    float* r_x = opt_R_ux;
    float* r_y = opt_R_uy;
    float* r_z = opt_R_uz;
    if (!r_s) { R_sigma.resize(N); r_s = R_sigma.data(); }
    if (!r_x) { R_ux.resize(N);    r_x = R_ux.data(); }
    if (!r_y) { R_uy.resize(N);    r_y = R_uy.data(); }
    if (!r_z) { R_uz.resize(N);    r_z = R_uz.data(); }

    cpu_phys_residuals(g, sigma_tm1, sigma_t, sigma_tp1, u_tm1, u_t, u_tp1, r_s, r_x, r_y, r_z);

    double acc_sigma = 0.0;
    double acc_u = 0.0;
    for (size_t i = 0; i < N; ++i) {
        acc_sigma += double(r_s[i]) * r_s[i];
        acc_u += double(r_x[i]) * r_x[i] + double(r_y[i]) * r_y[i] + double(r_z[i]) * r_z[i];
    }
    double invN = 1.0 / double(N);
    if (out_loss_sigma) *out_loss_sigma = float(w.w_sigma * acc_sigma * invN);
    if (out_loss_u)     *out_loss_u     = float(w.w_u * acc_u * invN);
}

void cpu_phys_loss_backward(const GridSpec& g,
                            const PhysWeights& w,
                            const float* R_sigma,
                            const float* R_ux,
                            const float* R_uy,
                            const float* R_uz,
                            float* g_sigma,
                            float* g_ux,
                            float* g_uy,
                            float* g_uz) {
    const size_t N = static_cast<size_t>(g.nx) * g.ny * g.nz;
    const float scale_sigma = 2.f * w.w_sigma / float(N);
    const float scale_u     = 2.f * w.w_u     / float(N);
    for (size_t i = 0; i < N; ++i) {
        g_sigma[i] = scale_sigma * R_sigma[i];
        g_ux[i]    = scale_u * R_ux[i];
        g_uy[i]    = scale_u * R_uy[i];
        g_uz[i]    = scale_u * R_uz[i];
    }
}

} // namespace phys
