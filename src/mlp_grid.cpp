#include "mlp_grid.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace phys {

void mlp_random_init(MLPWeights& w, const MLPDims& d, std::uint32_t seed, float scale) {
    w.W1.resize(d.H * d.In);
    w.b1.resize(d.H);
    w.W2.resize(d.Out * d.H);
    w.b2.resize(d.Out);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-scale, scale);
    for (auto& v : w.W1) v = U(rng);
    for (auto& v : w.b1) v = U(rng);
    for (auto& v : w.W2) v = U(rng);
    for (auto& v : w.b2) v = U(rng);
}

void make_grid_coords(const GridSpec& g, float t, CoordNorm norm, std::vector<float>& coords) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const std::size_t N = static_cast<std::size_t>(nx) * ny * nz;
    coords.resize(N * 4);
    auto norm_xyz = [&](float v, int n) -> float {
        if (n <= 1) return 0.0f;
        float u = v / float(n - 1); // [0,1]
        if (norm == CoordNorm::MinusOneToOne) return 2.f * u - 1.f;
        return u;
    };
    std::size_t idx = 0;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                coords[idx * 4 + 0] = norm_xyz(float(x), nx);
                coords[idx * 4 + 1] = norm_xyz(float(y), ny);
                coords[idx * 4 + 2] = norm_xyz(float(z), nz);
                coords[idx * 4 + 3] = (norm == CoordNorm::MinusOneToOne) ? t : (t + 0.5f);
                ++idx;
            }
        }
    }
}

void mlp_infer_cpu(const MLPDims& d, const MLPWeights& w, const float* coords, std::size_t N, float* out) {
    mlp_forward<ExecCpu>(coords, w.W1.data(), w.b1.data(), w.W2.data(), w.b2.data(), out, N, d.In, d.H, d.Out);
}

void mlp_infer_cuda(const MLPDims& d, const MLPWeights& w, const float* coords, std::size_t N, float* out) {
    mlp_forward<ExecCuda>(coords, w.W1.data(), w.b1.data(), w.W2.data(), w.b2.data(), out, N, d.In, d.H, d.Out);
}

void mlp_grid_infer_cpu(const GridSpec& g, const MLPGridConfig& cfg, const MLPWeights& w, float t, std::vector<float>& out) {
    std::vector<float> coords;
    make_grid_coords(g, t, cfg.norm, coords);
    const std::size_t N = static_cast<std::size_t>(g.nx) * g.ny * g.nz;
    out.resize(N * cfg.dims.Out);
    mlp_infer_cpu(cfg.dims, w, coords.data(), N, out.data());
}

void mlp_grid_infer_cuda(const GridSpec& g, const MLPGridConfig& cfg, const MLPWeights& w, float t, std::vector<float>& out) {
    std::vector<float> coords;
    make_grid_coords(g, t, cfg.norm, coords);
    const std::size_t N = static_cast<std::size_t>(g.nx) * g.ny * g.nz;
    out.resize(N * cfg.dims.Out);
    mlp_infer_cuda(cfg.dims, w, coords.data(), N, out.data());
}

static void split_outputs_to_fields(const std::vector<float>& y, std::size_t N,
                                    std::vector<float>& sigma,
                                    std::vector<float>& u) {
    sigma.resize(N);
    u.resize(3 * N);
    for (std::size_t i = 0; i < N; ++i) {
        sigma[i] = y[i * 4 + 0];
        u[i]             = y[i * 4 + 1];
        u[N + i]         = y[i * 4 + 2];
        u[2 * N + i]     = y[i * 4 + 3];
    }
}

void mlp_generate_fields_cpu(const GridSpec& g, const MLPGridConfig& cfg, const MLPWeights& w, float t, float dt,
                             std::vector<float>& sigma_tm1, std::vector<float>& sigma_t, std::vector<float>& sigma_tp1,
                             std::vector<float>& u_tm1, std::vector<float>& u_t, std::vector<float>& u_tp1) {
    const std::size_t N = static_cast<std::size_t>(g.nx) * g.ny * g.nz;
    std::vector<float> y_tm1, y_t, y_tp1;
    mlp_grid_infer_cpu(g, cfg, w, t - dt, y_tm1);
    mlp_grid_infer_cpu(g, cfg, w, t,      y_t);
    mlp_grid_infer_cpu(g, cfg, w, t + dt, y_tp1);
    split_outputs_to_fields(y_tm1, N, sigma_tm1, u_tm1);
    split_outputs_to_fields(y_t,   N, sigma_t,   u_t);
    split_outputs_to_fields(y_tp1, N, sigma_tp1, u_tp1);
}

void mlp_generate_fields_cuda(const GridSpec& g, const MLPGridConfig& cfg, const MLPWeights& w, float t, float dt,
                              std::vector<float>& sigma_tm1, std::vector<float>& sigma_t, std::vector<float>& sigma_tp1,
                              std::vector<float>& u_tm1, std::vector<float>& u_t, std::vector<float>& u_tp1) {
    const std::size_t N = static_cast<std::size_t>(g.nx) * g.ny * g.nz;
    std::vector<float> y_tm1, y_t, y_tp1;
    mlp_grid_infer_cuda(g, cfg, w, t - dt, y_tm1);
    mlp_grid_infer_cuda(g, cfg, w, t,      y_t);
    mlp_grid_infer_cuda(g, cfg, w, t + dt, y_tp1);
    split_outputs_to_fields(y_tm1, N, sigma_tm1, u_tm1);
    split_outputs_to_fields(y_t,   N, sigma_t,   u_t);
    split_outputs_to_fields(y_tp1, N, sigma_tp1, u_tp1);
}

} // namespace phys

