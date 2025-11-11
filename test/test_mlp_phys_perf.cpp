#include "mlp_grid.h"
#include "phys.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace phys;

struct ResultRow {
    const char* mode; // "nonfused" or "fused"
    int nx, ny, nz;
    int iters;
    double ms_kernel;
    double ms_phys_e2e;
    double ms_mlp;
    double ms_total;
};

static void bench_case(int nx, int ny, int nz, int iters, int warmup, std::vector<ResultRow>& out_rows) {
    GridSpec g; g.nx = nx; g.ny = ny; g.nz = nz; g.hx = 1.f; g.hy = 1.f; g.hz = 1.f; g.dt = 2e-3f; g.periodic = true;
    MLPGridConfig cfg; cfg.dims = {4, 128, 4}; cfg.norm = CoordNorm::MinusOneToOne;
    MLPWeights w; mlp_random_init(w, cfg.dims, 777u, 0.25f);
    const size_t N = (size_t) nx * ny * nz;
    std::vector<float> s_tm1, s_t, s_tp1, u_tm1, u_t, u_tp1;
    std::vector<float> R_sigma(N), R_ux(N), R_uy(N), R_uz(N);

    // warmup
    for (int i = 0; i < warmup; ++i) {
        mlp_generate_fields_cuda(g, cfg, w, 0.25f, g.dt, s_tm1, s_t, s_tp1, u_tm1, u_t, u_tp1);
        float k_ms = 0.f;
        cuda_phys_residuals_nonfused_timed(g, s_tm1.data(), s_t.data(), s_tp1.data(), u_tm1.data(), u_t.data(), u_tp1.data(), R_sigma.data(), R_ux.data(), R_uy.data(), R_uz.data(), &k_ms);
        cuda_phys_residuals_fused_timed(g, s_tm1.data(), s_t.data(), s_tp1.data(), u_tm1.data(), u_t.data(), u_tp1.data(), R_sigma.data(), R_ux.data(), R_uy.data(), R_uz.data(), &k_ms);
    }

    // measure non-fused
    double acc_kernel_nf = 0.0, acc_phys_e2e_nf = 0.0, acc_mlp = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto m0 = std::chrono::steady_clock::now();
        mlp_generate_fields_cuda(g, cfg, w, 0.25f, g.dt, s_tm1, s_t, s_tp1, u_tm1, u_t, u_tp1);
        auto m1 = std::chrono::steady_clock::now();
        float k_ms = 0.f;
        auto p0 = std::chrono::steady_clock::now();
        cuda_phys_residuals_nonfused_timed(g, s_tm1.data(), s_t.data(), s_tp1.data(), u_tm1.data(), u_t.data(), u_tp1.data(), R_sigma.data(), R_ux.data(), R_uy.data(), R_uz.data(), &k_ms);
        auto p1 = std::chrono::steady_clock::now();
        acc_kernel_nf += k_ms;
        acc_phys_e2e_nf += std::chrono::duration<double, std::milli>(p1 - p0).count();
        acc_mlp += std::chrono::duration<double, std::milli>(m1 - m0).count();
    }
    out_rows.push_back({"nonfused", nx, ny, nz, iters, acc_kernel_nf / iters, acc_phys_e2e_nf / iters, acc_mlp / iters, (acc_mlp + acc_phys_e2e_nf) / iters});

    // measure fused
    double acc_kernel_f = 0.0, acc_phys_e2e_f = 0.0, acc_mlp_f = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto m0 = std::chrono::steady_clock::now();
        mlp_generate_fields_cuda(g, cfg, w, 0.25f, g.dt, s_tm1, s_t, s_tp1, u_tm1, u_t, u_tp1);
        auto m1 = std::chrono::steady_clock::now();
        float k_ms = 0.f;
        auto p0 = std::chrono::steady_clock::now();
        cuda_phys_residuals_fused_timed(g, s_tm1.data(), s_t.data(), s_tp1.data(), u_tm1.data(), u_t.data(), u_tp1.data(), R_sigma.data(), R_ux.data(), R_uy.data(), R_uz.data(), &k_ms);
        auto p1 = std::chrono::steady_clock::now();
        acc_kernel_f += k_ms;
        acc_phys_e2e_f += std::chrono::duration<double, std::milli>(p1 - p0).count();
        acc_mlp_f += std::chrono::duration<double, std::milli>(m1 - m0).count();
    }
    out_rows.push_back({"fused", nx, ny, nz, iters, acc_kernel_f / iters, acc_phys_e2e_f / iters, acc_mlp_f / iters, (acc_mlp_f + acc_phys_e2e_f) / iters});
}

int main(int argc, char** argv) {
    int iters = 10, warmup = 2;
    int sizes[][3] = {{64,64,64},{96,96,64},{128,96,96}};
    int num = sizeof(sizes)/sizeof(sizes[0]);
    std::vector<ResultRow> rows;
    for (int s = 0; s < num; ++s) bench_case(sizes[s][0], sizes[s][1], sizes[s][2], iters, warmup, rows);
    std::cout << "test,mode,nx,ny,nz,iters,ms_kernel,ms_phys_e2e,ms_mlp,ms_total\n";
    for (auto& r : rows) {
        std::cout << "mlp_phys," << r.mode << "," << r.nx << "," << r.ny << "," << r.nz << ","
                  << r.iters << "," << r.ms_kernel << "," << r.ms_phys_e2e << "," << r.ms_mlp << "," << r.ms_total << "\n";
    }
    return 0;
}

