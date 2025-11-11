// Performance benchmark: fused vs non-fused CUDA (end-to-end wrapper time)
#include "phys.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace phys;

static inline size_t idx3(int x, int y, int z, int nx, int ny) {
    return static_cast<size_t>((z * ny + y) * nx + x);
}

struct Fields {
    std::vector<float> sigma_tm1, sigma_t, sigma_tp1;
    std::vector<float> u_tm1, u_t, u_tp1; // 3*N each
};

static Fields make_fields(const GridSpec& g) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const size_t N = static_cast<size_t>(nx) * ny * nz;
    Fields f{std::vector<float>(N), std::vector<float>(N), std::vector<float>(N),
             std::vector<float>(3*N), std::vector<float>(3*N), std::vector<float>(3*N)};
    auto off = [&](size_t i, int c) { return static_cast<size_t>(c) * N + i; };
    float t = 1.2345f; float tm1 = t - g.dt, tp1 = t + g.dt;
    for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
        float xf = x * g.hx, yf = y * g.hy, zf = z * g.hz; size_t i = idx3(x, y, z, nx, ny);
        f.sigma_tm1[i] = std::sinf(2*xf + 3*yf + 4*zf - tm1);
        f.sigma_t[i]   = std::sinf(2*xf + 3*yf + 4*zf - t);
        f.sigma_tp1[i] = std::sinf(2*xf + 3*yf + 4*zf - tp1);
        f.u_tm1[off(i,0)] = std::sinf(zf); f.u_tm1[off(i,1)] = std::cosf(xf); f.u_tm1[off(i,2)] = std::sinf(yf);
        f.u_t  [off(i,0)] = std::sinf(zf); f.u_t  [off(i,1)] = std::cosf(xf); f.u_t  [off(i,2)] = std::sinf(yf);
        f.u_tp1[off(i,0)] = std::sinf(zf); f.u_tp1[off(i,1)] = std::cosf(xf); f.u_tp1[off(i,2)] = std::sinf(yf);
    }
    return f;
}

static double bench_nonfused(const GridSpec& g, const Fields& f, int iters, int warmup) {
    const size_t N = (size_t)g.nx * g.ny * g.nz;
    std::vector<float> R_sigma(N), R_ux(N), R_uy(N), R_uz(N);
    for (int i = 0; i < warmup; ++i) {
        cuda_phys_residuals_nonfused(g, f.sigma_tm1.data(), f.sigma_t.data(), f.sigma_tp1.data(),
                                     f.u_tm1.data(), f.u_t.data(), f.u_tp1.data(),
                                     R_sigma.data(), R_ux.data(), R_uy.data(), R_uz.data());
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        cuda_phys_residuals_nonfused(g, f.sigma_tm1.data(), f.sigma_t.data(), f.sigma_tp1.data(),
                                     f.u_tm1.data(), f.u_t.data(), f.u_tp1.data(),
                                     R_sigma.data(), R_ux.data(), R_uy.data(), R_uz.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / double(iters);
}

static double bench_fused(const GridSpec& g, const Fields& f, int iters, int warmup) {
    const size_t N = (size_t)g.nx * g.ny * g.nz;
    std::vector<float> R_sigma(N), R_ux(N), R_uy(N), R_uz(N);
    for (int i = 0; i < warmup; ++i) {
        cuda_phys_residuals_fused(g, f.sigma_tm1.data(), f.sigma_t.data(), f.sigma_tp1.data(),
                                  f.u_tm1.data(), f.u_t.data(), f.u_tp1.data(),
                                  R_sigma.data(), R_ux.data(), R_uy.data(), R_uz.data());
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        cuda_phys_residuals_fused(g, f.sigma_tm1.data(), f.sigma_t.data(), f.sigma_tp1.data(),
                                  f.u_tm1.data(), f.u_t.data(), f.u_tp1.data(),
                                  R_sigma.data(), R_ux.data(), R_uy.data(), R_uz.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / double(iters);
}

int main(int argc, char** argv) {
    std::cout << "test,mode,nx,ny,nz,iters,ms\n";
    int iters = 10, warmup = 2;
    int sizes[][3] = {{64,64,64},{96,96,64},{128,96,96}};
    int num = sizeof(sizes)/sizeof(sizes[0]);
    for (int s = 0; s < num; ++s) {
        GridSpec g; g.nx = sizes[s][0]; g.ny = sizes[s][1]; g.nz = sizes[s][2];
        g.hx = 2*3.14159265358979323846f / g.nx; g.hy = 2*3.14159265358979323846f / g.ny; g.hz = 2*3.14159265358979323846f / g.nz;
        g.dt = 1e-3f; g.periodic = true;
        auto f = make_fields(g);
        double ms_nf = bench_nonfused(g, f, iters, warmup);
        double ms_f  = bench_fused(g, f, iters, warmup);
        std::cout << "phys,residuals_nonfused," << g.nx << "," << g.ny << "," << g.nz << "," << iters << "," << ms_nf << "\n";
        std::cout << "phys,residuals_fused," << g.nx << "," << g.ny << "," << g.nz << "," << iters << "," << ms_f << "\n";
    }
    return 0;
}
