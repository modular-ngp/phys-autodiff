#include "phys.h"
#include <cuda_runtime.h>
#include <vector>

namespace phys {

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ int wrapi(int v, int n) {
    int r = v % n;
    return r < 0 ? r + n : r;
}

__device__ __forceinline__ void unpack_idx(int idx, int nx, int ny, int nz, int& x, int& y, int& z) {
    int xy = nx * ny;
    z = idx / xy;
    int r = idx - z * xy;
    y = r / nx;
    x = r - y * nx;
}

__device__ __forceinline__ size_t off3(int x, int y, int z, int nx, int ny) {
    return (size_t)((z * ny + y) * nx + x);
}

__device__ __forceinline__ float at_scalar(const float* f, int x, int y, int z, int nx, int ny, int nz, bool periodic) {
    if (periodic) {
        x = wrapi(x, nx); y = wrapi(y, ny); z = wrapi(z, nz);
    } else {
        x = clampi(x, 0, nx - 1);
        y = clampi(y, 0, ny - 1);
        z = clampi(z, 0, nz - 1);
    }
    return f[off3(x, y, z, nx, ny)];
}

__device__ __forceinline__ float at_vec(const float* v, int c, int x, int y, int z, int nx, int ny, int nz, bool periodic, size_t N) {
    if (periodic) {
        x = wrapi(x, nx); y = wrapi(y, ny); z = wrapi(z, nz);
    } else {
        x = clampi(x, 0, nx - 1);
        y = clampi(y, 0, ny - 1);
        z = clampi(z, 0, nz - 1);
    }
    size_t idx = off3(x, y, z, nx, ny);
    return v[(size_t)c * N + idx];
}

__global__ void k_dt(int N,
                     const float* sigma_tm1, const float* sigma_tp1,
                     const float* u_tm1, const float* u_tp1,
                     float inv2dt,
                     float* dt_sigma,
                     float* dux_dt, float* duy_dt, float* duz_dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dt_sigma[i] = (sigma_tp1[i] - sigma_tm1[i]) * inv2dt;
    dux_dt[i]   = (u_tp1[i] - u_tm1[i]) * inv2dt;              // c=0 contiguous at offset 0
    duy_dt[i]   = (u_tp1[i + N] - u_tm1[i + N]) * inv2dt;      // c=1
    duz_dt[i]   = (u_tp1[i + 2 * N] - u_tm1[i + 2 * N]) * inv2dt; // c=2
}

__global__ void k_grad_sigma(int N, int nx, int ny, int nz,
                             float inv2hx, float inv2hy, float inv2hz,
                             bool periodic,
                             const float* sigma_t,
                             float* ds_dx, float* ds_dy, float* ds_dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int x, y, z; unpack_idx(i, nx, ny, nz, x, y, z);
    int xp = periodic ? (x + 1) % nx : min(x + 1, nx - 1);
    int xm = periodic ? (x + nx - 1) % nx : max(x - 1, 0);
    int yp = periodic ? (y + 1) % ny : min(y + 1, ny - 1);
    int ym = periodic ? (y + ny - 1) % ny : max(y - 1, 0);
    int zp = periodic ? (z + 1) % nz : min(z + 1, nz - 1);
    int zm = periodic ? (z + nz - 1) % nz : max(z - 1, 0);
    size_t i_xp = off3(xp, y, z, nx, ny), i_xm = off3(xm, y, z, nx, ny);
    size_t i_yp = off3(x, yp, z, nx, ny), i_ym = off3(x, ym, z, nx, ny);
    size_t i_zp = off3(x, y, zp, nx, ny), i_zm = off3(x, y, zm, nx, ny);
    ds_dx[i] = (sigma_t[i_xp] - sigma_t[i_xm]) * inv2hx;
    ds_dy[i] = (sigma_t[i_yp] - sigma_t[i_ym]) * inv2hy;
    ds_dz[i] = (sigma_t[i_zp] - sigma_t[i_zm]) * inv2hz;
}

__global__ void k_grad_u(int N, int nx, int ny, int nz,
                         float inv2hx, float inv2hy, float inv2hz,
                         bool periodic,
                         const float* u_t,
                         float* dux_dx, float* dux_dy, float* dux_dz,
                         float* duy_dx, float* duy_dy, float* duy_dz,
                         float* duz_dx, float* duz_dy, float* duz_dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int x, y, z; unpack_idx(i, nx, ny, nz, x, y, z);
    int xp = periodic ? (x + 1) % nx : min(x + 1, nx - 1);
    int xm = periodic ? (x + nx - 1) % nx : max(x - 1, 0);
    int yp = periodic ? (y + 1) % ny : min(y + 1, ny - 1);
    int ym = periodic ? (y + ny - 1) % ny : max(y - 1, 0);
    int zp = periodic ? (z + 1) % nz : min(z + 1, nz - 1);
    int zm = periodic ? (z + nz - 1) % nz : max(z - 1, 0);
    size_t Nsz = (size_t)N;
    // offsets for neighbor indices
    size_t i_xp = off3(xp, y, z, nx, ny), i_xm = off3(xm, y, z, nx, ny);
    size_t i_yp = off3(x, yp, z, nx, ny), i_ym = off3(x, ym, z, nx, ny);
    size_t i_zp = off3(x, y, zp, nx, ny), i_zm = off3(x, y, zm, nx, ny);
    // component 0 (ux)
    dux_dx[i] = (u_t[i_xp] - u_t[i_xm]) * inv2hx;
    dux_dy[i] = (u_t[i_yp] - u_t[i_ym]) * inv2hy;
    dux_dz[i] = (u_t[i_zp] - u_t[i_zm]) * inv2hz;
    // component 1 (uy)
    duy_dx[i] = (u_t[Nsz + i_xp] - u_t[Nsz + i_xm]) * inv2hx;
    duy_dy[i] = (u_t[Nsz + i_yp] - u_t[Nsz + i_ym]) * inv2hy;
    duy_dz[i] = (u_t[Nsz + i_zp] - u_t[Nsz + i_zm]) * inv2hz;
    // component 2 (uz)
    duz_dx[i] = (u_t[2 * Nsz + i_xp] - u_t[2 * Nsz + i_xm]) * inv2hx;
    duz_dy[i] = (u_t[2 * Nsz + i_yp] - u_t[2 * Nsz + i_ym]) * inv2hy;
    duz_dz[i] = (u_t[2 * Nsz + i_zp] - u_t[2 * Nsz + i_zm]) * inv2hz;
}

__global__ void k_residuals(int N, int nx, int ny, int nz,
                             const float* sigma_t,
                             const float* u_t,
                             const float* dt_sigma,
                             const float* dux_dt, const float* duy_dt, const float* duz_dt,
                             const float* ds_dx, const float* ds_dy, const float* ds_dz,
                             const float* dux_dx, const float* dux_dy, const float* dux_dz,
                             const float* duy_dx, const float* duy_dy, const float* duy_dz,
                             const float* duz_dx, const float* duz_dy, const float* duz_dz,
                             float* R_sigma, float* R_ux, float* R_uy, float* R_uz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    size_t Nsz = (size_t)N;
    float ux = u_t[i];
    float uy = u_t[Nsz + i];
    float uz = u_t[2 * Nsz + i];
    float div_u = dux_dx[i] + duy_dy[i] + duz_dz[i];
    float adv_sigma = ux * ds_dx[i] + uy * ds_dy[i] + uz * ds_dz[i];
    float adv_ux = ux * dux_dx[i] + uy * dux_dy[i] + uz * dux_dz[i];
    float adv_uy = ux * duy_dx[i] + uy * duy_dy[i] + uz * duy_dz[i];
    float adv_uz = ux * duz_dx[i] + uy * duz_dy[i] + uz * duz_dz[i];
    R_sigma[i] = dt_sigma[i] + adv_sigma + sigma_t[i] * div_u;
    R_ux[i]    = dux_dt[i] + adv_ux;
    R_uy[i]    = duy_dt[i] + adv_uy;
    R_uz[i]    = duz_dt[i] + adv_uz;
}

__global__ void k_backward_scales(int N, float scale_sigma, float scale_u,
                                  const float* R_sigma, const float* R_ux, const float* R_uy, const float* R_uz,
                                  float* g_sigma, float* g_ux, float* g_uy, float* g_uz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    g_sigma[i] = scale_sigma * R_sigma[i];
    g_ux[i] = scale_u * R_ux[i];
    g_uy[i] = scale_u * R_uy[i];
    g_uz[i] = scale_u * R_uz[i];
}

void cuda_phys_residuals_nonfused(const GridSpec& g,
                                  const float* h_sigma_tm1,
                                  const float* h_sigma_t,
                                  const float* h_sigma_tp1,
                                  const float* h_u_tm1,
                                  const float* h_u_t,
                                  const float* h_u_tp1,
                                  float* h_R_sigma,
                                  float* h_R_ux,
                                  float* h_R_uy,
                                  float* h_R_uz) {
    int nx = g.nx, ny = g.ny, nz = g.nz;
    size_t N = (size_t)nx * ny * nz;
    int Nint = (int)N;
    float *d_sigma_tm1, *d_sigma_t, *d_sigma_tp1;
    float *d_u_tm1, *d_u_t, *d_u_tp1;
    cudaMalloc(&d_sigma_tm1, N * sizeof(float));
    cudaMalloc(&d_sigma_t,   N * sizeof(float));
    cudaMalloc(&d_sigma_tp1, N * sizeof(float));
    cudaMalloc(&d_u_tm1, 3 * N * sizeof(float));
    cudaMalloc(&d_u_t,   3 * N * sizeof(float));
    cudaMalloc(&d_u_tp1, 3 * N * sizeof(float));
    cudaMemcpy(d_sigma_tm1, h_sigma_tm1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma_t,   h_sigma_t,   N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma_tp1, h_sigma_tp1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_tm1, h_u_tm1, 3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_t,   h_u_t,   3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_tp1, h_u_tp1, 3 * N * sizeof(float), cudaMemcpyHostToDevice);

    // intermediates
    float *dt_sigma, *dux_dt, *duy_dt, *duz_dt;
    float *ds_dx, *ds_dy, *ds_dz;
    float *dux_dx, *dux_dy, *dux_dz;
    float *duy_dx, *duy_dy, *duy_dz;
    float *duz_dx, *duz_dy, *duz_dz;
    float *R_sigma, *R_ux, *R_uy, *R_uz;
    cudaMalloc(&dt_sigma, N * sizeof(float));
    cudaMalloc(&dux_dt,   N * sizeof(float));
    cudaMalloc(&duy_dt,   N * sizeof(float));
    cudaMalloc(&duz_dt,   N * sizeof(float));
    cudaMalloc(&ds_dx, N * sizeof(float));
    cudaMalloc(&ds_dy, N * sizeof(float));
    cudaMalloc(&ds_dz, N * sizeof(float));
    cudaMalloc(&dux_dx, N * sizeof(float));
    cudaMalloc(&dux_dy, N * sizeof(float));
    cudaMalloc(&dux_dz, N * sizeof(float));
    cudaMalloc(&duy_dx, N * sizeof(float));
    cudaMalloc(&duy_dy, N * sizeof(float));
    cudaMalloc(&duy_dz, N * sizeof(float));
    cudaMalloc(&duz_dx, N * sizeof(float));
    cudaMalloc(&duz_dy, N * sizeof(float));
    cudaMalloc(&duz_dz, N * sizeof(float));
    cudaMalloc(&R_sigma, N * sizeof(float));
    cudaMalloc(&R_ux,    N * sizeof(float));
    cudaMalloc(&R_uy,    N * sizeof(float));
    cudaMalloc(&R_uz,    N * sizeof(float));

    int tb = 256;
    int blocks = (Nint + tb - 1) / tb;
    float inv2dt = 1.f / (2.f * g.dt);
    float inv2hx = 1.f / (2.f * g.hx);
    float inv2hy = 1.f / (2.f * g.hy);
    float inv2hz = 1.f / (2.f * g.hz);

    k_dt<<<blocks, tb>>>(Nint, d_sigma_tm1, d_sigma_tp1, d_u_tm1, d_u_tp1, inv2dt,
                         dt_sigma, dux_dt, duy_dt, duz_dt);
    k_grad_sigma<<<blocks, tb>>>(Nint, nx, ny, nz, inv2hx, inv2hy, inv2hz, g.periodic,
                                 d_sigma_t, ds_dx, ds_dy, ds_dz);
    k_grad_u<<<blocks, tb>>>(Nint, nx, ny, nz, inv2hx, inv2hy, inv2hz, g.periodic,
                             d_u_t,
                             dux_dx, dux_dy, dux_dz,
                             duy_dx, duy_dy, duy_dz,
                             duz_dx, duz_dy, duz_dz);
    k_residuals<<<blocks, tb>>>(Nint, nx, ny, nz,
                                d_sigma_t, d_u_t,
                                dt_sigma, dux_dt, duy_dt, duz_dt,
                                ds_dx, ds_dy, ds_dz,
                                dux_dx, dux_dy, dux_dz,
                                duy_dx, duy_dy, duy_dz,
                                duz_dx, duz_dy, duz_dz,
                                R_sigma, R_ux, R_uy, R_uz);

    cudaMemcpy(h_R_sigma, R_sigma, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R_ux,    R_ux,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R_uy,    R_uy,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R_uz,    R_uz,    N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_sigma_tm1); cudaFree(d_sigma_t); cudaFree(d_sigma_tp1);
    cudaFree(d_u_tm1); cudaFree(d_u_t); cudaFree(d_u_tp1);
    cudaFree(dt_sigma); cudaFree(dux_dt); cudaFree(duy_dt); cudaFree(duz_dt);
    cudaFree(ds_dx); cudaFree(ds_dy); cudaFree(ds_dz);
    cudaFree(dux_dx); cudaFree(dux_dy); cudaFree(dux_dz);
    cudaFree(duy_dx); cudaFree(duy_dy); cudaFree(duy_dz);
    cudaFree(duz_dx); cudaFree(duz_dy); cudaFree(duz_dz);
    cudaFree(R_sigma); cudaFree(R_ux); cudaFree(R_uy); cudaFree(R_uz);
}

void cuda_phys_residuals_nonfused_timed(const GridSpec& g,
                                        const float* h_sigma_tm1,
                                        const float* h_sigma_t,
                                        const float* h_sigma_tp1,
                                        const float* h_u_tm1,
                                        const float* h_u_t,
                                        const float* h_u_tp1,
                                        float* h_R_sigma,
                                        float* h_R_ux,
                                        float* h_R_uy,
                                        float* h_R_uz,
                                        float* kernel_ms) {
    int nx = g.nx, ny = g.ny, nz = g.nz;
    size_t N = (size_t)nx * ny * nz;
    int Nint = (int)N;
    float *d_sigma_tm1, *d_sigma_t, *d_sigma_tp1;
    float *d_u_tm1, *d_u_t, *d_u_tp1;
    cudaMalloc(&d_sigma_tm1, N * sizeof(float));
    cudaMalloc(&d_sigma_t,   N * sizeof(float));
    cudaMalloc(&d_sigma_tp1, N * sizeof(float));
    cudaMalloc(&d_u_tm1, 3 * N * sizeof(float));
    cudaMalloc(&d_u_t,   3 * N * sizeof(float));
    cudaMalloc(&d_u_tp1, 3 * N * sizeof(float));
    cudaMemcpy(d_sigma_tm1, h_sigma_tm1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma_t,   h_sigma_t,   N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma_tp1, h_sigma_tp1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_tm1, h_u_tm1, 3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_t,   h_u_t,   3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_tp1, h_u_tp1, 3 * N * sizeof(float), cudaMemcpyHostToDevice);

    // intermediates
    float *dt_sigma, *dux_dt, *duy_dt, *duz_dt;
    float *ds_dx, *ds_dy, *ds_dz;
    float *dux_dx, *dux_dy, *dux_dz;
    float *duy_dx, *duy_dy, *duy_dz;
    float *duz_dx, *duz_dy, *duz_dz;
    float *R_sigma, *R_ux, *R_uy, *R_uz;
    cudaMalloc(&dt_sigma, N * sizeof(float));
    cudaMalloc(&dux_dt,   N * sizeof(float));
    cudaMalloc(&duy_dt,   N * sizeof(float));
    cudaMalloc(&duz_dt,   N * sizeof(float));
    cudaMalloc(&ds_dx, N * sizeof(float));
    cudaMalloc(&ds_dy, N * sizeof(float));
    cudaMalloc(&ds_dz, N * sizeof(float));
    cudaMalloc(&dux_dx, N * sizeof(float));
    cudaMalloc(&dux_dy, N * sizeof(float));
    cudaMalloc(&dux_dz, N * sizeof(float));
    cudaMalloc(&duy_dx, N * sizeof(float));
    cudaMalloc(&duy_dy, N * sizeof(float));
    cudaMalloc(&duy_dz, N * sizeof(float));
    cudaMalloc(&duz_dx, N * sizeof(float));
    cudaMalloc(&duz_dy, N * sizeof(float));
    cudaMalloc(&duz_dz, N * sizeof(float));
    cudaMalloc(&R_sigma, N * sizeof(float));
    cudaMalloc(&R_ux,    N * sizeof(float));
    cudaMalloc(&R_uy,    N * sizeof(float));
    cudaMalloc(&R_uz,    N * sizeof(float));

    int tb = 256;
    int blocks = (Nint + tb - 1) / tb;
    float inv2dt = 1.f / (2.f * g.dt);
    float inv2hx = 1.f / (2.f * g.hx);
    float inv2hy = 1.f / (2.f * g.hy);
    float inv2hz = 1.f / (2.f * g.hz);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    k_dt<<<blocks, tb>>>(Nint, d_sigma_tm1, d_sigma_tp1, d_u_tm1, d_u_tp1, inv2dt,
                         dt_sigma, dux_dt, duy_dt, duz_dt);
    k_grad_sigma<<<blocks, tb>>>(Nint, nx, ny, nz, inv2hx, inv2hy, inv2hz, g.periodic,
                                 d_sigma_t, ds_dx, ds_dy, ds_dz);
    k_grad_u<<<blocks, tb>>>(Nint, nx, ny, nz, inv2hx, inv2hy, inv2hz, g.periodic,
                             d_u_t,
                             dux_dx, dux_dy, dux_dz,
                             duy_dx, duy_dy, duy_dz,
                             duz_dx, duz_dy, duz_dz);
    k_residuals<<<blocks, tb>>>(Nint, nx, ny, nz,
                                d_sigma_t, d_u_t,
                                dt_sigma, dux_dt, duy_dt, duz_dt,
                                ds_dx, ds_dy, ds_dz,
                                dux_dx, dux_dy, dux_dz,
                                duy_dx, duy_dy, duy_dz,
                                duz_dx, duz_dy, duz_dz,
                                R_sigma, R_ux, R_uy, R_uz);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
    if (kernel_ms) *kernel_ms = ms;
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    cudaMemcpy(h_R_sigma, R_sigma, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R_ux,    R_ux,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R_uy,    R_uy,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R_uz,    R_uz,    N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_sigma_tm1); cudaFree(d_sigma_t); cudaFree(d_sigma_tp1);
    cudaFree(d_u_tm1); cudaFree(d_u_t); cudaFree(d_u_tp1);
    cudaFree(dt_sigma); cudaFree(dux_dt); cudaFree(duy_dt); cudaFree(duz_dt);
    cudaFree(ds_dx); cudaFree(ds_dy); cudaFree(ds_dz);
    cudaFree(dux_dx); cudaFree(dux_dy); cudaFree(dux_dz);
    cudaFree(duy_dx); cudaFree(duy_dy); cudaFree(duy_dz);
    cudaFree(duz_dx); cudaFree(duz_dy); cudaFree(duz_dz);
    cudaFree(R_sigma); cudaFree(R_ux); cudaFree(R_uy); cudaFree(R_uz);
}

void cuda_phys_loss_forward_nonfused(const GridSpec& g,
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
    int nx = g.nx, ny = g.ny, nz = g.nz;
    size_t N = (size_t)nx * ny * nz;
    std::vector<float> R_sigma, R_ux, R_uy, R_uz;
    float *rS = opt_R_sigma, *rX = opt_R_ux, *rY = opt_R_uy, *rZ = opt_R_uz;
    if (!rS) { R_sigma.resize(N); rS = R_sigma.data(); }
    if (!rX) { R_ux.resize(N);    rX = R_ux.data(); }
    if (!rY) { R_uy.resize(N);    rY = R_uy.data(); }
    if (!rZ) { R_uz.resize(N);    rZ = R_uz.data(); }

    cuda_phys_residuals_nonfused(g, sigma_tm1, sigma_t, sigma_tp1, u_tm1, u_t, u_tp1, rS, rX, rY, rZ);
    double acc_s = 0.0, acc_u = 0.0;
    for (size_t i = 0; i < N; ++i) {
        acc_s += double(rS[i]) * rS[i];
        acc_u += double(rX[i]) * rX[i] + double(rY[i]) * rY[i] + double(rZ[i]) * rZ[i];
    }
    double invN = 1.0 / double(N);
    if (out_loss_sigma) *out_loss_sigma = float(w.w_sigma * acc_s * invN);
    if (out_loss_u)     *out_loss_u     = float(w.w_u * acc_u * invN);
}

void cuda_phys_loss_backward_nonfused(const GridSpec& g,
                                      const PhysWeights& w,
                                      const float* h_R_sigma,
                                      const float* h_R_ux,
                                      const float* h_R_uy,
                                      const float* h_R_uz,
                                      float* h_g_sigma,
                                      float* h_g_ux,
                                      float* h_g_uy,
                                      float* h_g_uz) {
    int nx = g.nx, ny = g.ny, nz = g.nz;
    size_t N = (size_t)nx * ny * nz;
    int Nint = (int)N;
    float *R_sigma, *R_ux, *R_uy, *R_uz;
    float *g_sigma, *g_ux, *g_uy, *g_uz;
    cudaMalloc(&R_sigma, N * sizeof(float));
    cudaMalloc(&R_ux,    N * sizeof(float));
    cudaMalloc(&R_uy,    N * sizeof(float));
    cudaMalloc(&R_uz,    N * sizeof(float));
    cudaMalloc(&g_sigma, N * sizeof(float));
    cudaMalloc(&g_ux,    N * sizeof(float));
    cudaMalloc(&g_uy,    N * sizeof(float));
    cudaMalloc(&g_uz,    N * sizeof(float));

    cudaMemcpy(R_sigma, h_R_sigma, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(R_ux,    h_R_ux,    N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(R_uy,    h_R_uy,    N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(R_uz,    h_R_uz,    N * sizeof(float), cudaMemcpyHostToDevice);

    float scale_sigma = 2.f * w.w_sigma / float(N);
    float scale_u     = 2.f * w.w_u     / float(N);
    int tb = 256; int blocks = (Nint + tb - 1) / tb;
    k_backward_scales<<<blocks, tb>>>(Nint, scale_sigma, scale_u, R_sigma, R_ux, R_uy, R_uz, g_sigma, g_ux, g_uy, g_uz);

    cudaMemcpy(h_g_sigma, g_sigma, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_ux,    g_ux,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_uy,    g_uy,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_uz,    g_uz,    N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(R_sigma); cudaFree(R_ux); cudaFree(R_uy); cudaFree(R_uz);
    cudaFree(g_sigma); cudaFree(g_ux); cudaFree(g_uy); cudaFree(g_uz);
}

} // namespace phys
