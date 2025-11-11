#include "phys.h"
#include <cuda_runtime.h>

namespace phys {

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ int wrapi(int v, int n) {
    int r = v % n;
    return r < 0 ? r + n : r;
}

__device__ __forceinline__ size_t off3(int x, int y, int z, int nx, int ny) {
    return (size_t)((z * ny + y) * nx + x);
}

__device__ __forceinline__ int idx_x(int i, int nx, int ny) {
    return i % nx;
}
__device__ __forceinline__ int idx_y(int i, int nx, int ny) {
    return (i / nx) % ny;
}
__device__ __forceinline__ int idx_z(int i, int nx, int ny) {
    return i / (nx * ny);
}

__device__ __forceinline__ int bound_or_wrap(int v, int n, bool periodic) {
    return periodic ? ((v % n) + n) % n : (v < 0 ? 0 : (v >= n ? n - 1 : v));
}

__global__ void k_residuals_fused(int N, int nx, int ny, int nz,
                                  float inv2dt, float inv2hx, float inv2hy, float inv2hz,
                                  bool periodic,
                                  const float* __restrict__ sigma_tm1,
                                  const float* __restrict__ sigma_t,
                                  const float* __restrict__ sigma_tp1,
                                  const float* __restrict__ u_tm1,
                                  const float* __restrict__ u_t,
                                  const float* __restrict__ u_tp1,
                                  float* __restrict__ R_sigma,
                                  float* __restrict__ R_ux,
                                  float* __restrict__ R_uy,
                                  float* __restrict__ R_uz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int x = idx_x(i, nx, ny);
    int y = idx_y(i, nx, ny);
    int z = idx_z(i, nx, ny);

    int xp = bound_or_wrap(x + 1, nx, periodic);
    int xm = bound_or_wrap(x - 1, nx, periodic);
    int yp = bound_or_wrap(y + 1, ny, periodic);
    int ym = bound_or_wrap(y - 1, ny, periodic);
    int zp = bound_or_wrap(z + 1, nz, periodic);
    int zm = bound_or_wrap(z - 1, nz, periodic);

    size_t Nsz = (size_t)N;
    size_t i_xp = off3(xp, y, z, nx, ny);
    size_t i_xm = off3(xm, y, z, nx, ny);
    size_t i_yp = off3(x, yp, z, nx, ny);
    size_t i_ym = off3(x, ym, z, nx, ny);
    size_t i_zp = off3(x, y, zp, nx, ny);
    size_t i_zm = off3(x, y, zm, nx, ny);

    float s_t   = sigma_t[i];
    float dt_sigma = (sigma_tp1[i] - sigma_tm1[i]) * inv2dt;
    float ds_dx = (sigma_t[i_xp] - sigma_t[i_xm]) * inv2hx;
    float ds_dy = (sigma_t[i_yp] - sigma_t[i_ym]) * inv2hy;
    float ds_dz = (sigma_t[i_zp] - sigma_t[i_zm]) * inv2hz;

    float ux_t = u_t[i];
    float uy_t = u_t[Nsz + i];
    float uz_t = u_t[2 * Nsz + i];
    float dux_dt = (u_tp1[i] - u_tm1[i]) * inv2dt;
    float duy_dt = (u_tp1[Nsz + i] - u_tm1[Nsz + i]) * inv2dt;
    float duz_dt = (u_tp1[2 * Nsz + i] - u_tm1[2 * Nsz + i]) * inv2dt;

    float dux_dx = (u_t[i_xp] - u_t[i_xm]) * inv2hx;
    float dux_dy = (u_t[i_yp] - u_t[i_ym]) * inv2hy;
    float dux_dz = (u_t[i_zp] - u_t[i_zm]) * inv2hz;
    float duy_dx = (u_t[Nsz + i_xp] - u_t[Nsz + i_xm]) * inv2hx;
    float duy_dy = (u_t[Nsz + i_yp] - u_t[Nsz + i_ym]) * inv2hy;
    float duy_dz = (u_t[Nsz + i_zp] - u_t[Nsz + i_zm]) * inv2hz;
    float duz_dx = (u_t[2 * Nsz + i_xp] - u_t[2 * Nsz + i_xm]) * inv2hx;
    float duz_dy = (u_t[2 * Nsz + i_yp] - u_t[2 * Nsz + i_ym]) * inv2hy;
    float duz_dz = (u_t[2 * Nsz + i_zp] - u_t[2 * Nsz + i_zm]) * inv2hz;

    float div_u = dux_dx + duy_dy + duz_dz;
    float adv_sigma = ux_t * ds_dx + uy_t * ds_dy + uz_t * ds_dz;
    float adv_ux = ux_t * dux_dx + uy_t * dux_dy + uz_t * dux_dz;
    float adv_uy = ux_t * duy_dx + uy_t * duy_dy + uz_t * duy_dz;
    float adv_uz = ux_t * duz_dx + uy_t * duz_dy + uz_t * duz_dz;

    R_sigma[i] = dt_sigma + adv_sigma + s_t * div_u;
    R_ux[i]    = dux_dt + adv_ux;
    R_uy[i]    = duy_dt + adv_uy;
    R_uz[i]    = duz_dt + adv_uz;
}

__global__ void k_backward_fused(int N, int nx, int ny, int nz,
                                 float inv2dt, float inv2hx, float inv2hy, float inv2hz,
                                 bool periodic, float scale_sigma, float scale_u,
                                 const float* __restrict__ sigma_tm1,
                                 const float* __restrict__ sigma_t,
                                 const float* __restrict__ sigma_tp1,
                                 const float* __restrict__ u_tm1,
                                 const float* __restrict__ u_t,
                                 const float* __restrict__ u_tp1,
                                 float* __restrict__ g_sigma,
                                 float* __restrict__ g_ux,
                                 float* __restrict__ g_uy,
                                 float* __restrict__ g_uz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int x = idx_x(i, nx, ny);
    int y = idx_y(i, nx, ny);
    int z = idx_z(i, nx, ny);
    int xp = bound_or_wrap(x + 1, nx, periodic);
    int xm = bound_or_wrap(x - 1, nx, periodic);
    int yp = bound_or_wrap(y + 1, ny, periodic);
    int ym = bound_or_wrap(y - 1, ny, periodic);
    int zp = bound_or_wrap(z + 1, nz, periodic);
    int zm = bound_or_wrap(z - 1, nz, periodic);
    size_t Nsz = (size_t)N;
    size_t i_xp = off3(xp, y, z, nx, ny);
    size_t i_xm = off3(xm, y, z, nx, ny);
    size_t i_yp = off3(x, yp, z, nx, ny);
    size_t i_ym = off3(x, ym, z, nx, ny);
    size_t i_zp = off3(x, y, zp, nx, ny);
    size_t i_zm = off3(x, y, zm, nx, ny);

    float s_t   = sigma_t[i];
    float dt_sigma = (sigma_tp1[i] - sigma_tm1[i]) * inv2dt;
    float ds_dx = (sigma_t[i_xp] - sigma_t[i_xm]) * inv2hx;
    float ds_dy = (sigma_t[i_yp] - sigma_t[i_ym]) * inv2hy;
    float ds_dz = (sigma_t[i_zp] - sigma_t[i_zm]) * inv2hz;

    float ux_t = u_t[i];
    float uy_t = u_t[Nsz + i];
    float uz_t = u_t[2 * Nsz + i];
    float dux_dt = (u_tp1[i] - u_tm1[i]) * inv2dt;
    float duy_dt = (u_tp1[Nsz + i] - u_tm1[Nsz + i]) * inv2dt;
    float duz_dt = (u_tp1[2 * Nsz + i] - u_tm1[2 * Nsz + i]) * inv2dt;

    float dux_dx = (u_t[i_xp] - u_t[i_xm]) * inv2hx;
    float dux_dy = (u_t[i_yp] - u_t[i_ym]) * inv2hy;
    float dux_dz = (u_t[i_zp] - u_t[i_zm]) * inv2hz;
    float duy_dx = (u_t[Nsz + i_xp] - u_t[Nsz + i_xm]) * inv2hx;
    float duy_dy = (u_t[Nsz + i_yp] - u_t[Nsz + i_ym]) * inv2hy;
    float duy_dz = (u_t[Nsz + i_zp] - u_t[Nsz + i_zm]) * inv2hz;
    float duz_dx = (u_t[2 * Nsz + i_xp] - u_t[2 * Nsz + i_xm]) * inv2hx;
    float duz_dy = (u_t[2 * Nsz + i_yp] - u_t[2 * Nsz + i_ym]) * inv2hy;
    float duz_dz = (u_t[2 * Nsz + i_zp] - u_t[2 * Nsz + i_zm]) * inv2hz;

    float div_u = dux_dx + duy_dy + duz_dz;
    float adv_sigma = ux_t * ds_dx + uy_t * ds_dy + uz_t * ds_dz;
    float adv_ux = ux_t * dux_dx + uy_t * dux_dy + uz_t * dux_dz;
    float adv_uy = ux_t * duy_dx + uy_t * duy_dy + uz_t * duy_dz;
    float adv_uz = ux_t * duz_dx + uy_t * duz_dy + uz_t * duz_dz;

    float R_sigma = dt_sigma + adv_sigma + s_t * div_u;
    float R_ux    = dux_dt + adv_ux;
    float R_uy    = duy_dt + adv_uy;
    float R_uz    = duz_dt + adv_uz;

    g_sigma[i] = scale_sigma * R_sigma;
    g_ux[i]    = scale_u * R_ux;
    g_uy[i]    = scale_u * R_uy;
    g_uz[i]    = scale_u * R_uz;
}

void cuda_phys_residuals_fused(const GridSpec& g,
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
    float *sigma_tm1, *sigma_t, *sigma_tp1;
    float *u_tm1, *u_t, *u_tp1;
    float *R_sigma, *R_ux, *R_uy, *R_uz;
    cudaMalloc(&sigma_tm1, N * sizeof(float));
    cudaMalloc(&sigma_t,   N * sizeof(float));
    cudaMalloc(&sigma_tp1, N * sizeof(float));
    cudaMalloc(&u_tm1, 3 * N * sizeof(float));
    cudaMalloc(&u_t,   3 * N * sizeof(float));
    cudaMalloc(&u_tp1, 3 * N * sizeof(float));
    cudaMalloc(&R_sigma, N * sizeof(float));
    cudaMalloc(&R_ux,    N * sizeof(float));
    cudaMalloc(&R_uy,    N * sizeof(float));
    cudaMalloc(&R_uz,    N * sizeof(float));
    cudaMemcpy(sigma_tm1, h_sigma_tm1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sigma_t,   h_sigma_t,   N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sigma_tp1, h_sigma_tp1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(u_tm1, h_u_tm1, 3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(u_t,   h_u_t,   3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(u_tp1, h_u_tp1, 3 * N * sizeof(float), cudaMemcpyHostToDevice);

    float inv2dt = 1.f / (2.f * g.dt);
    float inv2hx = 1.f / (2.f * g.hx);
    float inv2hy = 1.f / (2.f * g.hy);
    float inv2hz = 1.f / (2.f * g.hz);
    int tb = 256; int blocks = (Nint + tb - 1) / tb;
    k_residuals_fused<<<blocks, tb>>>(Nint, nx, ny, nz, inv2dt, inv2hx, inv2hy, inv2hz, g.periodic,
                                      sigma_tm1, sigma_t, sigma_tp1, u_tm1, u_t, u_tp1,
                                      R_sigma, R_ux, R_uy, R_uz);
    cudaMemcpy(h_R_sigma, R_sigma, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R_ux,    R_ux,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R_uy,    R_uy,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R_uz,    R_uz,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(sigma_tm1); cudaFree(sigma_t); cudaFree(sigma_tp1);
    cudaFree(u_tm1); cudaFree(u_t); cudaFree(u_tp1);
    cudaFree(R_sigma); cudaFree(R_ux); cudaFree(R_uy); cudaFree(R_uz);
}

void cuda_phys_loss_backward_fused(const GridSpec& g,
                                   const PhysWeights& w,
                                   const float* h_sigma_tm1,
                                   const float* h_sigma_t,
                                   const float* h_sigma_tp1,
                                   const float* h_u_tm1,
                                   const float* h_u_t,
                                   const float* h_u_tp1,
                                   float* h_g_sigma,
                                   float* h_g_ux,
                                   float* h_g_uy,
                                   float* h_g_uz) {
    int nx = g.nx, ny = g.ny, nz = g.nz;
    size_t N = (size_t)nx * ny * nz;
    int Nint = (int)N;
    float *sigma_tm1, *sigma_t, *sigma_tp1;
    float *u_tm1, *u_t, *u_tp1;
    float *g_sigma, *g_ux, *g_uy, *g_uz;
    cudaMalloc(&sigma_tm1, N * sizeof(float));
    cudaMalloc(&sigma_t,   N * sizeof(float));
    cudaMalloc(&sigma_tp1, N * sizeof(float));
    cudaMalloc(&u_tm1, 3 * N * sizeof(float));
    cudaMalloc(&u_t,   3 * N * sizeof(float));
    cudaMalloc(&u_tp1, 3 * N * sizeof(float));
    cudaMalloc(&g_sigma, N * sizeof(float));
    cudaMalloc(&g_ux,    N * sizeof(float));
    cudaMalloc(&g_uy,    N * sizeof(float));
    cudaMalloc(&g_uz,    N * sizeof(float));
    cudaMemcpy(sigma_tm1, h_sigma_tm1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sigma_t,   h_sigma_t,   N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sigma_tp1, h_sigma_tp1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(u_tm1, h_u_tm1, 3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(u_t,   h_u_t,   3 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(u_tp1, h_u_tp1, 3 * N * sizeof(float), cudaMemcpyHostToDevice);

    float inv2dt = 1.f / (2.f * g.dt);
    float inv2hx = 1.f / (2.f * g.hx);
    float inv2hy = 1.f / (2.f * g.hy);
    float inv2hz = 1.f / (2.f * g.hz);
    float scale_sigma = 2.f * w.w_sigma / float(N);
    float scale_u     = 2.f * w.w_u     / float(N);
    int tb = 256; int blocks = (Nint + tb - 1) / tb;
    k_backward_fused<<<blocks, tb>>>(Nint, nx, ny, nz, inv2dt, inv2hx, inv2hy, inv2hz, g.periodic, scale_sigma, scale_u,
                                     sigma_tm1, sigma_t, sigma_tp1, u_tm1, u_t, u_tp1,
                                     g_sigma, g_ux, g_uy, g_uz);

    cudaMemcpy(h_g_sigma, g_sigma, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_ux,    g_ux,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_uy,    g_uy,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_uz,    g_uz,    N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(sigma_tm1); cudaFree(sigma_t); cudaFree(sigma_tp1);
    cudaFree(u_tm1); cudaFree(u_t); cudaFree(u_tp1);
    cudaFree(g_sigma); cudaFree(g_ux); cudaFree(g_uy); cudaFree(g_uz);
}

} // namespace phys

