#include "backend.h"
#include "mlp.h"
#include <cuda_runtime.h>
#include <vector>

__device__ float relu(float x) {
    return x > 0.f ? x : 0.f;
}

__device__ float relu_g(float x) {
    return x > 0.f ? 1.f : 0.f;
}

__global__ void k_linear_relu(const float* x, const float* W, const float* b, float* z, float* a, std::size_t B, std::size_t In, std::size_t Out) {
    std::size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= B * Out) return;
    std::size_t i   = n / Out;
    std::size_t o   = n % Out;
    const float* wi = &W[o * In];
    const float* xi = &x[i * In];
    float s         = b[o];
    for (std::size_t k = 0; k < In; ++k) s += wi[k] * xi[k];
    z[n] = s;
    a[n] = relu(s);
}

__global__ void k_linear(const float* x, const float* W, const float* b, float* y, std::size_t B, std::size_t In, std::size_t Out) {
    std::size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= B * Out) return;
    std::size_t i   = n / Out;
    std::size_t o   = n % Out;
    const float* wi = &W[o * In];
    const float* xi = &x[i * In];
    float s         = b[o];
    for (std::size_t k = 0; k < In; ++k) s += wi[k] * xi[k];
    y[n] = s;
}

__global__ void k_gz2(const float* y, const float* y_target, float* gz2, std::size_t N, float norm) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    gz2[i] = norm * (y[i] - y_target[i]);
}

__global__ void k_dW2(const float* a1, const float* gz2, float* dW2, std::size_t B, std::size_t H, std::size_t Out) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Out * H) return;
    std::size_t o = idx / H;
    std::size_t h = idx % H;
    float s       = 0.f;
    for (std::size_t i = 0; i < B; ++i) s += gz2[i * Out + o] * a1[i * H + h];
    dW2[idx] = s;
}

__global__ void k_db2(const float* gz2, float* db2, std::size_t B, std::size_t Out) {
    std::size_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= Out) return;
    float s = 0.f;
    for (std::size_t i = 0; i < B; ++i) s += gz2[i * Out + o];
    db2[o] = s;
}

__global__ void k_gz1(const float* gz2, const float* W2, const float* z1, float* gz1, std::size_t B, std::size_t H, std::size_t Out) {
    std::size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= B * H) return;
    std::size_t i = n / H;
    std::size_t h = n % H;
    float s       = 0.f;
    for (std::size_t o = 0; o < Out; ++o) s += W2[o * H + h] * gz2[i * Out + o];
    gz1[n] = s * relu_g(z1[n]);
}

__global__ void k_dW1(const float* x, const float* gz1, float* dW1, std::size_t B, std::size_t In, std::size_t H) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * In) return;
    std::size_t h = idx / In;
    std::size_t k = idx % In;
    float s       = 0.f;
    for (std::size_t i = 0; i < B; ++i) s += gz1[i * H + h] * x[i * In + k];
    dW1[idx] = s;
}

__global__ void k_db1(const float* gz1, float* db1, std::size_t B, std::size_t H) {
    std::size_t h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    float s = 0.f;
    for (std::size_t i = 0; i < B; ++i) s += gz1[i * H + h];
    db1[h] = s;
}

template <>
void mlp_forward<ExecCuda>(const float* hx, const float* hW1, const float* hb1, const float* hW2, const float* hb2, float* hy, std::size_t B, std::size_t In, std::size_t H, std::size_t Out) {
    float *dx, *dW1, *db1, *dW2, *db2, *dz1, *da1, *dy;
    cudaMalloc(&dx, B * In * sizeof(float));
    cudaMalloc(&dW1, H * In * sizeof(float));
    cudaMalloc(&db1, H * sizeof(float));
    cudaMalloc(&dW2, Out * H * sizeof(float));
    cudaMalloc(&db2, Out * sizeof(float));
    cudaMalloc(&dz1, B * H * sizeof(float));
    cudaMalloc(&da1, B * H * sizeof(float));
    cudaMalloc(&dy, B * Out * sizeof(float));
    cudaMemcpy(dx, hx, B * In * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW1, hW1, H * In * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db1, hb1, H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW2, hW2, Out * H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db2, hb2, Out * sizeof(float), cudaMemcpyHostToDevice);
    int tb = 256;
    int g1 = (int) ((B * H + tb - 1) / tb);
    int g2 = (int) ((B * Out + tb - 1) / tb);
    k_linear_relu<<<g1, tb>>>(dx, dW1, db1, dz1, da1, B, In, H);
    k_linear<<<g2, tb>>>(da1, dW2, db2, dy, B, H, Out);
    cudaMemcpy(hy, dy, B * Out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dW1);
    cudaFree(db1);
    cudaFree(dW2);
    cudaFree(db2);
    cudaFree(dz1);
    cudaFree(da1);
    cudaFree(dy);
}

template <>
void mlp_backward<ExecCuda>(const float* hx, const float* hy_t, const float* hW1, const float* hb1, const float* hW2, const float* hb2, float* h_dW1, float* h_db1, float* h_dW2, float* h_db2, std::size_t B, std::size_t In, std::size_t H, std::size_t Out) {
    float *dx, *dW1, *db1, *dW2, *db2, *dz1, *a1, *y, *y_t, *gz2, *gz1, *W1, *b1, *W2, *b2;
    cudaMalloc(&dx, B * In * sizeof(float));
    cudaMalloc(&W1, H * In * sizeof(float));
    cudaMalloc(&b1, H * sizeof(float));
    cudaMalloc(&W2, Out * H * sizeof(float));
    cudaMalloc(&b2, Out * sizeof(float));
    cudaMalloc(&dz1, B * H * sizeof(float));
    cudaMalloc(&a1, B * H * sizeof(float));
    cudaMalloc(&y, B * Out * sizeof(float));
    cudaMalloc(&y_t, B * Out * sizeof(float));
    cudaMalloc(&gz2, B * Out * sizeof(float));
    cudaMalloc(&gz1, B * H * sizeof(float));
    cudaMalloc(&dW1, H * In * sizeof(float));
    cudaMalloc(&db1, H * sizeof(float));
    cudaMalloc(&dW2, Out * H * sizeof(float));
    cudaMalloc(&db2, Out * sizeof(float));

    cudaMemcpy(dx, hx, B * In * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W1, hW1, H * In * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1, hb1, H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W2, hW2, Out * H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b2, hb2, Out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_t, hy_t, B * Out * sizeof(float), cudaMemcpyHostToDevice);

    int tb = 256;
    int g1 = (int) ((B * H + tb - 1) / tb);
    int g2 = (int) ((B * Out + tb - 1) / tb);

    k_linear_relu<<<g1, tb>>>(dx, W1, b1, dz1, a1, B, In, H);
    k_linear<<<g2, tb>>>(a1, W2, b2, y, B, H, Out);

    float norm = 2.f / float(B * Out);
    k_gz2<<<g2, tb>>>(y, y_t, gz2, B * Out, norm);
    k_dW2<<<(int) ((Out * H + tb - 1) / tb), tb>>>(a1, gz2, dW2, B, H, Out);
    k_db2<<<(int) ((Out + tb - 1) / tb), tb>>>(gz2, db2, B, Out);
    k_gz1<<<g1, tb>>>(gz2, W2, dz1, gz1, B, H, Out);
    k_dW1<<<(int) ((H * In + tb - 1) / tb), tb>>>(dx, gz1, dW1, B, In, H);
    k_db1<<<(int) ((H + tb - 1) / tb), tb>>>(gz1, db1, B, H);

    cudaMemcpy(h_dW1, dW1, H * In * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_db1, db1, H * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dW2, dW2, Out * H * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_db2, db2, Out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dx);
    cudaFree(W1);
    cudaFree(b1);
    cudaFree(W2);
    cudaFree(b2);
    cudaFree(dz1);
    cudaFree(a1);
    cudaFree(y);
    cudaFree(y_t);
    cudaFree(gz2);
    cudaFree(gz1);
    cudaFree(dW1);
    cudaFree(db1);
    cudaFree(dW2);
    cudaFree(db2);
}
