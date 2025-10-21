#include "backend.h"
#include "mlp.h"
#include <algorithm>
#include <vector>


static inline float relu(float x) {
    return x > 0.f ? x : 0.f;
}
static inline float relu_g(float x) {
    return x > 0.f ? 1.f : 0.f;
}

template <>
void mlp_forward<ExecCpu>(const float* x, const float* W1, const float* b1, const float* W2, const float* b2, float* y, std::size_t B, std::size_t In, std::size_t H, std::size_t Out) {
    std::vector<float> z1(B * H), a1(B * H);
    for (std::size_t i = 0; i < B; ++i) {
        for (std::size_t h = 0; h < H; ++h) {
            float s         = b1[h];
            const float* wi = &W1[h * In];
            const float* xi = &x[i * In];
            for (std::size_t k = 0; k < In; ++k) s += wi[k] * xi[k];
            z1[i * H + h] = s;
            a1[i * H + h] = relu(s);
        }
    }
    for (std::size_t i = 0; i < B; ++i) {
        for (std::size_t o = 0; o < Out; ++o) {
            float s         = b2[o];
            const float* w2 = &W2[o * H];
            const float* a  = &a1[i * H];
            for (std::size_t h = 0; h < H; ++h) s += w2[h] * a[h];
            y[i * Out + o] = s;
        }
    }
}

template <>
void mlp_backward<ExecCpu>(const float* x, const float* y_target, const float* W1, const float* b1, const float* W2, const float* b2, float* dW1, float* db1, float* dW2, float* db2, std::size_t B, std::size_t In, std::size_t H, std::size_t Out) {
    std::vector<float> z1(B * H), a1(B * H), y(B * Out);
    for (std::size_t i = 0; i < B; ++i) {
        for (std::size_t h = 0; h < H; ++h) {
            float s = b1[h];
            for (std::size_t k = 0; k < In; ++k) s += W1[h * In + k] * x[i * In + k];
            z1[i * H + h] = s;
            a1[i * H + h] = relu(s);
        }
    }
    for (std::size_t i = 0; i < B; ++i) {
        for (std::size_t o = 0; o < Out; ++o) {
            float s = b2[o];
            for (std::size_t h = 0; h < H; ++h) s += W2[o * H + h] * a1[i * H + h];
            y[i * Out + o] = s;
        }
    }
    std::vector<float> gz2(B * Out), gz1(B * H);
    for (std::size_t i = 0; i < B; ++i)
        for (std::size_t o = 0; o < Out; ++o) gz2[i * Out + o] = (2.f / float(B * Out)) * (y[i * Out + o] - y_target[i * Out + o]);

    std::fill(dW2, dW2 + Out * H, 0.f);
    std::fill(db2, db2 + Out, 0.f);
    std::fill(dW1, dW1 + H * In, 0.f);
    std::fill(db1, db1 + H, 0.f);

    for (std::size_t o = 0; o < Out; ++o)
        for (std::size_t h = 0; h < H; ++h)
            for (std::size_t i = 0; i < B; ++i) dW2[o * H + h] += gz2[i * Out + o] * a1[i * H + h];

    for (std::size_t o = 0; o < Out; ++o)
        for (std::size_t i = 0; i < B; ++i) db2[o] += gz2[i * Out + o];

    for (std::size_t i = 0; i < B; ++i)
        for (std::size_t h = 0; h < H; ++h) {
            float s = 0.f;
            for (std::size_t o = 0; o < Out; ++o) s += gz2[i * Out + o] * W2[o * H + h];
            gz1[i * H + h] = s * relu_g(z1[i * H + h]);
        }

    for (std::size_t h = 0; h < H; ++h)
        for (std::size_t k = 0; k < In; ++k)
            for (std::size_t i = 0; i < B; ++i) dW1[h * In + k] += gz1[i * H + h] * x[i * In + k];

    for (std::size_t h = 0; h < H; ++h)
        for (std::size_t i = 0; i < B; ++i) db1[h] += gz1[i * H + h];
}
