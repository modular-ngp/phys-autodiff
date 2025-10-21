#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

static float relu(float x) {
    return x > 0.f ? x : 0.f;
}
static float relu_g(float x) {
    return x > 0.f ? 1.f : 0.f;
}

float forward_loss(const std::vector<float>& x, const std::vector<float>& W1, const std::vector<float>& b1, const std::vector<float>& W2, const std::vector<float>& b2, const std::vector<float>& y_target, size_t B, size_t In, size_t H, size_t Out) {
    std::vector<float> z1(B * H), a1(B * H), y(B * Out);
    for (size_t i = 0; i < B; ++i) {
        for (size_t h = 0; h < H; ++h) {
            float s = b1[h];
            for (size_t k = 0; k < In; ++k) s += W1[h * In + k] * x[i * In + k];
            z1[i * H + h] = s;
            a1[i * H + h] = relu(s);
        }
    }
    for (size_t i = 0; i < B; ++i) {
        for (size_t o = 0; o < Out; ++o) {
            float s = b2[o];
            for (size_t h = 0; h < H; ++h) s += W2[o * H + h] * a1[i * H + h];
            y[i * Out + o] = s;
        }
    }
    float loss = 0.f;
    for (size_t i = 0; i < B * Out; ++i) {
        float d = y[i] - y_target[i];
        loss += d * d;
    }
    return loss / (float) (B * Out);
}

void backward_grad(const std::vector<float>& x, const std::vector<float>& W1, const std::vector<float>& b1, const std::vector<float>& W2, const std::vector<float>& b2, const std::vector<float>& y_target, size_t B, size_t In, size_t H, size_t Out, std::vector<float>& dW1) {
    std::vector<float> z1(B * H), a1(B * H), y(B * Out);
    for (size_t i = 0; i < B; ++i) {
        for (size_t h = 0; h < H; ++h) {
            float s = b1[h];
            for (size_t k = 0; k < In; ++k) s += W1[h * In + k] * x[i * In + k];
            z1[i * H + h] = s;
            a1[i * H + h] = relu(s);
        }
    }
    for (size_t i = 0; i < B; ++i) {
        for (size_t o = 0; o < Out; ++o) {
            float s = b2[o];
            for (size_t h = 0; h < H; ++h) s += W2[o * H + h] * a1[i * H + h];
            y[i * Out + o] = s;
        }
    }

    std::vector<float> gz2(B * Out), gz1(B * H);
    for (size_t i = 0; i < B; ++i) {
        for (size_t o = 0; o < Out; ++o) gz2[i * Out + o] = (2.f / (float) (B * Out)) * (y[i * Out + o] - y_target[i * Out + o]);
    }
    for (size_t i = 0; i < B; ++i) {
        for (size_t h = 0; h < H; ++h) {
            float s = 0.f;
            for (size_t o = 0; o < Out; ++o) s += W2[o * H + h] * gz2[i * Out + o];
            gz1[i * H + h] = s * relu_g(z1[i * H + h]);
        }
    }
    std::fill(dW1.begin(), dW1.end(), 0.f);
    for (size_t h = 0; h < H; ++h) {
        for (size_t k = 0; k < In; ++k) {
            float s = 0.f;
            for (size_t i = 0; i < B; ++i) s += gz1[i * H + h] * x[i * In + k];
            dW1[h * In + k] = s;
        }
    }
}

int main() {
    size_t B = 2, In = 4, H = 8, Out = 3;
    std::vector<float> x(B * In), W1(H * In), b1(H), W2(Out * H), b2(Out), y_target(B * Out);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    for (auto& v : x) v = U(rng);
    for (auto& v : W1) v = U(rng);
    for (auto& v : b1) v = U(rng);
    for (auto& v : W2) v = U(rng);
    for (auto& v : b2) v = U(rng);
    for (auto& v : y_target) v = U(rng);

    std::vector<float> dW1(H * In);
    backward_grad(x, W1, b1, W2, b2, y_target, B, In, H, Out, dW1);

    size_t idx   = 5;
    float eps    = 1e-3f;
    float w_orig = W1[idx];
    W1[idx]      = w_orig + eps;
    float Lp     = forward_loss(x, W1, b1, W2, b2, y_target, B, In, H, Out);
    W1[idx]      = w_orig - eps;
    float Lm     = forward_loss(x, W1, b1, W2, b2, y_target, B, In, H, Out);
    W1[idx]      = w_orig;

    float fd = (Lp - Lm) / (2.f * eps);
    std::cout << "Analytic dW1[" << idx << "] = " << dW1[idx] << "\n";
    std::cout << "FiniteDiff = " << fd << "\n";
    std::cout << "Abs error = " << std::fabs(dW1[idx] - fd) << std::endl;
}
