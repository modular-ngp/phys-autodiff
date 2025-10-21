#include "backend.h"
#include "mlp.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs((double) a[i] - b[i]));
    return m;
}

int main() {
    std::size_t B = 4, In = 8, H = 16, Out = 6;
    std::vector<float> x(B * In), W1(H * In), b1(H), W2(Out * H), b2(Out), y_t(B * Out);
    std::vector<float> dW1_cpu(H * In), db1_cpu(H), dW2_cpu(Out * H), db2_cpu(Out);
    std::vector<float> dW1_gpu(H * In), db1_gpu(H), dW2_gpu(Out * H), db2_gpu(Out);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    for (auto& v : x) v = U(rng);
    for (auto& v : W1) v = U(rng);
    for (auto& v : b1) v = U(rng);
    for (auto& v : W2) v = U(rng);
    for (auto& v : b2) v = U(rng);
    for (auto& v : y_t) v = U(rng);

    mlp_backward<ExecCpu>(x.data(), y_t.data(), W1.data(), b1.data(), W2.data(), b2.data(), dW1_cpu.data(), db1_cpu.data(), dW2_cpu.data(), db2_cpu.data(), B, In, H, Out);

    mlp_backward<ExecCuda>(x.data(), y_t.data(), W1.data(), b1.data(), W2.data(), b2.data(), dW1_gpu.data(), db1_gpu.data(), dW2_gpu.data(), db2_gpu.data(), B, In, H, Out);

    std::cout << "max_abs_diff dW1: " << max_abs_diff(dW1_cpu, dW1_gpu) << "\n";
    std::cout << "max_abs_diff db1: " << max_abs_diff(db1_cpu, db1_gpu) << "\n";
    std::cout << "max_abs_diff dW2: " << max_abs_diff(dW2_cpu, dW2_gpu) << "\n";
    std::cout << "max_abs_diff db2: " << max_abs_diff(db2_cpu, db2_gpu) << "\n";
    std::cout << "dW1_cpu[0..4]: ";
    for (size_t i = 0; i < 5; ++i) std::cout << dW1_cpu[i] << " ";
    std::cout << "\ndW1_gpu[0..4]: ";
    for (size_t i = 0; i < 5; ++i) std::cout << dW1_gpu[i] << " ";
    std::cout << std::endl;
}
