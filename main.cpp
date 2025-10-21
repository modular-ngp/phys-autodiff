#include "backend.h"
#include "mlp.h"
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::size_t B = 2, In = 4, H = 8, Out = 3;
    std::vector<float> x(B * In), W1(H * In), b1(H), W2(Out * H), b2(Out), y(B * Out);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    for (auto& v : x) v = U(rng);
    for (auto& v : W1) v = U(rng);
    for (auto& v : b1) v = U(rng);
    for (auto& v : W2) v = U(rng);
    for (auto& v : b2) v = U(rng);
    mlp_forward<ExecCpu>(x.data(), W1.data(), b1.data(), W2.data(), b2.data(), y.data(), B, In, H, Out);
    std::cout << "y[0..4]: ";
    for (std::size_t i = 0; i < std::min<std::size_t>(5, y.size()); ++i) std::cout << y[i] << " ";
    std::cout << std::endl;
}
