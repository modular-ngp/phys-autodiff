#include "backend.h"
#include "mlp.h"
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::size_t B = 2, In = 4, H = 8, Out = 3;
    std::vector<float> x(B * In), W1(H * In), b1(H), W2(Out * H), b2(Out), y_target(B * Out);
    std::vector<float> dW1(H * In), db1(H), dW2(Out * H), db2(Out);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    for (auto& v : x) v = U(rng);
    for (auto& v : W1) v = U(rng);
    for (auto& v : b1) v = U(rng);
    for (auto& v : W2) v = U(rng);
    for (auto& v : b2) v = U(rng);
    for (auto& v : y_target) v = U(rng);

    mlp_backward<ExecCpu>(x.data(), y_target.data(), W1.data(), b1.data(), W2.data(), b2.data(), dW1.data(), db1.data(), dW2.data(), db2.data(), B, In, H, Out);

    std::cout << "dW1[0..4]: ";
    for (std::size_t i = 0; i < 5; ++i) std::cout << dW1[i] << " ";
    std::cout << "\ndb1[0..4]: ";
    for (std::size_t i = 0; i < 5; ++i) std::cout << db1[i] << " ";
    std::cout << std::endl;
}
