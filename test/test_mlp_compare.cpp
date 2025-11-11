#include "backend.h"
#include "mlp.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs((double) a[i] - b[i]));
    return m;
}

int main(int argc, char** argv) {
    std::size_t B = 512, In = 256, H = 512, Out = 256;
    int warmup = 2, iters = 10;
    if (argc >= 5) {
        B = std::stoul(argv[1]);
        In = std::stoul(argv[2]);
        H = std::stoul(argv[3]);
        Out = std::stoul(argv[4]);
    }
    if (argc >= 6) iters = std::stoi(argv[5]);

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

    for (int i = 0; i < warmup; ++i)
        mlp_backward<ExecCpu>(x.data(), y_t.data(), W1.data(), b1.data(), W2.data(), b2.data(), dW1_cpu.data(), db1_cpu.data(), dW2_cpu.data(), db2_cpu.data(), B, In, H, Out);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        mlp_backward<ExecCpu>(x.data(), y_t.data(), W1.data(), b1.data(), W2.data(), b2.data(), dW1_cpu.data(), db1_cpu.data(), dW2_cpu.data(), db2_cpu.data(), B, In, H, Out);
    auto t1 = std::chrono::steady_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / double(iters);

    for (int i = 0; i < warmup; ++i)
        mlp_backward<ExecCuda>(x.data(), y_t.data(), W1.data(), b1.data(), W2.data(), b2.data(), dW1_gpu.data(), db1_gpu.data(), dW2_gpu.data(), db2_gpu.data(), B, In, H, Out);

    auto g0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        mlp_backward<ExecCuda>(x.data(), y_t.data(), W1.data(), b1.data(), W2.data(), b2.data(), dW1_gpu.data(), db1_gpu.data(), dW2_gpu.data(), db2_gpu.data(), B, In, H, Out);
    auto g1 = std::chrono::steady_clock::now();
    double ms_gpu = std::chrono::duration<double, std::milli>(g1 - g0).count() / double(iters);

    std::cout << "Shapes B=" << B << " In=" << In << " H=" << H << " Out=" << Out << "\n";
    std::cout << "CPU backward avg: " << ms_cpu << " ms\n";
    std::cout << "GPU backward avg: " << ms_gpu << " ms\n";
    if (ms_gpu > 0.0) {
        double speedup = ms_cpu / ms_gpu;
        double pct = (ms_cpu - ms_gpu) / ms_cpu * 100.0;
        std::cout << "GPU speedup vs CPU: " << speedup << "x (" << pct << "%)\n";
    }

    std::cout << "max_abs_diff dW1: " << max_abs_diff(dW1_cpu, dW1_gpu) << "\n";
    std::cout << "max_abs_diff db1: " << max_abs_diff(db1_cpu, db1_gpu) << "\n";
    std::cout << "max_abs_diff dW2: " << max_abs_diff(dW2_cpu, dW2_gpu) << "\n";
    std::cout << "max_abs_diff db2: " << max_abs_diff(db2_cpu, db2_gpu) << "\n";
    return 0;
}

