#include "mlp_grid.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace phys;

static double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { double d = double(a[i]) - double(b[i]); num += d*d; den += double(a[i])*double(a[i]); }
    return std::sqrt(num / (den + 1e-30));
}

int main() {
    GridSpec g; g.nx = 32; g.ny = 32; g.nz = 24; g.hx = 1.f; g.hy = 1.f; g.hz = 1.f; g.dt = 1e-2f; g.periodic = false;
    MLPGridConfig cfg; cfg.dims = {4, 64, 4}; cfg.norm = CoordNorm::MinusOneToOne;
    MLPWeights w; mlp_random_init(w, cfg.dims, 123u, 0.25f);
    const size_t N = (size_t) g.nx * g.ny * g.nz;
    std::vector<float> y_cpu, y_gpu;
    mlp_grid_infer_cpu(g, cfg, w, 0.3f, y_cpu);
    mlp_grid_infer_cuda(g, cfg, w, 0.3f, y_gpu);
    double r = rel_l2(y_cpu, y_gpu);
    std::cout << "rel_l2(cpu,gpu): " << r << "\n";
    if (r > 1e-6) { std::cerr << "[FAIL] MLP grid inference mismatch\n"; return 1; }
    std::cout << "[PASS] test_mlp_grid_infer\n"; return 0;
}

