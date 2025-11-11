#include "mlp_grid.h"
#include "phys.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace phys;

static bool has_nan(const std::vector<float>& v) {
    for (float x : v) if (std::isnan(x) || std::isinf(x)) return true;
    return false;
}

int main() {
    GridSpec g; g.nx = 48; g.ny = 48; g.nz = 32; g.hx = 1.f; g.hy = 1.f; g.hz = 1.f; g.dt = 2e-3f; g.periodic = true;
    MLPGridConfig cfg; cfg.dims = {4, 64, 4}; cfg.norm = CoordNorm::MinusOneToOne;
    MLPWeights w; mlp_random_init(w, cfg.dims, 321u, 0.25f);
    std::vector<float> s_tm1, s_t, s_tp1, u_tm1, u_t, u_tp1;
    mlp_generate_fields_cuda(g, cfg, w, 0.25f, g.dt, s_tm1, s_t, s_tp1, u_tm1, u_t, u_tp1);
    const size_t N = (size_t) g.nx * g.ny * g.nz;
    if (s_tm1.size() != N || s_t.size() != N || s_tp1.size() != N ||
        u_tm1.size() != 3*N || u_t.size() != 3*N || u_tp1.size() != 3*N) {
        std::cerr << "[FAIL] unexpected output sizes\n"; return 1;
    }
    if (has_nan(s_tm1) || has_nan(s_t) || has_nan(s_tp1) || has_nan(u_tm1) || has_nan(u_t) || has_nan(u_tp1)) {
        std::cerr << "[FAIL] NaN/Inf detected in generated fields\n"; return 2;
    }
    std::cout << "checksum "
              << s_tm1[0] + s_t[1] + s_tp1[2] + u_t[10] + u_t[N+20] + u_t[2*N+30]
              << "\n[PASS] test_mlp_phys_integration_inputs\n";
    return 0;
}

