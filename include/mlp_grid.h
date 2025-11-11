#ifndef PHYS_AUTODIFF_MLP_GRID_H
#define PHYS_AUTODIFF_MLP_GRID_H

#include "backend.h"
#include "mlp.h"
#include "phys.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace phys {

struct MLPDims {
    std::size_t In{4};
    std::size_t H{64};
    std::size_t Out{4}; // [sigma, ux, uy, uz]
};

struct MLPWeights {
    std::vector<float> W1; // H x In
    std::vector<float> b1; // H
    std::vector<float> W2; // Out x H
    std::vector<float> b2; // Out
};

enum class CoordNorm { ZeroToOne, MinusOneToOne };

struct MLPGridConfig {
    MLPDims dims{};
    CoordNorm norm{CoordNorm::MinusOneToOne};
};

// Initialize weights with uniform random in [-scale, scale]
void mlp_random_init(MLPWeights& w, const MLPDims& d, std::uint32_t seed = 42, float scale = 0.5f);

// Build grid coordinates [x,y,z,t] for all grid points; coords has size N*4
void make_grid_coords(const GridSpec& g, float t, CoordNorm norm, std::vector<float>& coords);

// Inference wrappers over existing MLP forward implementations
void mlp_infer_cpu(const MLPDims& d, const MLPWeights& w, const float* coords, std::size_t N, float* out);
void mlp_infer_cuda(const MLPDims& d, const MLPWeights& w, const float* coords, std::size_t N, float* out);

// Convenience: evaluate MLP over a grid (x,y,z at time t); output is N*Out
void mlp_grid_infer_cpu(const GridSpec& g, const MLPGridConfig& cfg, const MLPWeights& w, float t, std::vector<float>& out);
void mlp_grid_infer_cuda(const GridSpec& g, const MLPGridConfig& cfg, const MLPWeights& w, float t, std::vector<float>& out);

// Generate physics input fields using MLP at t-dt, t, t+dt
// Outputs:
//  sigma_tm1, sigma_t, sigma_tp1: N
//  u_tm1, u_t, u_tp1: 3*N, channel-major per time slice
void mlp_generate_fields_cpu(const GridSpec& g, const MLPGridConfig& cfg, const MLPWeights& w, float t, float dt,
                             std::vector<float>& sigma_tm1, std::vector<float>& sigma_t, std::vector<float>& sigma_tp1,
                             std::vector<float>& u_tm1, std::vector<float>& u_t, std::vector<float>& u_tp1);

void mlp_generate_fields_cuda(const GridSpec& g, const MLPGridConfig& cfg, const MLPWeights& w, float t, float dt,
                              std::vector<float>& sigma_tm1, std::vector<float>& sigma_t, std::vector<float>& sigma_tp1,
                              std::vector<float>& u_tm1, std::vector<float>& u_t, std::vector<float>& u_tp1);

} // namespace phys

#endif // PHYS_AUTODIFF_MLP_GRID_H

