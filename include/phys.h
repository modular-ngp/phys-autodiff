#ifndef PHYS_AUTODIFF_PHYS_H
#define PHYS_AUTODIFF_PHYS_H

#include <cstddef>

namespace phys {

struct GridSpec {
    int nx{0}, ny{0}, nz{0};
    float hx{1.f}, hy{1.f}, hz{1.f};
    float dt{1.f};
    bool periodic{true};
};

struct PhysWeights {
    float w_sigma{1.f};
    float w_u{1.f};
};

// Layout for vector fields u: channel-major contiguous [ux(0..N-1), uy(0..N-1), uz(0..N-1)]
// Scalars are length N = nx*ny*nz arrays.

// Compute residuals R_sigma (scalar) and R_u (vector components) using central differences in space
// and central difference in time: df/dt ~ (f(t+dt) - f(t-dt)) / (2*dt).
// Inputs are fields at t-1, t, t+1 for sigma and u.
void cpu_phys_residuals(const GridSpec& g,
                        const float* sigma_tm1,
                        const float* sigma_t,
                        const float* sigma_tp1,
                        const float* u_tm1,  // 3*N layout
                        const float* u_t,    // 3*N layout
                        const float* u_tp1,  // 3*N layout
                        float* R_sigma,
                        float* R_ux,
                        float* R_uy,
                        float* R_uz);

// Compute weighted mean squared losses. Optionally returns residuals via pointers above.
void cpu_phys_loss_forward(const GridSpec& g,
                           const PhysWeights& w,
                           const float* sigma_tm1,
                           const float* sigma_t,
                           const float* sigma_tp1,
                           const float* u_tm1,
                           const float* u_t,
                           const float* u_tp1,
                           float* out_loss_sigma,
                           float* out_loss_u,
                           float* opt_R_sigma = nullptr,
                           float* opt_R_ux    = nullptr,
                           float* opt_R_uy    = nullptr,
                           float* opt_R_uz    = nullptr);

// Backward wrt residuals: g_sigma = 2*w_sigma/N * R_sigma; g_u = 2*w_u/N * R_u (per-component)
void cpu_phys_loss_backward(const GridSpec& g,
                            const PhysWeights& w,
                            const float* R_sigma,
                            const float* R_ux,
                            const float* R_uy,
                            const float* R_uz,
                            float* g_sigma,
                            float* g_ux,
                            float* g_uy,
                            float* g_uz);

} // namespace phys

#endif // PHYS_AUTODIFF_PHYS_H

