#ifndef PHYS_AUTODIFF_MLP_H
#define PHYS_AUTODIFF_MLP_H
#include <cstddef>

template <typename Exec>
void mlp_forward(const float* x, const float* W1, const float* b1, const float* W2, const float* b2, float* y, std::size_t B, std::size_t In, std::size_t H, std::size_t Out);

template <typename Exec>
void mlp_backward(const float* x, const float* y_target, const float* W1, const float* b1, const float* W2, const float* b2, float* dW1, float* db1, float* dW2, float* db2, std::size_t B, std::size_t In, std::size_t H, std::size_t Out);
#endif // PHYS_AUTODIFF_MLP_H
