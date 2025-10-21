#include "mlp.h"
#include "backend.h"
#include <vector>
#include <algorithm>

static inline float relu(float x){ return x>0.f?x:0.f; }

template<>
void mlp_forward<ExecCpu>(const float* x, const float* W1, const float* b1,
                          const float* W2, const float* b2,
                          float* y,
                          std::size_t B, std::size_t In, std::size_t H, std::size_t Out){
    std::vector<float> z1(B*H), a1(B*H);
    for(std::size_t i=0;i<B;++i){
        for(std::size_t h=0;h<H;++h){
            float s=b1[h];
            const float* wi=&W1[h*In];
            const float* xi=&x[i*In];
            for(std::size_t k=0;k<In;++k) s+=wi[k]*xi[k];
            z1[i*H+h]=s;
            a1[i*H+h]=relu(s);
        }
    }
    for(std::size_t i=0;i<B;++i){
        for(std::size_t o=0;o<Out;++o){
            float s=b2[o];
            const float* w2=&W2[o*H];
            const float* a=&a1[i*H];
            for(std::size_t h=0;h<H;++h) s+=w2[h]*a[h];
            y[i*Out+o]=s;
        }
    }
}