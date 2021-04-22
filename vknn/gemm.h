#pragma once

#include "vknn.h"
#include "../engine/layer.h"

struct gemm_param
{
    uint32_t total;
    uint32_t batchsize;
    float alpha;
    float beta;
    uint32_t use_bias;
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

class gemm : public layer
{
    gemm_param m_param;
    bool m_transpose_x;
    bool m_transpose_w;

public:
    explicit gemm(float alpha, float beta, bool use_bias, bool transpose_x = false, bool transpose_w = false);
    void forward(tensor& y, tensor& x, tensor& w, tensor& b);
};

extern void test_gemm();