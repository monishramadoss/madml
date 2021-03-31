#pragma once

#include "vknn.h"
#include "../engine/layer.h"



struct gemm_param
{
    uint32_t total;
    uint32_t batchsize;
    
    uint32_t use_bias;
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

class gemm : public layer
{
    gemm_param m_param;

public:
    explicit gemm(float alpha, float beta, bool use_bias);
    void forward(tensor& y,tensor& x, tensor& w, tensor& b);
};

extern void test_gemm();