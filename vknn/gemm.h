#pragma once

#include "vknn.h"
#include "../kernel/layer.h"

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

public:
    explicit gemm(float alpha, float beta, bool use_bias);
    void forward(std::shared_ptr<tensor>& y, const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w, const std::shared_ptr<tensor>& b);
};