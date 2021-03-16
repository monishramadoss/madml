#pragma once

#include "vknn.h"
#include "../kernel/layer.h"

struct gemm_param
{
    int total;
    int batchsize;
    float alpha;
    float beta;
    bool use_bias;
    int m;
    int n;
    int k;
};

class gemm : public layer
{
    gemm_param m_param;

public:
    explicit gemm(float alpha, float beta, bool use_bias);
    void forward(std::shared_ptr<tensor>& y, const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w, const std::shared_ptr<tensor>& b);
};