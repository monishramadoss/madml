#pragma once

#include "vknn.h"
#include "../kernel/layer.h"

struct gemm_param
{
    int total;
    int batchsize;
    float alpha;
    float beta;
    int m;
    int n;
    int k;
};

class gemm : public layer
{
    gemm_param m_param;

public:
    explicit gemm(float alpha, float beta);
    void forward(std::shared_ptr<tensor>& y, const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
};