#pragma once

#include "vknn.h"
#include "../engine/layer.h"

class optimizer : public layer
{
};
struct adam_params
{
    float lr;
    float beta_a;
    float beta_b;
    float eps;
    float weight_decay;
    bool amsgrad;
};

class adam : public layer
{
    adam_params m_param;
public:
    adam(std::vector<tensor>& params, float lr, float beta_a, float beta_b, float eps, float weight_decay, bool amsgrad);
    void forward(tensor& p, tensor& m, tensor& r, tensor& m_k_hat, tensor& r_k_hat);
    void forward();
};