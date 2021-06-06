#pragma once

#include "vknn.h"
#include "../engine/layer.h"

struct mse_param
{
    uint32_t total;
    bool reduction;
};

class mse : public layer
{
    mse_param m_param;
    bool m_derivative;
public:
    mse(bool reduction);
    void forward(tensor& loss, tensor& l, tensor& t, tensor& dx);
};
