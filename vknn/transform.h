#pragma once

#include "vknn.h"
#include "../engine/layer.h"

struct transpose_param
{
    int total;
    int num_axes;
};

class transpose : public layer
{
    transpose_param m_param;
    std::vector<int> m_stride;
    tensor m_stride_tensor;
public:
    explicit transpose(std::vector<int>& order);
    void forward(tensor& y, tensor& x);
};

