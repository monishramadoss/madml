#pragma once

#include "vknn.h"
#include "../engine/layer.h"

struct single_param
{
    int total;
    float alpha;
};

class relu : public layer
{
    single_param m_param;
    bool m_inplace;
    bool m_derivative;
public:
    explicit relu(bool in_place, bool derivative);
    void forward(tensor& y, tensor& x, tensor& w);
};