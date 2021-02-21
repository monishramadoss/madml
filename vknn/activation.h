#pragma once

#include "vknn.h"
#include "../backend/layer.h"

struct single_param{
    int total;
    float alpha;
};



class relu : public layer{
    single_param m_param;
    bool inplace;
public:
    explicit relu(bool in_place);
    void forward(std::shared_ptr<tensor>& y, const std::shared_ptr<tensor>& x);
};