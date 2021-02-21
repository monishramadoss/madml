#pragma once

#include "vknn.h"
#include "../kernel/layer.h"

struct vol2col_param {
    int total;
    int batchsize;
    int channels;
    int kernel_h;
    int kernel_w;
    int kernel_d;
    int pad_h;
    int pad_w;
    int pad_d;
    int stride_h;
    int stride_w;
    int stride_d;
    int dilation_h;
    int dilation_w;
    int dilation_d;
    int height_col; // height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
    int width_col; // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
    int depth_col; // depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1
    int height_vol;
    int width_vol;
    int depth_vol;
};


class vol2col: public layer{
    vol2col_param m_param;
public:
    explicit vol2col(std::vector<int>& params);
    void forward(std::shared_ptr<tensor>& col, const std::shared_ptr<tensor>& vol);
};

class col2vol: public layer{
    vol2col_param m_param;
public:
    explicit col2vol(std::vector<int>& params);
    void forward(std::shared_ptr<tensor>& vol, const std::shared_ptr<tensor>& col);
};