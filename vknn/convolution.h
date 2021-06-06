#pragma once

#include "vknn.h"
#include "../engine/layer.h"

struct vol2col_param
{
    uint32_t total;
    uint32_t batchsize;
    uint32_t channels;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t kernel_d;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t pad_d;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t stride_d;
    uint32_t dilation_h;
    uint32_t dilation_w;
    uint32_t dilation_d;
    uint32_t height_col; // height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
    uint32_t width_col; // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
    uint32_t depth_col; // depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1
    uint32_t height_vol;
    uint32_t width_vol;
    uint32_t depth_vol;
};

class vol2col : public layer
{
    vol2col_param m_param;
public:
    explicit vol2col(std::vector<int>& params);
    void forward(tensor& col, tensor& vol);
};

class col2vol : public layer
{
    vol2col_param m_param;
public:
    explicit col2vol(std::vector<int>& params);
    void forward(tensor& vol, tensor& col);
};

void cpu_vol2col(py::array_t<float, py::array::c_style | py::array::forcecast> vol,
    py::array_t<float, py::array::c_style | py::array::forcecast> col,
    int n_output_plane, int index_length, std::vector<int>& params);
void cpu_col2vol(py::array_t<float, py::array::c_style | py::array::forcecast> vol,
    py::array_t<float, py::array::c_style | py::array::forcecast> col,
    int n_output_plane, int index_length, std::vector<int>& params);