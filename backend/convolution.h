#ifndef CONV_H
#define CONV_H

#include <vector>
#include <utility>
#include "backend.h"
#include "layer.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

struct dhw
{
    float d;
    float h;
    float w;
};

struct vol2col_param
{
    int total;
    int batchsize;
    int channels;
    float kernel_h;
    float kernel_w;
    float kernel_d;
    float pad_h;
    float pad_w;
    float pad_d;
    float stride_h;
    float stride_w;
    float stride_d;
    float dilation_h;
    float dilation_w;
    float dilation_d;
    float height_col; // height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
    float width_col; // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
    float depth_col; // depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1
    float height_vol;
    float width_vol;
    float depth_vol;
};

class vol2col : public Base_Layer<vol2col_param>
{
private:
    void computeGroupCount() override;
public:
    vol2col(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation);
    vol2col(int in_channel, std::vector<float>& params);
    void operator()(std::shared_ptr<tensor>& y, const std::shared_ptr<tensor>& x);
    std::vector<int> output_shape() const;
};

void init_vol2col(py::module& m);


class col2vol : public Base_Layer<vol2col_param>
{
private:
    void computeGroupCount() override;
public:
    col2vol(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation);
    col2vol(int in_channel, std::vector<float>& params);
    void operator()(std::shared_ptr<tensor>& y, const std::shared_ptr<tensor>& x);
    std::vector<int> output_shape() const;
};

void init_col2vol(py::module& m);


namespace nn
{
    class conv : public Module
    {
    public:
        conv(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type, bool use_bias);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
        int set_backward() override;
        void update_weight() override;

    private:
        int m_num_filters;
        dhw m_kernel_size, m_stride, m_padding, m_dilation;
        bool USE_BIAS;

        std::shared_ptr<gemm> mm;
        std::shared_ptr<vol2col> kernel;
        std::shared_ptr<math::add> bias;
        std::shared_ptr<transpose> trans;
    };

    class convTranspose : public Module
    {
    public:
        convTranspose(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
            bool use_bias);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
        int set_backward() override;
        void update_weight() override;

    private:
        int m_num_filters;
        dhw m_kernel_size, m_stride, m_padding, m_dilation;
        bool USE_BIAS;

        std::shared_ptr<gemm> mm;
        std::shared_ptr<vol2col> kernel;
        std::shared_ptr<math::add> bias;
        std::shared_ptr<transpose> trans;
    };
}

py::array_t<float> im2col_cpu(py::array_t<float> input1, py::array_t<float> result, std::vector<int>& params);
py::array_t<float> col2im_cpu(py::array_t<float> input1, py::array_t<float> result, std::vector<int>& params);
#endif
