#include "pooling.h"
#include "common.h"
#include "utils.h"

constexpr int local_sz_x_conv = 16;
constexpr int local_sz_y_conv = 64;

maxPooling::maxPooling(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
    bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size), m_stride(stride),
    m_padding(padding), m_dilation(dilation), USE_BIAS(use_bias)
{    
    trans = std::make_shared<transpose>(transpose(std::vector<int>{1, 0, 2, 3, 4}));
}

std::shared_ptr<tensor>& maxPooling::operator()(const std::shared_ptr<tensor>& x_)
{
    x = x_;

    auto input_shape = x->getShape();
    int channels = 1;
    int batch_size = input_shape[0] * input_shape[1];

    if (!kernel)
        kernel = std::make_shared<vol2col>(vol2col(channels, m_kernel_size, m_padding, m_stride, m_dilation));

    kernel->operator()(t1, x);

    // argmax t1 across n_output_plane
    auto out = kernel->output_shape();

    y->reShape(std::vector<int>{m_num_filters, batch_size, out[0], out[1], out[2]});
    t4 = trans->operator()(y);

    return t4;
}

int maxPooling::set_backward()
{
    return 1;
}

void maxPooling::update_weight()
{
}

maxUnPooling::maxUnPooling(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
    bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size), m_stride(stride),
    m_padding(padding), m_dilation(dilation), USE_BIAS(use_bias)
{
    
    trans = std::make_shared<transpose>(transpose(std::vector<int>{1, 0, 2, 3, 4}));
}

std::shared_ptr<tensor>& maxUnPooling::operator()(const std::shared_ptr<tensor>& x_)
{
    x = x_;

    auto input_shape = x->getShape();
    int channels = 1;
    int batch_size = input_shape[0] * input_shape[1];

    if (!kernel)
        kernel = std::make_shared<col2vol>(col2vol(channels, m_kernel_size, m_padding, m_stride, m_dilation));

    kernel->operator()(t1, x);

    // argmax t1 across n_output_plane
    auto out = kernel->output_shape();

    y->reShape(std::vector<int>{m_num_filters, batch_size, out[0], out[1], out[2]});
    t4 = trans->operator()(y);

    return t4;
}

int maxUnPooling::set_backward()
{
    return 1;
}

void maxUnPooling::update_weight()
{
}

avgPooling::avgPooling(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
    bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size), m_stride(stride),
    m_padding(padding), m_dilation(dilation), USE_BIAS(use_bias)
{
    
    trans = std::make_shared<transpose>(transpose(std::vector<int>{1, 0, 2, 3, 4}));
}

std::shared_ptr<tensor>& avgPooling::operator()(const std::shared_ptr<tensor>& x_)
{
    x = x_;

    auto input_shape = x->getShape();
    int channels = 1;
    int batch_size = input_shape[0] * input_shape[1];

    if (!kernel)
        kernel = std::make_shared<vol2col>(vol2col(channels, m_kernel_size, m_padding, m_stride, m_dilation));

    kernel->operator()(t1, x);

    // mean t1 across n_output_plane
    auto out = kernel->output_shape();

    y->reShape(std::vector<int>{m_num_filters, batch_size, out[0], out[1], out[2]});
    t4 = trans->operator()(y);

    return t4;
}

int avgPooling::set_backward()
{
    return 1;
}

void avgPooling::update_weight()
{
}