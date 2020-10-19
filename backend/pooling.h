#ifndef POOLING_H
#define POOLING_H

#include <vector>
#include <utility>
#include "backend.h"
#include "layer.h"

class maxPooling : Module
{
public:
    class maxPooling(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation,
        int padding_type,
        bool use_bias
    );
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    int set_backward() override;
    void update_weight() override;

private:
    int m_num_filters;
    dhw m_kernel_size, m_stride, m_padding, m_dilation;
    bool USE_BIAS;

    //arg max
    std::shared_ptr<vol2col> kernel;
    std::shared_ptr<transpose> trans;
};

class maxUnPooling : Module
{
public:
    class maxUnPooling(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation,
        int padding_type,
        bool use_bias
    );
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    int set_backward() override;
    void update_weight() override;

private:
    int m_num_filters;
    dhw m_kernel_size, m_stride, m_padding, m_dilation;
    bool USE_BIAS;

    // arg max
    std::shared_ptr<col2vol> kernel;
    std::shared_ptr<transpose> trans;
};

class avgPooling : Module
{
public:
    class avgPooling( int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation,
        int padding_type,
        bool use_bias
    );
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    int set_backward() override;
    void update_weight() override;

private:
    int m_num_filters;
    dhw m_kernel_size, m_stride, m_padding, m_dilation;
    bool USE_BIAS;

    // avg
    std::shared_ptr<vol2col> kernel;
    std::shared_ptr<transpose> trans;
};

#endif
