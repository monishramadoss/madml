#include "common.h"
#include "utils.h"
#include "convolution.h"

constexpr int local_sz_x_conv = 16;
constexpr int local_sz_y_conv = 64;

vol2col::vol2col(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation) : Base_Layer<vol2col_param>(2)

{
    m_type = "vol2col";
    m_param = {
        0, 1, channels, kernel.h, kernel.w, kernel.d, pad.h, pad.w, pad.d, stride.h, stride.w, stride.d, dilation.h,
        dilation.w, dilation.d, 0, 0, 0, 0, 0, 0
    };
}

void vol2col::computeGroupCount()
{
    size_t tmp = m_param.channels;
    tmp *= static_cast<size_t>(m_param.kernel_h);
    tmp *= static_cast<size_t>(m_param.kernel_w);
    tmp *= static_cast<size_t>(m_param.kernel_d);

    m_group_x = static_cast<int>(alignSize(tmp, local_sz_x_conv)) / local_sz_x_conv;
    if (m_group_x > max_compute_work_group_count)
        m_group_x = max_compute_work_group_count;
    m_group_y = static_cast<int>(alignSize(m_param.batchsize, local_sz_y_conv)) / local_sz_y_conv;
    if (m_group_y > max_compute_work_group_count)
        m_group_y = max_compute_work_group_count;
    m_group_z = 1;
}

std::shared_ptr<tensor>& vol2col::operator()(const std::shared_ptr<tensor>& x_)
{
    if (m_pipeline == nullptr)
    {
        const float depth = static_cast<float>(x_->getShape()[x_->getShape().size() - 3]);
        const float height = static_cast<float>(x_->getShape()[x_->getShape().size() - 2]);
        const float width = static_cast<float>(x_->getShape()[x_->getShape().size() - 1]);
        m_param.batchsize = x_->getShape()[0];
        m_param.channels = x_->getShape()[1];
        m_param.depth_vol = depth;
        m_param.height_vol = height;
        m_param.width_vol = width;
        m_param.depth_col = (depth + 2 * m_param.pad_d - (m_param.dilation_d * (m_param.kernel_d - 1) - 1)) / m_param.stride_d +
            1;
        m_param.height_col = (height + 2 * m_param.pad_h - (m_param.dilation_h * (m_param.kernel_h - 1) - 1)) / m_param.stride_h
            + 1;
        m_param.width_col = (width + 2 * m_param.pad_w - (m_param.dilation_w * (m_param.kernel_w - 1) - 1)) / m_param.stride_w +
            1;
    }
    const int n_out_plane = static_cast<int>(m_param.channels * m_param.kernel_d * m_param.kernel_h * m_param.kernel_w);
    const int output_length = static_cast<int>(m_param.batchsize * m_param.depth_col * m_param.height_col * m_param.width_col);
    layer_construct_forward(kernel::shaders::vol2col_spv, sizeof(kernel::shaders::vol2col_spv), x_, Format::kFormatFp32,
                            std::vector<int>{n_out_plane, output_length});
    return y;
}

std::vector<int> vol2col::output_shape() const
{
    int d = static_cast<int>(m_param.depth_col);
    int h = static_cast<int>(m_param.height_col);
    int w = static_cast<int>(m_param.width_col);
    return std::vector<int>{d, h, w};
}

col2vol::col2vol(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation) : Base_Layer<vol2col_param>(2)
{
    m_type = "col2vol";
    m_param = {
        0, 1, channels, kernel.h, kernel.w, kernel.d, pad.h, pad.w, pad.d, stride.h, stride.w, stride.d, dilation.h,
        dilation.w, dilation.d, 0, 0, 0, 0, 0, 0
    };
}

void col2vol::computeGroupCount()
{
    size_t tmp = m_param.channels;
    tmp *= static_cast<size_t>(m_param.kernel_h);
    tmp *= static_cast<size_t>(m_param.kernel_w);
    tmp *= static_cast<size_t>(m_param.kernel_d);

    m_group_x = static_cast<int>(alignSize(tmp, local_sz_x_conv)) / local_sz_x_conv;
    if (m_group_x > max_compute_work_group_count)
        m_group_x = max_compute_work_group_count;
    m_group_y = static_cast<int>(alignSize(m_param.batchsize, local_sz_y_conv)) / local_sz_y_conv;
    if (m_group_y > max_compute_work_group_count)
        m_group_y = max_compute_work_group_count;
    m_group_z = 1;
}

std::shared_ptr<tensor>& col2vol::operator()(const std::shared_ptr<tensor>& x_)
{
    if (m_pipeline == nullptr)
    {
        const float depth = static_cast<float>(x_->getShape()[x_->getShape().size() - 3]);
        const float height = static_cast<float>(x_->getShape()[x_->getShape().size() - 2]);
        const float width = static_cast<float>(x_->getShape()[x_->getShape().size() - 1]);
        m_param.batchsize = x_->getShape()[0];

        m_param.depth_col = depth;
        m_param.height_col = height;
        m_param.width_col = width;
        m_param.depth_vol = (depth - 1) * m_param.stride_d - 2 * m_param.pad_d + m_param.dilation_d * (m_param.kernel_d - 1)
            + m_param.pad_d + 1;
        m_param.height_vol = (height - 1) * m_param.stride_h - 2 * m_param.pad_h + m_param.dilation_h * (m_param.kernel_h -
            1) + m_param.pad_h + 1;
        m_param.width_vol = (width - 1) * m_param.stride_w - 2 * m_param.pad_w + m_param.dilation_w * (m_param.kernel_w - 1)
            + m_param.pad_w + 1;
    }
    const int n_out_plane = static_cast<int>(m_param.channels * m_param.kernel_d * m_param.kernel_h * m_param.kernel_w);
    const int output_length = static_cast<int>(m_param.batchsize * m_param.depth_vol * m_param.height_vol * m_param.
        width_vol);
    layer_construct_forward(kernel::shaders::col2vol_spv, sizeof(kernel::shaders::col2vol_spv), x_, Format::kFormatFp32,
                            std::vector<int>{n_out_plane, output_length});

    float* t = (float*)y->toHost();
    std::cout << std::endl;
    for (int i = 0; i < n_out_plane; ++i)
    {
        std::cout << "[ ";
        for (int j = 0; j < output_length; ++j)
        {
            std::cout << t[i * output_length + j] << ", ";
        }
        std::cout << "]" << std::endl;
    }

    return y;
}

std::vector<int> col2vol::output_shape() const
{
    int d = static_cast<int>(m_param.depth_vol);
    int h = static_cast<int>(m_param.height_vol);
    int w = static_cast<int>(m_param.width_vol);
    return std::vector<int>{d, h, w};
}

namespace nn
{
    conv::conv(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
               bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size), m_stride(stride),
                                m_padding(padding), m_dilation(dilation), USE_BIAS(use_bias)
    {
        m_type = "conv";
        mm = std::make_shared<gemm>(gemm(1., 1., false));
        if (USE_BIAS)
            bias = std::make_shared<math::add>(math::add());
        trans = std::make_shared<transpose>(transpose(std::vector<int>{1, 0, 2, 3, 4}));
    }

    std::shared_ptr<tensor>& conv::operator()(const std::shared_ptr<tensor>& x_)
    {
        x = x_;

        auto input_shape = x->getShape();

        int channels = input_shape[1];
        int batch_size = input_shape[0];

        if (!kernel)
            kernel = std::make_shared<vol2col>(vol2col(channels, m_kernel_size, m_padding, m_stride, m_dilation));
        if (!w)
        {
            int c = static_cast<int>(channels * m_kernel_size.d * m_kernel_size.h * m_kernel_size.w);
            w = std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_num_filters, c}));
        }

        t1 = kernel->operator()(x); //27 9
        y = mm->operator()(w, t1);
        auto out = kernel->output_shape();

        if (USE_BIAS)
        {
            if (!b)
                b = std::make_shared<tensor>(tensor(1.0, y->getShape()));
            t2 = bias->operator()(y, b);
            t2->reShape(std::vector<int>{m_num_filters, batch_size, out[0], out[1], out[2]}); //8,9
            t4 = trans->operator()(t2);
        }
        else
        {
            y->reShape(std::vector<int>{m_num_filters, batch_size, out[0], out[1], out[2]}); //8,9
            t4 = trans->operator()(y);
        }

        return t4;
    }

    int conv::set_backward()
    {
        if (USE_BIAS)
        {
            bias->dy = dy;
            bias->is_bias = true;
            bias->set_backward();
            db = bias->dw;

            mm->dy = bias->dx;
            mm->set_backward();
            dx = mm->dx;
            dw = mm->dw;
        }
        else
        {
            mm->dy = dy;
            mm->set_backward();
            dx = mm->dx;
            dw = mm->dw;
        }

        return 1;
    }

    void conv::update_weight()
    {
    }

    convTranspose::convTranspose(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation,
                                 int padding_type,
                                 bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size),
                                                  m_stride(stride), m_padding(padding), m_dilation(dilation),
                                                  USE_BIAS(use_bias)
    {
        m_type = "convT";

        mm = std::make_shared<gemm>(gemm(1., 1., false));
        if (USE_BIAS)
            bias = std::make_shared<math::add>(math::add());
        trans = std::make_shared<transpose>(transpose(std::vector<int>{1, 0, 2, 3, 4}));
    }

    std::shared_ptr<tensor>& convTranspose::operator()(const std::shared_ptr<tensor>& x_)
    {
        x = x_;

        auto input_shape = x->getShape();

        int channels = input_shape[1];
        int batch_size = input_shape[0];
        if (!kernel)
        {
            //TODO dilation broken
            m_padding.d = (m_kernel_size.d - 1) * m_dilation.d - m_padding.d;
            m_padding.h = (m_kernel_size.h - 1) * m_dilation.h - m_padding.h;
            m_padding.w = (m_kernel_size.w - 1) * m_dilation.w - m_padding.w;
            m_stride.d = m_stride.d != 0 && m_stride.d > 1 ? 1 / m_stride.d : 1;
            m_stride.h = m_stride.h != 0 && m_stride.h > 1 ? 1 / m_stride.h : 1;
            m_stride.w = m_stride.w != 0 && m_stride.w > 1 ? 1 / m_stride.w : 1;
            kernel = std::make_shared<vol2col>(vol2col(channels, m_kernel_size, m_padding, m_stride, m_dilation));
        }
        if (!w)
        {
            int c = static_cast<int>(channels * m_kernel_size.d * m_kernel_size.h * m_kernel_size.w);
            w = std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_num_filters, c}));
        }

        t1 = kernel->operator()(x);
        y = mm->operator()(w, t1);
        auto out = kernel->output_shape();
        if (USE_BIAS)
        {
            if (!b)
                b = std::make_shared<tensor>(tensor(1.0, y->getShape()));
            t2 = bias->operator()(y, b);
            t2->reShape(std::vector<int>{m_num_filters, batch_size, out[0], out[1], out[2]});
            t4 = trans->operator()(t2);
        }
        else
        {
            y->reShape(std::vector<int>{m_num_filters, batch_size, out[0], out[1], out[2]});
            t4 = trans->operator()(y);
        }

        return t4;
    }

    int convTranspose::set_backward()
    {
        if (USE_BIAS)
        {
            bias->dy = dy;
            bias->is_bias = true;
            bias->set_backward();
            db = bias->dw;

            mm->dy = bias->dx;
            mm->set_backward();
            dx = mm->dx;
            dw = mm->dw;

            //col2im
        }
        else
        {
            mm->dy = dy;
            mm->set_backward();
            dx = mm->dx;
            dw = mm->dw;
        }

        return 1;
    }

    void convTranspose::update_weight()
    {
    }
}

py::array_t<float> im2col_cpu(py::array_t<float> input1, py::array_t<float> result, std::vector<int>& params)
{
    int batch_size = params[0];
    int in_channels = params[1];

    // std::vector<int> _vol = { params[2], params[3], params[4] };
    // std::vector<int> _col = { params[5], params[6], params[7] };
    // std::vector<int> kernel_size = { params[8], params[9], params[10] };
    // std::vector<int> stride = { params[11], params[12], params[13] };
    // std::vector<int> padding = { params[14], params[15], params[16] };
    // std::vector<int> dilation = { params[17], params[18], params[19] };

    int n_output_plane = in_channels * params[8] * params[9] * params[10];
    int output_length = batch_size * params[5] * params[6] * params[7];
    int index_length = in_channels * params[5] * params[6] * params[7];

    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = result.request();
    float* ptr1 = static_cast<float*>(buf1.ptr);
    float* ptr2 = static_cast<float*>(buf2.ptr);

    for (int elt = 0; elt < batch_size; ++elt)
    {
        int data_col = elt * in_channels * params[2] * params[3] * params[4];
        int data_vol = elt * n_output_plane * params[5] * params[6] * params[7];

        for (int index = 0; index < index_length; ++index)
        {
            int w_offset = index % params[10];
            int h_offset = (index / params[10]) % params[9];
            int d_offset = (index / params[10] / params[9]) % params[8];
            int c_vol = static_cast<int>(index / params[10] / params[9] / params[8]);

            for (int d_col = 0; d_col < params[5]; ++d_col)
            {
                int d_vol = d_col * params[11] - params[14] + d_offset * params[17];
                for (int h_col = 0; h_col < params[6]; ++h_col)
                {
                    int h_vol = h_col * params[12] - params[15] + h_offset * params[18];
                    for (int w_col = 0; w_col < params[7]; ++w_col)
                    {
                        int w_vol = w_col * params[13] - params[16] + w_offset * params[19];
                        if (d_vol >= 0 && d_vol < params[2] && h_vol >= 0 && h_vol < params[3] && w_vol >= 0 && w_vol < params[4
                        ])
                        {
                            int data_vol_idx = data_vol + ((c_vol * params[2] + d_vol) * params[3] + h_vol) * params[4] + w_vol;
                            int data_col_idx = data_col + ((index * params[5] + d_col) * params[6] + h_col) * params[7] + w_col;
                            if (data_col_idx < buf2.shape[0] && data_vol_idx < buf1.shape[0])
                                ptr2[data_col_idx] = ptr1[data_vol_idx];
                        }
                    }
                }
            }
        }
    }

    return result;
}

py::array_t<float> col2im_cpu(py::array_t<float> input1, py::array_t<float> result, std::vector<int>& params)
{
    int batch_size = params[0];
    int in_channels = params[1];

    // std::vector<int> _vol = { params[2], params[3], params[4] };
    // std::vector<int> _col = { params[5], params[6], params[7] };
    // std::vector<int> kernel_size = { params[8], params[9], params[10] };
    // std::vector<int> stride = { params[11], params[12], params[13] };
    // std::vector<int> padding = { params[14], params[15], params[16] };
    // std::vector<int> dilation = { params[17], params[18], params[19] };

    int n_output_plane = in_channels * params[8] * params[9] * params[10];
    int output_length = batch_size * params[5] * params[6] * params[7];
    int index_length = in_channels * params[5] * params[6] * params[7];

    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = result.request();
    float* ptr1 = static_cast<float*>(buf1.ptr);
    float* ptr2 = static_cast<float*>(buf2.ptr);

    for (int elt = 0; elt < batch_size; ++elt)
    {
        int data_col = elt * in_channels * params[2] * params[3] * params[4];
        int data_vol = elt * n_output_plane * params[5] * params[6] * params[7];

        for (int index = 0; index < index_length; ++index)
        {
            int w_offset = index % params[10];
            int h_offset = (index / params[10]) % params[9];
            int d_offset = (index / params[10] / params[9]) % params[8];
            int c_vol = static_cast<int>(index / params[10] / params[9] / params[8]);

            for (int d_col = 0; d_col < params[5]; ++d_col)
            {
                int d_vol = d_col * params[11] - params[14] + d_offset * params[17];
                for (int h_col = 0; h_col < params[6]; ++h_col)
                {
                    int h_vol = h_col * params[12] - params[15] + h_offset * params[18];
                    for (int w_col = 0; w_col < params[7]; ++w_col)
                    {
                        int w_vol = w_col * params[13] - params[16] + w_offset * params[19];
                        if (d_vol >= 0 && d_vol < params[2] && h_vol >= 0 && h_vol < params[3] && w_vol >= 0 && w_vol < params[4
                        ])
                        {
                            int data_vol_idx = data_vol + ((c_vol * params[2] + d_vol) * params[3] + h_vol) * params[4] + w_vol;
                            int data_col_idx = data_col + ((index * params[5] + d_col) * params[6] + h_col) * params[7] + w_col;
                            if (data_col_idx < buf2.shape[0] && data_vol_idx < buf1.shape[0])
                                ptr2[data_vol_idx] = ptr1[data_col_idx];
                        }
                    }
                }
            }
        }
    }

    return result;
}
