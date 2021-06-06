#include "../engine/common.h"
#include "../engine/utils.h"
#include "convolution.h"

constexpr int local_sz_x_conv = 16;
constexpr int local_sz_y_conv = 64;

vol2col::vol2col(std::vector<int>& params)
{
    m_future = std::async(&vol2col::initVulkanThing, &*this, 2);
    m_type = "vol2col";

    m_param.batchsize = params[0];
    m_param.channels = params[1];

    m_param.kernel_d = params[2];
    m_param.kernel_h = params[3];
    m_param.kernel_w = params[4];

    m_param.pad_d = params[5];
    m_param.pad_h = params[9];
    m_param.pad_w = params[7];

    m_param.stride_d = params[8];
    m_param.stride_h = params[9];
    m_param.stride_w = params[10];

    m_param.dilation_d = params[11];
    m_param.dilation_h = params[12];
    m_param.dilation_w = params[13];

    m_param.depth_col = params[14];
    m_param.height_col = params[15];
    m_param.width_col = params[16];

    m_param.depth_vol = params[17];
    m_param.height_vol = params[18];
    m_param.width_vol = params[19];

    m_futures.resize(2);
}

void vol2col::forward(tensor& col, tensor& vol)
{
    if (m_pipeline == nullptr)
    {
        m_param.total = vol.count();

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
        m_future.wait();
        createShaderModule(vol2col_spv, sizeof(vol2col_spv));
        createPipeline(sizeof(vol2col_param));
    }

    bindtensor(vol, 0);
    bindtensor(col, 1);
    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(vol2col_param));
}

col2vol::col2vol(std::vector<int>& params)
{
    m_future = std::async(&vol2col::initVulkanThing, &*this, 2);
    m_type = "col2vol";
    m_param.batchsize = params[0];
    m_param.channels = params[1];

    m_param.kernel_d = params[2];
    m_param.kernel_h = params[3];
    m_param.kernel_w = params[4];

    m_param.pad_d = params[5];
    m_param.pad_h = params[9];
    m_param.pad_w = params[7];

    m_param.stride_d = params[8];
    m_param.stride_h = params[9];
    m_param.stride_w = params[10];

    m_param.dilation_d = params[11];
    m_param.dilation_h = params[12];
    m_param.dilation_w = params[13];

    m_param.depth_col = params[14];
    m_param.height_col = params[15];
    m_param.width_col = params[16];

    m_param.depth_vol = params[17];
    m_param.height_vol = params[18];
    m_param.width_vol = params[19];
}

void col2vol::forward(tensor& vol, tensor& col)
{
    if (m_pipeline == nullptr)
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
        m_future.wait();
        createShaderModule(col2vol_spv, sizeof(col2vol_spv));
        createPipeline(sizeof(vol2col_param));
    }

    bindtensor(vol, 0);
    bindtensor(col, 1);
    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(vol2col_param));
}

void cpu_vol2col(py::array_t<float, py::array::c_style | py::array::forcecast> vol,
    py::array_t<float, py::array::c_style | py::array::forcecast> col,
    int n_output_plane, int index_length, std::vector<int>& params)
{
    const float* vol_data_ptr = (float*)vol.data();
    float* col_data_ptr = (float*)col.data();

    int batchsize = params[0];
    int channels = params[1];

    int kernel_d = params[2];
    int kernel_h = params[3];
    int kernel_w = params[4];

    int pad_d = params[5];
    int pad_h = params[9];
    int pad_w = params[7];

    int stride_d = params[8];
    int stride_h = params[9];
    int stride_w = params[10];

    int dilation_d = params[11];
    int dilation_h = params[12];
    int dilation_w = params[13];

    int depth_col = params[14];
    int height_col = params[15];
    int width_col = params[16];

    int depth_vol = params[17];
    int height_vol = params[18];
    int width_vol = params[19];

    for (int elt = 0; elt < batchsize; ++elt)
    {
        int data_vol = elt * channels * depth_vol * height_vol * width_vol;
        int data_col = elt * n_output_plane * depth_col * height_col * width_col;
        for (int c_col = 0; c_col < index_length; ++c_col)
        {
            int w_offset = c_col % kernel_w;
            int h_offset = (c_col / kernel_w) % kernel_h;
            int d_offset = (c_col / kernel_w / kernel_h) % kernel_d;
            int c_vol = c_col / kernel_w / kernel_h / kernel_d;
            for (int d_col = 0; d_col < depth_col; ++d_col)
            {
                int d_vol = d_col * stride_d - pad_d + d_offset * dilation_d;
                for (int h_col = 0; h_col < height_col; ++h_col)
                {
                    int h_vol = h_col * stride_h - pad_h + h_offset * dilation_h;
                    for (int w_col = 0; w_col < width_col; ++w_col)
                    {
                        int w_vol = w_col * stride_w - pad_w + w_offset * dilation_w;
                        if ((0 <= d_vol && d_vol < depth_vol) &&
                            (0 <= h_vol && h_vol < height_vol) &&
                            (0 <= w_vol && w_vol < width_vol))
                        {
                            int vol_idx = data_vol + (((c_vol * depth_vol + d_vol) * height_vol + h_vol) * width_vol + w_vol);
                            int col_idx = data_col + (((c_col * depth_col + d_col) * height_col + h_col) * width_col + w_col);
                            col_data_ptr[col_idx] = vol_data_ptr[vol_idx];
                        }
                    }
                }
            }
        }
    }
}
void cpu_col2vol(py::array_t<float, py::array::c_style | py::array::forcecast> vol,
    py::array_t<float, py::array::c_style | py::array::forcecast> col,
    int n_output_plane, int index_length, std::vector<int>& params)
{
    float* vol_data_ptr = (float*)vol.data();
    const float* col_data_ptr = (float*)col.data();

    int batchsize = params[0];
    int channels = params[1];

    int kernel_d = params[2];
    int kernel_h = params[3];
    int kernel_w = params[4];

    int pad_d = params[5];
    int pad_h = params[9];
    int pad_w = params[7];

    int stride_d = params[8];
    int stride_h = params[9];
    int stride_w = params[10];

    int dilation_d = params[11];
    int dilation_h = params[12];
    int dilation_w = params[13];

    int depth_col = params[14];
    int height_col = params[15];
    int width_col = params[16];

    int depth_vol = params[17];
    int height_vol = params[18];
    int width_vol = params[19];

    for (int elt = 0; elt < batchsize; ++elt)
    {
        int data_vol = elt * channels * depth_vol * height_vol * width_vol;
        int data_col = elt * n_output_plane * depth_col * height_col * width_col;
        for (int c_col = 0; c_col < index_length; ++c_col)
        {
            int w_offset = c_col % kernel_w;
            int h_offset = (c_col / kernel_w) % kernel_h;
            int d_offset = (c_col / kernel_w / kernel_h) % kernel_d;
            int c_vol = c_col / kernel_w / kernel_h / kernel_d;
            for (int d_col = 0; d_col < depth_col; ++d_col)
            {
                int d_vol = d_col * stride_d - pad_d + d_offset * dilation_d;
                for (int h_col = 0; h_col < height_col; ++h_col)
                {
                    int h_vol = h_col * stride_h - pad_h + h_offset * dilation_h;
                    for (int w_col = 0; w_col < width_col; ++w_col)
                    {
                        int w_vol = w_col * stride_w - pad_w + w_offset * dilation_w;
                        if (0 <= d_vol && d_vol < depth_vol && 0 <= h_vol && h_vol < height_vol && 0 <= w_vol && w_vol < width_vol)
                        {
                            int vol_idx = data_vol + (((c_vol * depth_vol + d_vol) * height_vol + h_vol) * width_vol + w_vol);
                            int col_idx = data_col + (((c_col * depth_col + d_col) * height_col + h_col) * width_col + w_col);
                            vol_data_ptr[vol_idx] += col_data_ptr[col_idx];
                        }
                    }
                }
            }
        }
    }
}