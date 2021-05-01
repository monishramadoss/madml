#include "../engine/common.h"
#include "../engine/utils.h"
#include "transform.h"

std::vector<int> prepareStrides(const Shape& shape_before, const Shape& shape_after, Shape& stride)
{
    size_t dims = shape_before.size();
    stride[2 * dims - 1] = 1;
    stride[3 * dims - 1] = 1;

    for (int64_t i = dims - 2; i >= 0; i--)
    {
        stride[dims * 2 + i] = stride[dims * 2 + i + 1] * shape_before[i + 1];
        stride[dims + i] = stride[dims + i + 1] * shape_after[i + 1];
    }
    return stride;
}

transpose::transpose(std::vector<int>& order)
{
    m_future = std::async(&transpose::initVulkanThing, &*this, 3);
    m_param.num_axes = static_cast<int>(order.size());
    m_stride.resize(order.size() * 3);
    for (int i = 0; i < order.size(); ++i)
        m_stride[i] = order[i];
    m_futures.resize(3);
}

void transpose::forward(tensor& y, tensor& x)
{
    if (m_pipeline == nullptr)
    {
        std::vector<int> new_shape(x.dimNum());
        std::vector<int> old_shape = x.getShape();
        for (int i = 0; i < old_shape.size(); ++i)
        {
            new_shape[i] = old_shape[m_stride[i]];
        }

        m_stride = prepareStrides(old_shape, new_shape, m_stride);
        m_stride_tensor = tensor((char*)m_stride.data(), std::vector<int>{m_param.num_axes * 3}, Format::kFormatInt32);
        m_param.total = x.count();

        m_group_x = static_cast<int>(alignSize(m_param.total, local_sz_x)) / local_sz_x;
        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count;
        m_future.wait();
        createShaderModule(transpose_spv, sizeof(transpose_spv));
        createPipeline(sizeof(transpose_param));
    }

    bindtensor(x, 0);
    bindtensor(m_stride_tensor, 1);
    bindtensor(y, 2);
    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(transpose_param));  
}