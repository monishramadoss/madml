#include "../engine/common.h"
#include "../engine/utils.h"

#include "pooling.h"
#include <future>

max_reduce::max_reduce(int in_channels, int batch_size, bool derivative) : m_derivative(derivative)
{
    m_future = std::async(&max_reduce::initVulkanThing, &*this, 3);
    m_param.y_size = in_channels * batch_size;
}

void max_reduce::forward(tensor& y, tensor& col, tensor& max_idx)
{
    if (m_pipeline == nullptr)
    {
        m_param.channel_offset = col.getShape()[1];
        m_param.out_size = y.getShape()[1];
        m_group_x = static_cast<int>(alignSize(m_param.y_size, 4)) / 4;
        m_group_y = static_cast<int>(alignSize(m_param.out_size, 16)) / 16;
        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count - 1;
        if (m_group_y > max_compute_work_group_count)
            m_group_y = max_compute_work_group_count - 1;
        m_future.wait();
        if (m_derivative)
            createShaderModule(d_max_reduce_spv, sizeof(d_max_reduce_spv));
        else
            createShaderModule(max_reduce_spv, sizeof(max_reduce_spv));
        createPipeline(sizeof(max_reduce_param));
    }

    bindtensor(col, 0);
    bindtensor(y, 1);
    bindtensor(max_idx, 2);
    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(max_reduce_param));
}