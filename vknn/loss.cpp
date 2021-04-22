#include "../engine/common.h"
#include "../engine/utils.h"
#include "loss.h"

mse::mse(bool reduction)
{
    m_future = std::async(&mse::initVulkanThing, &*this, 3);
    m_param.reduction = reduction;
}

void mse::forward(tensor& loss, tensor& l, tensor& t, tensor& dx)
{
    if (m_pipeline == nullptr)
    {
        m_param.total = l.count();
        m_group_x = static_cast<int>(alignSize(m_param.total, 1024)) / 1024;
        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count - 1;

        m_future.wait();
        createShaderModule(mse_spv, sizeof(mse_spv));

        createPipeline(sizeof(mse_param));
    }

    bindtensor(loss, 0);
    bindtensor(l, 1);
    bindtensor(t, 2);
    bindtensor(dx, 3);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(mse_param));

    return;
}