#include "../kernel/common.h"
#include "../kernel/utils.h"
#include "activation.h"

relu::relu(bool in_place): inplace(in_place)
{
    initVulkanThing(2);
    m_param.alpha = 0.f;
}

void relu::forward(std::shared_ptr<tensor>& y, const std::shared_ptr<tensor>& x)
{
    if (m_pipeline == nullptr)
    {
        m_param.total = x->count();

        m_group_x = static_cast<int>(alignSize(m_param.total, 1024)) / 1024;
        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count - 1;

        //createshaderModule(fwd_shader, fwd_codeSize);
        createPipeline(sizeof(single_param));
    }

    bindtensor(*x, 0);
    if (inplace)
        bindtensor(*x, 1);
    else
        bindtensor(*y, 1);
    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(single_param));
    runCommandBuffer();
}