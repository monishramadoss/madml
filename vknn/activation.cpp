#include "../engine/common.h"
#include "../engine/utils.h"
#include "activation.h"

relu::relu(bool in_place, bool derivative) : m_inplace(in_place), m_derivative(derivative)
{
    m_future = std::async(&relu::initVulkanThing, &*this, 3);
    m_param.alpha = 1.f;
    m_futures.resize(3);
}

void relu::forward(tensor& y, tensor& x, tensor& w)
{
    if (m_pipeline == nullptr)
    {
        m_param.total = x.count();

        m_group_x = static_cast<int>(alignSize(m_param.total, local_sz_x)) / local_sz_x;
        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count - 1;

        m_future.wait();
        if (m_derivative)
            createShaderModule(d_relu_spv, sizeof(d_relu_spv));
        else
            createShaderModule(relu_spv, sizeof(relu_spv));
        createPipeline(sizeof(single_param));
    }

    bindtensor(x, 0);
    bindtensor(w, 1);

    if (m_inplace)
        bindtensor(x, 2);
    else
        bindtensor(y, 2);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(single_param));
}