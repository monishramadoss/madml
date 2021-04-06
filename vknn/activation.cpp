#include "../engine/common.h"
#include "../engine/utils.h"
#include "activation.h"

relu::relu(bool in_place, bool derivative): m_inplace(in_place), m_derivative(derivative)
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

    m_futures[0] = std::async(&relu::bindtensor, &*this, x, 0);
    m_futures[1] = std::async(&relu::bindtensor, &*this, w, 1);

    if (m_inplace)
        m_futures[2] = std::async(&relu::bindtensor, &*this, x, 2);
    else
        m_futures[2] = std::async(&relu::bindtensor, &*this, y, 2);

    m_future = std::async(&relu::recordCommandBuffer, &*this,  static_cast<void*>(&m_param), sizeof(single_param));
    runCommandBuffer();
}