#include "../kernel/common.h"
#include "../kernel/utils.h"

#include "gemm.h"


gemm::gemm(float alpha, float beta)
{
    initVulkanThing(3);
    m_type = "gemm";
    m_param.alpha = alpha;
    m_param.beta = beta;
}

void gemm::forward(std::shared_ptr<tensor>& y, const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w){
    if(m_pipeline == nullptr){
        if (x->getShape().size() == w->getShape().size() + 1)
        {
            m_param.batchsize = x->getShape()[0];
            m_param.m = x->getShape()[1];
            m_param.k = x->getShape()[2];
            m_param.n = w->getShape()[1];
        }
        else
        {
            m_param.batchsize = 1;
            m_param.m = x->getShape()[0];
            m_param.k = x->getShape()[1];
            m_param.n = w->getShape()[1];
        }   
        m_param.total = w->count();

        m_group_x = static_cast<int>(alignSize(m_param.m, 64)) / 64;
        m_group_y = static_cast<int>(alignSize(m_param.n, 64)) / 64;
        m_group_z = static_cast<int>(alignSize(m_param.batchsize, 1)) / 1;

        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count - 1;
        if (m_group_y > max_compute_work_group_count)
            m_group_y = max_compute_work_group_count - 1;
        if (m_group_z > max_compute_work_group_count)
            m_group_z = max_compute_work_group_count - 1;


        //createShaderModule(fwd_shader, fwd_codeSize);
        createPipeline(sizeof(gemm_param));
    }

    bindtensor(x, 0);
    bindtensor(w, 1);
    bindtensor(y, 2);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(gemm_param));
    runCommandBuffer();
}