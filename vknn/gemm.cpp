#include "../engine/common.h"
#include "../engine/utils.h"

#include "gemm.h"
#include <future>

gemm::gemm(float alpha, float beta, bool use_bias, bool transpose_x, bool transpose_w) : m_transpose_w(transpose_w), m_transpose_x(transpose_x)
{
    m_future = std::async(&gemm::initVulkanThing, &*this, 4);
    m_type = "gemm";
    m_param.alpha = alpha;
    m_param.beta = beta;
    m_param.use_bias = use_bias;
    m_futures.resize(4);
}

void gemm::forward(tensor& y, tensor& x, tensor& w, tensor& b)
{
    if (m_pipeline == nullptr)
    {
        //TODO nneds to switch modes based on relationship to y

        int stage = 0;
        auto out_shape = y.getShape(); // 32 x 38336
        auto in_shape_1 = x.getShape(); // 9 x 38336
        auto in_shape_2 = w.getShape(); // 32 x 9
        auto tmp2 = out_shape.size() > 2 && out_shape[0] == in_shape_1[0] && out_shape[2] == in_shape_2[2]
            && out_shape[1] == in_shape_1[1];

        if (out_shape[0] == in_shape_1[0] && out_shape[1] == in_shape_2[1]) // mxn & nxk = mxk
        {
            m_param.batchsize = out_shape[0];
            m_param.m = 1;
            m_param.n = out_shape[1];
            m_param.k = in_shape_2[1];
        }
        else if (out_shape[0] == in_shape_1[1] && out_shape[1] == in_shape_2[1])
        {
            m_param.batchsize = 1;
            m_param.m = out_shape[0];
            m_param.n = out_shape[1];
            m_param.k = in_shape_1[0];
            m_transpose_x = true;
        }
        else if (out_shape[0] == in_shape_1[0] && out_shape[1] == in_shape_2[0])
        {
            m_param.batchsize = 1;
            m_param.m = out_shape[0];
            m_param.n = out_shape[1];
            m_param.k = in_shape_2[1];
            m_transpose_w = true;
        }
        else
        {
            std::cerr << " GEMM y:[" << out_shape[0] << " " << out_shape[1] << "] X:[" << in_shape_1[0] << " " << in_shape_1[1] << "] Y:[" << in_shape_2[0] << " " << in_shape_2[1] << "]\n";
            std::cerr << "\n";
            throw std::runtime_error("gemm cannot compute");
        }

        m_param.total = w.count();
        m_group_x = static_cast<int>(alignSize(m_param.m, 16)) / 16;
        m_group_y = static_cast<int>(alignSize(m_param.n, 16)) / 16;
        m_group_z = static_cast<int>(alignSize(m_param.batchsize, 2)) / 2;

        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count - 1;
        if (m_group_y > max_compute_work_group_count)
            m_group_y = max_compute_work_group_count - 1;
        if (m_group_z > max_compute_work_group_count)
            m_group_z = max_compute_work_group_count - 1;
        m_future.wait();
        if (m_transpose_x)
            createShaderModule(xt_gemm_spv, sizeof(xt_gemm_spv));
        else if (m_transpose_w)
            createShaderModule(wt_gemm_spv, sizeof(wt_gemm_spv));
        else
            createShaderModule(gemm_spv, sizeof(gemm_spv));
        createPipeline(sizeof(gemm_param));
    }

    bindtensor(x, 0);
    bindtensor(w, 1);
    bindtensor(b, 2);
    bindtensor(y, 3);
    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(gemm_param));
}