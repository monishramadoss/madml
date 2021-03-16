#include "../kernel/common.h"
#include "../kernel/utils.h"

#include "gemm.h"


const std::string gemm_shader = R"(
#version 460
layout(push_constant) uniform pushBlock {
    int total;
    int b;
    float alpha;
    float beta;
    bool use_bias;
    int M;
    int N;
    int K;
};

layout (local_size_x = 64, local_size_y = 64, local_size_z = 1) in;
layout (binding = 0) readonly buffer ssbA { float A[]; };
layout (binding = 1) readonly buffer ssbB { float B[]; };
layout (binding = 2) readonly buffer ssbC { float C[]; };
layout (binding = 3) buffer ssbD { float D[]; };

void gemm_1(){   
    for(uint globalDepth = gl_GlobalInvocationID.z; globalDepth < b; globalDepth += gl_NumWorkGroups.z * gl_WorkGroupSize.z){
        for (uint globalRow = gl_GlobalInvocationID.x; globalRow < M; globalRow += gl_NumWorkGroups.x * gl_WorkGroupSize.x){
            for (uint globalCol = gl_GlobalInvocationID.y; globalCol < N; globalCol += gl_NumWorkGroups.y * gl_WorkGroupSize.y){
                float acc = use_bias ? beta * C[globalRow*N + globalCol]: 0.0;
                for (uint k=0u; k < K; k++)
                    acc += p.alpha * A[globalDepth*M*K + globalRow*K + k] * B[k*N + globalCol]; 
                D[globalDepth*M*N + globalRow*N + globalCol] = acc;
            }
        }
    }
}

void main() {
    gemm_1();
})";


gemm::gemm(float alpha, float beta, bool use_bias)
{
    initVulkanThing(3);
    m_type = "gemm";
    m_param.alpha = alpha;
    m_param.beta = beta;
    m_param.use_bias = use_bias;
}

void gemm::forward(std::shared_ptr<tensor>& y, const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w, const std::shared_ptr<tensor>& b)
{
    if (m_pipeline == nullptr)
    {
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
        createShaderModule(nullptr, 0, gemm_shader);
        createPipeline(sizeof(gemm_param));
    }

    bindtensor(x, 0);
    bindtensor(w, 1);
    bindtensor(b, 2);
    bindtensor(y, 3);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(gemm_param));
    runCommandBuffer();
}