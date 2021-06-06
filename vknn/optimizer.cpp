#include "../engine/common.h"
#include "../engine/utils.h"
#include "optimizer.h"
#include <future>

sgd::sgd(float lr, float momentum, float dampening, float weight_decay, bool nestrov)
{
    m_future = std::async(&sgd::initVulkanThing, &*this, 3);
    m_param.lr = lr;
    m_param.momentum = momentum;
    m_param.dampening = dampening;
    m_param.weight_decay = weight_decay;
    m_param.nestrov = nestrov;
}

void sgd::forward(tensor& p, tensor& dp, tensor& v)
{
    if (m_pipeline == nullptr)
    {
        m_param.total = p.count();
        m_group_x = static_cast<int>(alignSize(m_param.total, 1024)) / 1024;
        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count - 1;
        m_future.wait();
        createShaderModule(sgd_spv, sizeof(sgd_spv));
        createPipeline(sizeof(sgd_param));
    }

    bindtensor(p, 0);
    bindtensor(dp, 1);
    bindtensor(v, 2);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(sgd_param));
}

adam::adam(float lr, float beta_a, float beta_b, float eps, float weight_decay, bool amsgrad)
{
    m_future = std::async(&adam::initVulkanThing, &*this, 6);
    m_param.lr = lr;
    m_param.beta_a = beta_a;
    m_param.beta_b = beta_b;
    m_param.eps = eps;
    m_param.weight_decay = weight_decay;
    m_param.amsgrad = amsgrad;
}

void adam::forward(int counter, tensor& p, tensor& dp, tensor& m, tensor& r, tensor& m_k_hat, tensor& r_k_hat)
{
    if (m_pipeline == nullptr)
    {
        m_param.total = p.count();
        m_group_x = static_cast<int>(alignSize(m_param.total, 1024)) / 1024;
        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count - 1;
        m_future.wait();
        createShaderModule(adam_spv, sizeof(adam_spv));
        createPipeline(sizeof(adam_param));
    }
    m_param.counter = counter;

    bindtensor(p, 0);
    bindtensor(dp, 1);
    bindtensor(m, 2);
    bindtensor(r, 3);
    bindtensor(m_k_hat, 4);
    bindtensor(r_k_hat, 5);
    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(adam_param));
};

adagrad::adagrad(float lr, float eps, float lr_decay, float weight_decay)
{
    m_future = std::async(&adagrad::initVulkanThing, &*this, 3);
    m_param.lr = lr;
    m_param.eps = eps;
    m_param.lr_decay = lr_decay;
    m_param.weight_decay = weight_decay;
}

void adagrad::forward(int counter, tensor& p, tensor& dp, tensor& v)
{
    if (m_pipeline == nullptr)
    {
        m_param.total = p.count();
        m_group_x = static_cast<int>(alignSize(m_param.total, 1024)) / 1024;
        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count - 1;
        m_future.wait();

        //createShaderModule(adagrad_spv, sizeof(adagrad_spv));
        createPipeline(sizeof(adagrad_param));
    }
    m_param.counter = counter;
    bindtensor(p, 0);
    bindtensor(dp, 1);
    bindtensor(v, 2);
    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(adagrad_param));
}

rmsprop::rmsprop(float lr, float alpha, float eps, float weight_decay, float momentum, bool centered)
{
    m_future = std::async(&adagrad::initVulkanThing, &*this, 3);
    m_param.lr = lr;
    m_param.alpha = alpha;
    m_param.eps = eps;
    m_param.weight_decay = weight_decay;
    m_param.momentum = momentum;
    m_param.centered = centered;
}

void rmsprop::forward(tensor& p, tensor& dp, tensor& v)
{
    if (m_pipeline == nullptr)
    {
        m_param.total = p.count();
        m_group_x = static_cast<int>(alignSize(m_param.total, 1024)) / 1024;
        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count - 1;
        m_future.wait();

        //createShaderModule(rmsprop_spv, sizeof(rmsprop_spv));
        createPipeline(sizeof(rmsprop_param));
    }

    bindtensor(p, 0);
    bindtensor(dp, 1);
    bindtensor(v, 2);
    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(rmsprop_param));
}