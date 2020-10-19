#include "common.h"
#include "utils.h"
#include "normalization.h"

BatchNormalization::BatchNormalization(float eps, float momentum, int num_features, bool affine) : Base_Layer<norm_param>(5)
{
    m_type = "batch normalization";
    m_param.eps = eps;
    m_param.momentum = momentum;
}

void BatchNormalization::computeGroupCount()
{
    m_group_x = static_cast<int>(alignSize(m_param.lower_offset, 1024)) / 1024;
    m_group_y = 1;
    m_group_z = 1;

    if (m_group_x > max_compute_work_group_count)
        m_group_x = max_compute_work_group_count - 1;
    if (m_group_y > max_compute_work_group_count)
        m_group_y = max_compute_work_group_count - 1;
    if (m_group_z > max_compute_work_group_count)
        m_group_z = max_compute_work_group_count - 1;
}

std::shared_ptr<tensor>& BatchNormalization::operator()(const std::shared_ptr<tensor>& x)
{
    this->x = x;
    float f = 0.f;
    m_param.upper_offset = x->getShape()[0];
    m_param.C = x->getShape()[1];
    m_param.lower_offset = x->count() / m_param.upper_offset / m_param.C;
    if (!t1)
        t1 = std::make_shared<tensor>(tensor(1., std::vector<int>{m_param.C * 2}));
    if (!t2)
        t2 = std::make_shared<tensor>(tensor(1., std::vector<int>{m_param.C * 2}));
    if (!t3)
        t3 = std::make_shared<tensor>(tensor(1., std::vector<int>{m_param.upper_offset, m_param.lower_offset, 2}));
    if (!y)
        y = std::make_shared<tensor>(tensor(0., x->getShape()));

    if (m_pipeline == nullptr)
    {
        m_param.total = x->count();
        computeGroupCount();
        createShaderModule(kernel::shaders::batch_normalization_spv, sizeof(kernel::shaders::batch_normalization_spv));
        createPipeline(sizeof(norm_param));
    }

    bindTensor(x, 0);
    bindTensor(t1, 1);
    bindTensor(t2, 2);
    bindTensor(y, 3);
    bindTensor(t3, 4);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(norm_param));
    runCommandBuffer();

    return y;
}

int BatchNormalization::set_backward() { return -1; }

InstanceNormalization::InstanceNormalization(float eps, float momentum, int num_features, bool affine) : Base_Layer<norm_param>(
    2)
{
    m_type = "instance normalization";
    m_param.eps = eps;
    m_param.momentum = momentum;
}

void InstanceNormalization::computeGroupCount()
{
}

std::shared_ptr<tensor>& InstanceNormalization::operator()(const std::shared_ptr<tensor>& x)
{
    if (!y)
        y = std::make_shared<tensor>(tensor(0., x->getShape()));

    return y;
}

int InstanceNormalization::set_backward() { return -1; }

LayerNormalization::LayerNormalization(float eps, float momentum, int num_features, bool affine) : Base_Layer<norm_param>(2)
{
    m_type = "layer normalization";
    m_param.eps = eps;
    m_param.momentum = momentum;
}

void LayerNormalization::computeGroupCount()
{
}

std::shared_ptr<tensor>& LayerNormalization::operator()(const std::shared_ptr<tensor>& x)
{
    if (!y)
        y = std::make_shared<tensor>(tensor(0., x->getShape()));

    return y;
}

int LayerNormalization::set_backward() { return -1; }
