#include "common.h"
#include "utils.h"
#include "transform.h"

constexpr int local_sz_x_conv = 16;
constexpr int local_sz_y_conv = 64;

copy::copy() : Base_Layer<>(2)
{
    m_type = "copy";
    fwd_shader = kernel::shaders::unary_operator_spv;
    fwd_codeSize = sizeof(kernel::shaders::unary_operator_spv);

}

std::shared_ptr<tensor>& copy::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}

void copy::computeGroupCount()
{
    m_group_x = static_cast<int>(alignSize(m_param.total, local_sz_x)) / local_sz_x;
    if (m_group_x > max_compute_work_group_count)
        m_group_x = max_compute_work_group_count;
    m_group_y = 1;
    m_group_z = 1;
}

std::vector<int> prepareStrides(const Shape& shape_before, const Shape& shape_after, Shape& stride)
{
    size_t dims = shape_before.size();
    stride[2 * dims - 1] = 1;
    stride[3 * dims - 1] = 1;

    for (int64_t i = dims - 2; i >= 0; i--)
    {
        stride[dims * 2 + i] = stride[dims * 2 + i + 1] * shape_before[i + 1];
        stride[dims + i] = stride[dims + i + 1] * shape_after[i + 1];
    }
    return stride;
}

transpose::transpose(const std::vector<int>& order) : Base_Layer<transpose_param>(3)
{
    m_type = "transpose";
    m_param.num_axes = static_cast<int>(order.size());
    stride.resize(order.size() * 3);
    d_stride.resize(order.size() * 3);
    for (size_t i = 0; i < m_param.num_axes; ++i)
        stride[i] = order[i];
    for (size_t i = 0; i < m_param.num_axes; ++i)
        d_stride[i] = order[i];
    bck_shader = kernel::shaders::transpose_spv;
    bck_codeSize = sizeof(kernel::shaders::transpose_spv);
    fwd_shader = kernel::shaders::transpose_spv;
    bck_codeSize = sizeof(kernel::shaders::transpose_spv);
}

std::shared_ptr<tensor>& transpose::operator()(const std::shared_ptr<tensor>& _x)
{
    if (!w || !new_shape.size())
    {
        new_shape.resize(stride.size() / 3);
        for (size_t i = 0; i < m_param.num_axes; ++i)
            new_shape[i] = _x->getShape()[stride[i]];
        old_shape = _x->getShape();
        stride = prepareStrides(old_shape, new_shape, stride);
        w = std::make_shared<tensor>(tensor((char*)stride.data(), std::vector<int>{m_param.num_axes * 3},
            Format::kFormatInt32));
    }
    layer_construct_forward(_x, w, Format::kFormatFp32, new_shape);
    return y;
}

int transpose::set_backward()
{
    if (!dw)
    {
        d_stride = prepareStrides(new_shape, old_shape, d_stride);
        dw = std::make_shared<tensor>(tensor((char*)d_stride.data(), std::vector<int>{m_param.num_axes * 3},
            Format::kFormatInt32));
    }
    derivative->fwd_shader = bck_shader;
    derivative->fwd_codeSize = bck_codeSize;
    dx = derivative->layer_construct_forward(dy, dw, Format::kFormatFp32, old_shape);
    return dy->getId();
}

void transpose::computeGroupCount()
{
    m_group_x = static_cast<int>(alignSize(m_param.total, local_sz_x)) / local_sz_x;
    if (m_group_x > max_compute_work_group_count)
        m_group_x = max_compute_work_group_count;
    m_group_y = 1;
    m_group_z = 1;
}

void init_transpose(py::module& m)
{
    py::class_<transpose>(m, "transpose")
        .def(py::init<const std::vector<int>&>());
}