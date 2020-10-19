#include "common.h"
#include "utils.h"
#include "activation.h"

celu::celu(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "celu";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
}

std::shared_ptr<tensor>& celu::operator()(const std::shared_ptr<tensor>& x)
{
    alpha = std::make_shared<tensor>(tensor(1.0, x->getShape(), Format::kFormatFp32));
    return layer_construct_forward(kernel::shaders::celu_spv, sizeof(kernel::shaders::celu_spv), x, alpha);
}

elu::elu(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "elu";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_elu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_elu_spv);
}

std::shared_ptr<tensor>& elu::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::elu_spv, sizeof(kernel::shaders::elu_spv), x);
}

gelu::gelu(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "gelu";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_elu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_elu_spv);
}

std::shared_ptr<tensor>& gelu::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::gelu_spv, sizeof(kernel::shaders::gelu_spv), x);
}

hardshrink::hardshrink(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "hardshrink";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
}

std::shared_ptr<tensor>& hardshrink::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::hardshrink_spv, sizeof(kernel::shaders::hardshrink_spv), x);
}

hardsigmoid::hardsigmoid(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "hardsigmoid";
    m_param.alpha = 0.;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
}

std::shared_ptr<tensor>& hardsigmoid::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::hardsigmoid_spv, sizeof(kernel::shaders::hardsigmoid_spv), x);
}

hardswish::hardswish(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "hardswish";
    m_param.alpha = 0.;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
}

std::shared_ptr<tensor>& hardswish::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::hardswish_spv, sizeof(kernel::shaders::hardswish_spv), x);
}

hardtanh::hardtanh(float min_val, float max_val, bool in_place) : Base_Layer<two_param>(2, in_place)
{
    m_type = "hardtanh";
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    m_param = {0, min_val, max_val};
}

std::shared_ptr<tensor>& hardtanh::operator()(const std::shared_ptr<tensor>& x)
{
    m_param.total = x->count();
    return layer_construct_forward(kernel::shaders::hardshrink_spv, sizeof(kernel::shaders::hardshrink_spv), x);
}

leakyrelu::leakyrelu(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "leakyrelu";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
}

std::shared_ptr<tensor>& leakyrelu::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::leakyrelu_spv, sizeof(kernel::shaders::leakyrelu_spv), x);
}

logsigmoid::logsigmoid(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "logsigmoid";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
}

std::shared_ptr<tensor>& logsigmoid::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::logsigmoid_spv, sizeof(kernel::shaders::logsigmoid_spv), x);
}

prelu::prelu(float alpha, bool in_place) : Base_Layer<two_param>(2, in_place)
{
    m_type = "prelu";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
}

std::shared_ptr<tensor>& prelu::operator()(const std::shared_ptr<tensor>& x)
{
    m_param.beta = x->count() / m_param.alpha;

    return layer_construct_forward(kernel::shaders::prelu_spv, sizeof(kernel::shaders::prelu_spv), x);
}

relu::relu(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "relu";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::d_relu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_relu_spv);
}

std::shared_ptr<tensor>& relu::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::relu_spv, sizeof(kernel::shaders::relu_spv), x);
}

rrelu::rrelu(float lower, float upper, bool in_place) : Base_Layer<activation_param>(3, in_place), min(lower), max(upper)
{
    m_type = "rrelu";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::d_relu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_relu_spv);
}

std::shared_ptr<tensor>& rrelu::operator()(const std::shared_ptr<tensor>& x)
{
    w = std::make_shared<tensor>(tensor(init::uniform_distribution_init(x->getShape(), min, max), x->getShape()));
    return layer_construct_forward(kernel::shaders::rrelu_spv, sizeof(kernel::shaders::rrelu_spv), x, w);
}

selu::selu(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "selu";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
}

std::shared_ptr<tensor>& selu::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::selu_spv, sizeof(kernel::shaders::selu_spv), x);
}

sigmoid::sigmoid(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "sigmoid";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::d_sigmoid_spv;
    bck_codeSize = sizeof(kernel::shaders::d_sigmoid_spv);
}

std::shared_ptr<tensor>& sigmoid::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::sigmoid_spv, sizeof(kernel::shaders::sigmoid_spv), x);
}

softplus::softplus(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "softplus";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
}

std::shared_ptr<tensor>& softplus::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::softplus_spv, sizeof(kernel::shaders::softplus_spv), x);
}

softshrink::softshrink(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "softshrink";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
}

std::shared_ptr<tensor>& softshrink::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::softshrink_spv, sizeof(kernel::shaders::softshrink_spv), x);
}

softsign::softsign(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "softsign";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
}

std::shared_ptr<tensor>& softsign::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::softsign_spv, sizeof(kernel::shaders::softsign_spv), x);
}

tanhshrink::tanhshrink(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "tanhshrink";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::d_tanh_spv;
    bck_codeSize = sizeof(kernel::shaders::d_tanh_spv);
}

std::shared_ptr<tensor>& tanhshrink::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(kernel::shaders::tanhshrink_spv, sizeof(kernel::shaders::tanhshrink_spv), x);
}

void init_celu(py::module& m)
{
    py::class_<celu>(m, "celu")
        .def(py::init<float&, bool&>());
}

void init_elu(py::module& m)
{
    py::class_<elu>(m, "elu")
        .def(py::init<float&, bool&>());
}

void init_gelu(py::module& m)
{
    py::class_<gelu>(m, "gelu")
        .def(py::init<float&, bool&>());
}

void init_hardshrink(py::module& m)
{
    py::class_<hardshrink>(m, "hardshrink")
        .def(py::init<float&, bool&>());
}

void init_hardsigmoid(py::module& m)
{
    py::class_<hardsigmoid>(m, "hardsigmoid")
        .def(py::init<bool&>());
}

void init_hardswish(py::module& m)
{
    py::class_<hardswish>(m, "hardswish")
        .def(py::init<bool&>());
}

void init_hardtanh(py::module& m)
{
    py::class_<hardtanh>(m, "hardtanh")
        .def(py::init<float&, float&, bool&>());
}

void init_leakyrelu(py::module& m)
{
    py::class_<leakyrelu>(m, "leakyrelu")
        .def(py::init<float&, bool&>());
}

void init_logsigmoid(py::module& m)
{
    py::class_<logsigmoid>(m, "logsigmoid")
        .def(py::init<float&, bool&>());
}

void init_prelu(py::module& m)
{
    py::class_<prelu>(m, "prelu")
        .def(py::init<float&, bool&>());
}

void init_relu(py::module& m)
{
    py::class_<relu>(m, "relu")
        .def(py::init<bool&>())
        .def("__call__", &relu::operator());
}

void init_rrelu(py::module& m)
{
    py::class_<rrelu>(m, "rrelu")
        .def(py::init<float&, float&, bool&>());
}

void init_selu(py::module& m)
{
    py::class_<selu>(m, "selu")
        .def(py::init<bool&>());
}

void init_sigmoid(py::module& m)
{
    py::class_<sigmoid>(m, "sigmoid")
        .def(py::init<bool&>());
}

void init_softplus(py::module& m)
{
    py::class_<softplus>(m, "softplus")
        .def(py::init<float&, bool&>());
}

void init_softshrink(py::module& m)
{
    py::class_<softshrink>(m, "softshrink")
        .def(py::init<float&, bool&>());
}

void init_softsign(py::module& m)
{
    py::class_<softsign>(m, "softsign")
        .def(py::init<bool&>());
}

void init_tanshrink(py::module& m)
{
    py::class_<tanhshrink>(m, "tanhshrink")
        .def(py::init<bool&>());
}
