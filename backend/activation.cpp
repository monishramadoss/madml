#include "common.h"
#include "utils.h"
#include "activation.h"

std::string ACTIVATION_SHADER_BASE(const char* params, const char* body)
{
    std::string activation_base = R"(
#Version 450
#define LOCAL_SZ_X 1024
layout(binding = 0) readonly buffer buf1 { float X[]; };
layout(binding = 1) writeonly buffer buf2 { float Y[]; };
layout(local_size_x = LOCAL_SZ_X, local_size_y = 1, local_size_z = 1) in;
)";
    activation_base += params;
    activation_base += R"(
void main() {
    for (int i = int(gl_GlobalInvocationID.x); i < p.total; i += int(gl_NumWorkGroups.x * gl_WorkGroupSize.x)){
)";
    activation_base += body;
    activation_base += R"(
    }
}
)";
    return activation_base;
}

const char* ACTIVATION_PARAM = R"(
layout(push_constant) uniform pushBlock {
      int total;
      float alpha;
} p;
)";


celu::celu(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "celu";
    m_param.alpha = alpha;
    
    bck_shader = kernel::shaders::celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::celu_spv;
    fwd_codeSize = sizeof(kernel::shaders::celu_spv);
}

std::shared_ptr<tensor>& celu::operator()(const std::shared_ptr<tensor>& x)
{
    alpha = std::make_shared<tensor>(tensor(1.0, x->getShape(), Format::kFormatFp32));
    return layer_construct_forward(x, alpha);
}

elu::elu(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "elu";
    m_param.alpha = alpha;
 
    bck_shader = kernel::shaders::d_elu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_elu_spv);
    auto code = compile(m_type, ACTIVATION_SHADER_BASE(ACTIVATION_PARAM, R"(
        if(X[i] >= 0) Y[i] = X[i];
        else Y[i] = p.alpha * (exp(X[i] - 1));
)"));
    fwd_shader = code.data();
    fwd_codeSize = code.size();
}

std::shared_ptr<tensor>& elu::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}

gelu::gelu(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "gelu";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_elu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_elu_spv);
    fwd_shader = kernel::shaders::gelu_spv;
    fwd_codeSize = sizeof(kernel::shaders::gelu_spv);
}

std::shared_ptr<tensor>& gelu::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}

hardshrink::hardshrink(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "hardshrink";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::hardshrink_spv;
    fwd_codeSize = sizeof(kernel::shaders::hardshrink_spv);
}

std::shared_ptr<tensor>& hardshrink::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}


hardsigmoid::hardsigmoid(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "hardsigmoid";
    m_param.alpha = 0.;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::hardsigmoid_spv;
    fwd_codeSize = sizeof(kernel::shaders::hardsigmoid_spv);
}

std::shared_ptr<tensor>& hardsigmoid::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}

hardswish::hardswish(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "hardswish";
    m_param.alpha = 0.;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::hardswish_spv;
    fwd_codeSize = sizeof(kernel::shaders::hardswish_spv);
}

std::shared_ptr<tensor>& hardswish::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}


hardtanh::hardtanh(float min_val, float max_val, bool in_place) : Base_Layer<two_param>(2, in_place)
{
    m_type = "hardtanh";
    m_param = { 0, min_val, max_val };
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::hardshrink_spv;
    fwd_codeSize = sizeof(kernel::shaders::hardshrink_spv);
}

std::shared_ptr<tensor>& hardtanh::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}

leakyrelu::leakyrelu(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "leakyrelu";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::leakyrelu_spv;
    fwd_codeSize = sizeof(kernel::shaders::leakyrelu_spv);
}

std::shared_ptr<tensor>& leakyrelu::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}

logsigmoid::logsigmoid(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "logsigmoid";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::logsigmoid_spv;
    fwd_codeSize = sizeof(kernel::shaders::logsigmoid_spv);
}

std::shared_ptr<tensor>& logsigmoid::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}


prelu::prelu(float alpha, bool in_place) : Base_Layer<two_param>(2, in_place)
{
    m_type = "prelu";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::relu_spv;
    fwd_codeSize = sizeof(kernel::shaders::relu_spv);
}

std::shared_ptr<tensor>& prelu::operator()(const std::shared_ptr<tensor>& x)
{
    m_param.beta = x->count() / m_param.alpha;

    return layer_construct_forward(x);
}


relu::relu(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "relu";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::relu_spv;
    bck_codeSize = sizeof(kernel::shaders::relu_spv);
}

std::shared_ptr<tensor>& relu::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}


rrelu::rrelu(float lower, float upper, bool in_place) : Base_Layer<activation_param>(3, in_place), min(lower), max(upper)
{
    m_type = "rrelu";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::d_relu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_relu_spv);
    fwd_shader = kernel::shaders::rrelu_spv;
    fwd_codeSize = sizeof(kernel::shaders::rrelu_spv);
}

std::shared_ptr<tensor>& rrelu::operator()(const std::shared_ptr<tensor>& x)
{
    w = std::make_shared<tensor>(tensor(init::uniform_distribution_init(x->getShape(), min, max), x->getShape()));
    return layer_construct_forward(x, w);
}

selu::selu(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "selu";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::selu_spv;
    fwd_codeSize = sizeof(kernel::shaders::selu_spv);
}

std::shared_ptr<tensor>& selu::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}


sigmoid::sigmoid(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "sigmoid";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::d_sigmoid_spv;
    bck_codeSize = sizeof(kernel::shaders::d_sigmoid_spv);
    fwd_shader = kernel::shaders::sigmoid_spv;
    fwd_codeSize = sizeof(kernel::shaders::sigmoid_spv);
}

std::shared_ptr<tensor>& sigmoid::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}


softplus::softplus(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "softplus";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::softplus_spv;
    fwd_codeSize = sizeof(kernel::shaders::softplus_spv);
}

std::shared_ptr<tensor>& softplus::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}


softshrink::softshrink(float alpha, bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "softshrink";
    m_param.alpha = alpha;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::softshrink_spv;
    fwd_codeSize = sizeof(kernel::shaders::softshrink_spv);
}

std::shared_ptr<tensor>& softshrink::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}


softsign::softsign(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "softsign";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::d_celu_spv;
    bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
    fwd_shader = kernel::shaders::softsign_spv;
    fwd_codeSize = sizeof(kernel::shaders::softsign_spv);
}

std::shared_ptr<tensor>& softsign::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
}


tanhshrink::tanhshrink(bool in_place) : Base_Layer<activation_param>(2, in_place)
{
    m_type = "tanhshrink";
    m_param.alpha = 0;
    bck_shader = kernel::shaders::d_tanh_spv;
    bck_codeSize = sizeof(kernel::shaders::d_tanh_spv);
    fwd_shader = kernel::shaders::tanhshrink_spv;
    fwd_codeSize = sizeof(kernel::shaders::tanhshrink_spv);
}

std::shared_ptr<tensor>& tanhshrink::operator()(const std::shared_ptr<tensor>& x)
{
    return layer_construct_forward(x);
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