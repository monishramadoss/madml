#include "common.h"
#include "utils.h"
#include "math.h"
#include <future>

std::string UNARY_SHADER_BASE(const char* params, const char* body)
{
    std::string unary_base = R"(
#Version 450
#define LOCAL_SZ_X 1024
layout(binding = 0) readonly buffer buf1 { float X[]; };
layout(binding = 1) writeonly buffer buf2 { float Y[]; };
layout(local_size_x = LOCAL_SZ_X, local_size_y = 1, local_size_z = 1) in;
)";
    unary_base += params;
    unary_base += R"(
void main() {
    for (int i = int(gl_GlobalInvocationID.x); i < p.total; i += int(gl_NumWorkGroups.x * gl_WorkGroupSize.x)){
)";
    unary_base += body;
    unary_base += R"(
    }
}
)";
    return unary_base;
}


const char* UNARY_PARAM = R"(
layout(push_constant) uniform pushBlock {
      int total;
} p;
)";

std::string BINARY_SHADER_BASE(const char* params, const char* body)
{
    std::string binary_base = R"(
#Version 450
#define LOCAL_SZ_X 1024
layout(binding = 0) readonly buffer buf1 { float X[]; };
layout(binding = 1) readonly buffer buf2 { float W[]; };
layout(binding = 2) writeonly buffer buf3 { float Y[]; };
layout(local_size_x = LOCAL_SZ_X, local_size_y = 1, local_size_z = 1) in;
)";
    binary_base += params;
    binary_base += R"(
void main() {
    for (int i = int(gl_GlobalInvocationID.x); i < p.total / p.batch_size; i += int(gl_NumWorkGroups.x * gl_WorkGroupSize.x)){
        for (int b = 0; b < p.batch_size; ++b){

)";
    binary_base += body;
    binary_base += R"(
        }
    }    
}
)";
    return binary_base;
}

const char* BINARY_PARAM = R"(
layout(push_constant) uniform pushBlock {
      int total;
      int batch_size;
} p;
)";


namespace math
{
    abs::abs(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "abs";
        bck_shader = kernel::shaders::d_abs_spv;
        bck_codeSize = sizeof(kernel::shaders::d_abs_spv);
        fwd_shader = kernel::shaders::abs_spv;
        fwd_codeSize = sizeof(kernel::shaders::abs_spv);
    }

    std::shared_ptr<tensor>& abs::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    ceil::ceil(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "ceil";
        bck_shader = kernel::shaders::unary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::unary_operator_spv);
        fwd_shader = kernel::shaders::ceil_spv;
        fwd_codeSize = sizeof(kernel::shaders::ceil_spv);
    }

    std::shared_ptr<tensor>& ceil::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    clip::clip(float min, float max, bool in_place) : Base_Layer<clip_operator_param>(2, in_place)
    {
        m_type = "clip";
        m_param = { 0, min, max };
        bck_shader = kernel::shaders::unary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::unary_operator_spv);
        fwd_shader = kernel::shaders::clip_spv;
        fwd_codeSize = sizeof(kernel::shaders::clip_spv);
    }

    std::shared_ptr<tensor>& clip::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    exp::exp(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "exp";
        bck_shader = kernel::shaders::d_exp_spv;
        bck_codeSize = sizeof(kernel::shaders::d_exp_spv);
        fwd_shader = kernel::shaders::exp_spv;
        fwd_codeSize = sizeof(kernel::shaders::exp_spv);
    }

    std::shared_ptr<tensor>& exp::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x); 
    }

    floor::floor(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "floor";
        bck_shader = kernel::shaders::unary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::unary_operator_spv);
        fwd_shader = kernel::shaders::floor_spv;
        fwd_codeSize = sizeof(kernel::shaders::floor_spv);
    }

    std::shared_ptr<tensor>& floor::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    ln::ln(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "ln";
        bck_shader = kernel::shaders::d_ln_spv;
        bck_codeSize = sizeof(kernel::shaders::d_ln_spv);
        fwd_shader = kernel::shaders::ln_spv;
        fwd_codeSize = sizeof(kernel::shaders::ln_spv);
    }

    std::shared_ptr<tensor>& ln::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    round::round(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "round";
        bck_shader = kernel::shaders::unary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::unary_operator_spv);
        fwd_shader = kernel::shaders::round_spv;
        fwd_codeSize = sizeof(kernel::shaders::round_spv);
    }

    std::shared_ptr<tensor>& round::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    sqrt::sqrt(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "sqrt";
        bck_shader = kernel::shaders::d_sqrt_spv;
        bck_codeSize = sizeof(kernel::shaders::d_sqrt_spv);
        fwd_shader = kernel::shaders::sqrt_spv;
        fwd_codeSize = sizeof(kernel::shaders::sqrt_spv);
    }

    std::shared_ptr<tensor>& sqrt::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    acos::acos(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "acos";
        bck_shader = kernel::shaders::d_acos_spv;
        bck_codeSize = sizeof(kernel::shaders::d_acos_spv);
        fwd_shader = kernel::shaders::acos_spv;
        fwd_codeSize = sizeof(kernel::shaders::acos_spv);
    }

    std::shared_ptr<tensor>& acos::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    acosh::acosh(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "acosh";
        bck_shader = kernel::shaders::d_acosh_spv;
        bck_codeSize = sizeof(kernel::shaders::d_acosh_spv);
        fwd_shader = kernel::shaders::acosh_spv;
        fwd_codeSize = sizeof(kernel::shaders::acosh_spv);
    }

    std::shared_ptr<tensor>& acosh::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    asin::asin(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "asin";
        bck_shader = kernel::shaders::d_asin_spv;
        bck_codeSize = sizeof(kernel::shaders::d_asin_spv);
        fwd_shader = kernel::shaders::asin_spv;
        fwd_codeSize = sizeof(kernel::shaders::asin_spv);
    }

    std::shared_ptr<tensor>& asin::operator()(const std::shared_ptr<tensor>&)
    {
        return layer_construct_forward(x);
    }

    asinh::asinh(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "asinh";
        bck_shader = kernel::shaders::d_asinh_spv;
        bck_codeSize = sizeof(kernel::shaders::d_asinh_spv);
        fwd_shader = kernel::shaders::asinh_spv;
        fwd_codeSize = sizeof(kernel::shaders::asinh_spv);
    }

    std::shared_ptr<tensor>& asinh::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    atan::atan(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "atan";
        bck_shader = kernel::shaders::d_atan_spv;
        bck_codeSize = sizeof(kernel::shaders::d_atan_spv);
        fwd_shader = kernel::shaders::atan_spv;
        fwd_codeSize = sizeof(kernel::shaders::atan_spv);
    }

    std::shared_ptr<tensor>& atan::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    atanh::atanh(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "atan";
        bck_shader = kernel::shaders::d_atanh_spv;
        bck_codeSize = sizeof(kernel::shaders::d_atanh_spv);
        fwd_shader = kernel::shaders::atanh_spv;
        fwd_codeSize = sizeof(kernel::shaders::atanh_spv);
    }

    std::shared_ptr<tensor>& atanh::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    cos::cos(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "cos";
        bck_shader = kernel::shaders::d_cos_spv;
        bck_codeSize = sizeof(kernel::shaders::d_cos_spv);
        fwd_shader = kernel::shaders::cos_spv;
        fwd_codeSize = sizeof(kernel::shaders::cos_spv);
    }

    std::shared_ptr<tensor>& cos::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    cosh::cosh(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "cosh";
        bck_shader = kernel::shaders::d_cosh_spv;
        bck_codeSize = sizeof(kernel::shaders::d_cosh_spv);
        fwd_shader = kernel::shaders::cosh_spv;
        fwd_codeSize = sizeof(kernel::shaders::cosh_spv);
    }

    std::shared_ptr<tensor>& cosh::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    sin::sin(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "sin";
        bck_shader = kernel::shaders::d_sin_spv;
        bck_codeSize = sizeof(kernel::shaders::d_sin_spv);
        fwd_shader = kernel::shaders::sin_spv;
        fwd_codeSize = sizeof(kernel::shaders::sin_spv);
    }

    std::shared_ptr<tensor>& sin::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    sinh::sinh(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "sinh";
        bck_shader = kernel::shaders::d_sinh_spv;
        bck_codeSize = sizeof(kernel::shaders::d_sinh_spv);
        fwd_shader = kernel::shaders::sinh_spv;
        fwd_codeSize = sizeof(kernel::shaders::sinh_spv);
    }

    std::shared_ptr<tensor>& sinh::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    tan::tan(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "tan";
        bck_shader = kernel::shaders::d_tan_spv;
        bck_codeSize = sizeof(kernel::shaders::d_tan_spv);
        fwd_shader = kernel::shaders::tan_spv;
        fwd_codeSize = sizeof(kernel::shaders::tan_spv);
    }

    std::shared_ptr<tensor>& tan::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    tanh::tanh(bool in_place) : Base_Layer<>(2, in_place)
    {
        m_type = "tanh";
        bck_shader = kernel::shaders::d_tanh_spv;
        bck_codeSize = sizeof(kernel::shaders::d_tanh_spv);
        fwd_shader = kernel::shaders::tanh_spv;
        fwd_codeSize = sizeof(kernel::shaders::tanh_spv);
    }

    std::shared_ptr<tensor>& tanh::operator()(const std::shared_ptr<tensor>& x)
    {
        return layer_construct_forward(x);
    }

    add::add(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "add";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::abs_spv;
        fwd_codeSize = sizeof(kernel::shaders::add_spv);
    }

    std::shared_ptr<tensor>& add::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w);
    }

    sub::sub(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "sub";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::sub_spv;
        fwd_codeSize = sizeof(kernel::shaders::sub_spv);
    }

    std::shared_ptr<tensor>& sub::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w);
    }

    mul::mul(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "mul";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::mul_spv;
        fwd_codeSize = sizeof(kernel::shaders::mul_spv);
    }

    std::shared_ptr<tensor>& mul::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w);
    }

    div::div(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "div";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::div_spv;
        fwd_codeSize = sizeof(kernel::shaders::div_spv);
    }

    std::shared_ptr<tensor>& div::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w);
    }

    mod::mod(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "mod";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::mod_spv;
        fwd_codeSize = sizeof(kernel::shaders::mod_spv);
    }

    std::shared_ptr<tensor>& mod::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w);
    }

    pow::pow(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "pow";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::pow_spv;
        fwd_codeSize = sizeof(kernel::shaders::pow_spv);
    }

    std::shared_ptr<tensor>& pow::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w);
    }

    max::max(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "max";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::max_spv;
        fwd_codeSize = sizeof(kernel::shaders::max_spv);
    }

    std::shared_ptr<tensor>& max::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w);
    }

    min::min(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "min";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::min_spv;
        fwd_codeSize = sizeof(kernel::shaders::min_spv);
    }

    std::shared_ptr<tensor>& min::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w);
    }

    eq::eq(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "eq";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::equal_spv;
        fwd_codeSize = sizeof(kernel::shaders::equal_spv);
    }

    std::shared_ptr<tensor>& eq::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w, Format::kFormatBool);
    }

    ne::ne(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "ne";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::nequal_spv;
        fwd_codeSize = sizeof(kernel::shaders::nequal_spv);
    }

    std::shared_ptr<tensor>& ne::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w, Format::kFormatBool);
    }

    lt::lt(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "lt";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::less_than_spv;
        fwd_codeSize = sizeof(kernel::shaders::less_than_spv);
    }

    std::shared_ptr<tensor>& lt::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w, Format::kFormatBool);
    }

    le::le(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "le";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::less_eq_spv;
        fwd_codeSize = sizeof(kernel::shaders::less_eq_spv);
    }

    std::shared_ptr<tensor>& le::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w, Format::kFormatBool);
    }

    gt::gt(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "gt";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::greater_than_spv;
        fwd_codeSize = sizeof(kernel::shaders::greater_than_spv);
    }

    std::shared_ptr<tensor>& gt::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w, Format::kFormatBool);
    }

    ge::ge(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "ge";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::greater_eq_spv;
        fwd_codeSize = sizeof(kernel::shaders::greater_eq_spv);
    }

    std::shared_ptr<tensor>& ge::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        return layer_construct_forward(x, w, Format::kFormatBool);
    }

    xr::xr(bool in_place) : Base_Layer<>(3, in_place)
    {
        m_type = "xor";
        bck_shader = kernel::shaders::binary_operator_spv;
        bck_codeSize = sizeof(kernel::shaders::binary_operator_spv);
        fwd_shader = kernel::shaders::xor_spv;
        fwd_codeSize = sizeof(kernel::shaders::xor_spv);
    }

    std::shared_ptr<tensor>& xr::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
    {
        if (x->getFormat() != Format::kFormatBool && w->getFormat() != Format::kFormatBool)
        {
            std::cerr << "XOR KERNEL REQUIRES BOTH INPUTS BE BOOLEAN VALUES" << std::endl;
        }
        return layer_construct_forward(x, w, Format::kFormatBool);
    }
}

void init_abs(py::module& m)
{
    py::class_<math::abs>(m, "abs")
        .def(py::init<bool&>());
}

void init_ceil(py::module& m)
{
    py::class_<math::ceil>(m, "ceil")
        .def(py::init<bool&>());
}

void init_clip(py::module& m)
{
    py::class_<math::clip>(m, "clip")
        .def(py::init<bool&>());
}

void init_exp(py::module& m)
{
    py::class_<math::exp>(m, "exp")
        .def(py::init<bool&>());
}

void init_floor(py::module& m)
{
    py::class_<math::floor>(m, "floor")
        .def(py::init<bool&>());
}

void init_ln(py::module& m)
{
    py::class_<math::ln>(m, "ln")
        .def(py::init<bool&>());
}

void init_round(py::module& m)
{
    py::class_<math::round>(m, "round")
        .def(py::init<bool&>());
}

void init_sqrt(py::module& m)
{
    py::class_<math::sqrt>(m, "sqrt")
        .def(py::init<bool&>());
}

void init_acos(py::module& m)
{
    py::class_<math::acos>(m, "acos")
        .def(py::init<bool&>());
}

void init_acosh(py::module& m)
{
    py::class_<math::acosh>(m, "acosh")
        .def(py::init<bool&>());
}

void init_asin(py::module& m)
{
    py::class_<math::asin>(m, "asin")
        .def(py::init<bool&>());
}

void init_asinh(py::module& m)
{
    py::class_<math::asinh>(m, "asinh")
        .def(py::init<bool&>());
}

void init_atan(py::module& m)
{
    py::class_<math::atan>(m, "atan")
        .def(py::init<bool&>());
}

void init_atanh(py::module& m)
{
    py::class_<math::atanh>(m, "atanh")
        .def(py::init<bool&>());
}

void init_cos(py::module& m)
{
    py::class_<math::cos>(m, "cos")
        .def(py::init<bool&>());
}

void init_cosh(py::module& m)
{
    py::class_<math::cosh>(m, "cosh")
        .def(py::init<bool&>());
}

void init_sin(py::module& m)
{
    py::class_<math::sin>(m, "sin")
        .def(py::init<bool&>());
}

void init_sinh(py::module& m)
{
    py::class_<math::sinh>(m, "sinh")
        .def(py::init<bool&>());
}

void init_tan(py::module& m)
{
    py::class_<math::tan>(m, "tan")
        .def(py::init<bool&>());
}

void init_tanh(py::module& m)
{
    py::class_<math::tanh>(m, "tanh")
        .def(py::init<bool&>());
}

void init_add(py::module& m)
{
    py::class_<math::add>(m, "add")
        .def(py::init<bool&>());
}

void init_sub(py::module& m)
{
    py::class_<math::sub>(m, "sub")
        .def(py::init<bool&>());
}

void init_mul(py::module& m)
{
    py::class_<math::mul>(m, "mul")
        .def(py::init<bool&>());
}

void init_div(py::module& m)
{
    py::class_<math::div>(m, "div")
        .def(py::init<bool&>());
}

void init_mod(py::module& m)
{
    py::class_<math::mod>(m, "mod")
        .def(py::init<bool&>());
}

void init_pow(py::module& m)
{
    py::class_<math::pow>(m, "pow")
        .def(py::init<bool&>());
}

void init_min(py::module& m)
{
    py::class_<math::min>(m, "min")
        .def(py::init<bool&>());
}

void init_max(py::module& m)
{
    py::class_<math::max>(m, "max")
        .def(py::init<bool&>());
}

void init_eq(py::module& m)
{
    py::class_<math::eq>(m, "eq")
        .def(py::init<bool&>());
}

void init_ne(py::module& m)
{
    py::class_<math::ne>(m, "ne")
        .def(py::init<bool&>());
}

void init_lt(py::module& m)
{
    py::class_<math::lt>(m, "lt")
        .def(py::init<bool&>());
}

void init_le(py::module& m)
{
    py::class_<math::le>(m, "le")
        .def(py::init<bool&>());
}

void init_gt(py::module& m)
{
    py::class_<math::gt>(m, "gt")
        .def(py::init<bool&>());
}

void init_ge(py::module& m)
{
    py::class_<math::ge>(m, "ge")
        .def(py::init<bool&>());
}