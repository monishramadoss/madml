#include "common.h"
#include "utils.h"
#include "math.h"

#define LOCAL_SZ_X 1024
#define MAX_COMPUTE_WORK_GROUP_COUNT 65535

// https://stats.stackexchange.com/questions/268820/gradient-backpropagation-through-resnet-skip-connections
namespace kernel
{
	namespace layers
	{
		namespace math
		{
			unary_operator::unary_operator(bool in_place) : Base_Layer(2, -1, in_place)
			{
			}

			void unary_operator::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.total, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
					m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
				m_group_y = 1;
				m_group_z = 1;
			}

			binary_operator::binary_operator(bool in_place) : Base_Layer(3, -1, in_place)
			{
			}

			void binary_operator::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.total, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
					m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
				m_group_y = 1;
				m_group_z = 1;
			}
		}
	}
}

namespace kernel
{
	namespace layers
	{
		namespace math
		{
			abs::abs(bool in_place) : unary_operator(in_place)
			{
				m_type = "abs";
			}

			std::shared_ptr<tensor>abs::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::abs_spv, sizeof(shaders::abs_spv), x, m_param);
			}

			void abs::backward()
			{
				layer_construct_backward(shaders::d_abs_spv, sizeof(shaders::d_abs_spv), m_param);
			}

			ceil::ceil(bool in_place) : unary_operator(in_place)
			{
				m_type = "ceil";
			}

			std::shared_ptr<tensor>ceil::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::ceil_spv, sizeof(shaders::ceil_spv), x, m_param);
			}

			void ceil::backward()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			clip::clip(float min, float max, bool in_place) : unary_operator(in_place), m_param({ 0, min, max })
			{
				m_type = "clip";
			}

			std::shared_ptr<tensor>clip::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<clip_operator_param>(shaders::clip_spv, sizeof(shaders::clip_spv), x, m_param);
			}

			void clip::backward()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			exp::exp(bool in_place) : unary_operator(in_place)
			{
				m_type = "exp";
			}

			std::shared_ptr<tensor>exp::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::exp_spv, sizeof(shaders::exp_spv), x, m_param);
			}

			void exp::backward()
			{
				layer_construct_backward(shaders::d_exp_spv, sizeof(shaders::d_exp_spv), m_param);
			}

			floor::floor(bool in_place) : unary_operator(in_place)
			{
				m_type = "floor";
			}

			std::shared_ptr<tensor>floor::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::floor_spv, sizeof(shaders::floor_spv), x, m_param);
			}

			void floor::backward()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			ln::ln(bool in_place) : unary_operator(in_place)
			{
				m_type = "ln";
			}

			std::shared_ptr<tensor>ln::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::ln_spv, sizeof(shaders::ln_spv), x, m_param);
			}

			void ln::backward()
			{
				layer_construct_backward(shaders::d_ln_spv, sizeof(shaders::d_ln_spv), m_param);
			}

			round::round(bool in_place) : unary_operator(in_place)
			{
				m_type = "round";
			}

			std::shared_ptr<tensor>round::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::round_spv, sizeof(shaders::round_spv), x, m_param);
			}

			void round::backward()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			sqrt::sqrt(bool in_place) : unary_operator(in_place)
			{
				m_type = "sqrt";
			}

			std::shared_ptr<tensor>sqrt::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::sqrt_spv, sizeof(shaders::sqrt_spv), x, m_param);
			}

			void sqrt::backward()
			{
				layer_construct_backward(shaders::d_sqrt_spv, sizeof(shaders::d_sqrt_spv), m_param);
			}

			acos::acos(bool in_place) : unary_operator(in_place)
			{
				m_type = "acos";
			}

			std::shared_ptr<tensor>acos::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::acos_spv, sizeof(shaders::acos_spv), x, m_param);
			}

			void acos::backward()
			{
				layer_construct_backward(shaders::d_acos_spv, sizeof(shaders::d_acos_spv), m_param);
			}

			acosh::acosh(bool in_place) : unary_operator(in_place)
			{
				m_type = "acosh";
			}

			std::shared_ptr<tensor>acosh::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::acosh_spv, sizeof(shaders::acosh_spv), x, m_param);
			}

			void acosh::backward()
			{
				layer_construct_backward(shaders::d_acosh_spv, sizeof(shaders::d_acosh_spv), m_param);
			}

			asin::asin(bool in_place) : unary_operator(in_place)
			{
				m_type = "asin";
			}

			std::shared_ptr<tensor>asin::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::asin_spv, sizeof(shaders::asin_spv), x, m_param);
			}

			void asin::backward()
			{
				layer_construct_backward(shaders::d_asin_spv, sizeof(shaders::d_asin_spv), m_param);
			}

			asinh::asinh(bool in_place) : unary_operator(in_place)
			{
				m_type = "asinh";
			}

			std::shared_ptr<tensor>asinh::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::asinh_spv, sizeof(shaders::asinh_spv), x, m_param);
			}

			void asinh::backward()
			{
				layer_construct_backward(shaders::d_asinh_spv, sizeof(shaders::d_asinh_spv), m_param);
			}

			atan::atan(bool in_place) : unary_operator(in_place)
			{
				m_type = "atan";
			}

			std::shared_ptr<tensor>atan::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::atan_spv, sizeof(shaders::atan_spv), x, m_param);
			}

			void atan::backward()
			{
				layer_construct_backward(shaders::d_atan_spv, sizeof(shaders::d_atan_spv), m_param);
			}

			atanh::atanh(bool in_place) : unary_operator(in_place)
			{
				m_type = "atan";
			}

			std::shared_ptr<tensor>atanh::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::atanh_spv, sizeof(shaders::atanh_spv), x, m_param);
			}

			void atanh::backward()
			{
				layer_construct_backward(shaders::d_atanh_spv, sizeof(shaders::d_atanh_spv), m_param);
			}

			cos::cos(bool in_place) : unary_operator(in_place)
			{
				m_type = "cos";
			}

			std::shared_ptr<tensor>cos::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::cos_spv, sizeof(shaders::cos_spv), x, m_param);
			}

			void cos::backward()
			{
				layer_construct_backward(shaders::d_cos_spv, sizeof(shaders::d_cos_spv), m_param);
			}

			cosh::cosh(bool in_place) : unary_operator(in_place)
			{
				m_type = "cosh";
			}

			std::shared_ptr<tensor>cosh::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::cosh_spv, sizeof(shaders::cosh_spv), x, m_param);
			}

			void cosh::backward()
			{
				layer_construct_backward(shaders::d_cosh_spv, sizeof(shaders::d_cosh_spv), m_param);
			}

			sin::sin(bool in_place) : unary_operator(in_place)
			{
				m_type = "sin";
			}

			std::shared_ptr<tensor>sin::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::sin_spv, sizeof(shaders::sin_spv), x, m_param);
			}

			void sin::backward()
			{
				layer_construct_backward(shaders::d_sin_spv, sizeof(shaders::d_sin_spv), m_param);
			}

			sinh::sinh(bool in_place) : unary_operator(in_place)
			{
				m_type = "sinh";
			}

			std::shared_ptr<tensor>sinh::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::sinh_spv, sizeof(shaders::sinh_spv), x, m_param);
			}

			void sinh::backward()
			{
				layer_construct_backward(shaders::d_sinh_spv, sizeof(shaders::d_sinh_spv), m_param);
			}

			tan::tan(bool in_place) : unary_operator(in_place)
			{
				m_type = "tan";
			}

			std::shared_ptr<tensor>tan::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::tan_spv, sizeof(shaders::tan_spv), x, m_param);
			}

			void tan::backward()
			{
				layer_construct_backward(shaders::d_tan_spv, sizeof(shaders::d_tan_spv), m_param);
			}

			tanh::tanh(bool in_place) : unary_operator(in_place)
			{
				m_type = "tanh";
			}

			std::shared_ptr<tensor>tanh::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward(shaders::tanh_spv, sizeof(shaders::tanh_spv), x, m_param);
			}

			void tanh::backward()
			{
				layer_construct_backward(shaders::d_tanh_spv, sizeof(shaders::d_tanh_spv), m_param);
			}
		}
	}
}

namespace kernel
{
	namespace layers
	{
		namespace math
		{
			add::add(bool in_place) : binary_operator(in_place)
			{
				m_type = "add";
			}

			std::shared_ptr<tensor>add::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::add_spv, sizeof(shaders::add_spv), x, w, m_param);
			}

			void add::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			sub::sub(bool in_place) : binary_operator(in_place)
			{
				m_type = "sub";
			}

			std::shared_ptr<tensor>sub::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::sub_spv, sizeof(shaders::sub_spv), x, w, m_param);
			}

			void sub::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			mul::mul(bool in_place) : binary_operator(in_place)
			{
				m_type = "mul";
			}

			std::shared_ptr<tensor>mul::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::mul_spv, sizeof(shaders::mul_spv), x, w, m_param);
			}

			void mul::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			div::div(bool in_place) : binary_operator(in_place)
			{
				m_type = "div";
			}

			std::shared_ptr<tensor>div::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::div_spv, sizeof(shaders::div_spv), x, w, m_param);
			}

			void div::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			mod::mod(bool in_place) : binary_operator(in_place)
			{
				m_type = "mod";
			}

			std::shared_ptr<tensor>mod::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::mod_spv, sizeof(shaders::mod_spv), x, w, m_param);
			}

			void mod::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			pow::pow(bool in_place) : binary_operator(in_place)
			{
				m_type = "pow";
			}

			std::shared_ptr<tensor>pow::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::pow_spv, sizeof(shaders::pow_spv), x, w, m_param);
			}

			void pow::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			max::max(bool in_place) : binary_operator(in_place)
			{
				m_type = "max";
			}

			std::shared_ptr<tensor>max::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::max_spv, sizeof(shaders::max_spv), x, w, m_param);
			}

			void max::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			min::min(bool in_place) : binary_operator(in_place)
			{
				m_type = "min";
			}

			std::shared_ptr<tensor>min::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::min_spv, sizeof(shaders::min_spv), x, w, m_param);
			}

			void min::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			eq::eq(bool in_place) : binary_operator(in_place)
			{
				m_type = "eq";
			}

			std::shared_ptr<tensor>eq::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::equal_spv, sizeof(shaders::equal_spv), x, w, m_param, Format::kFormatBool);
			}

			void eq::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			ne::ne(bool in_place) : binary_operator(in_place)
			{
				m_type = "ne";
			}

			std::shared_ptr<tensor>ne::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::nequal_spv, sizeof(shaders::nequal_spv), x, w, m_param, Format::kFormatBool);
			}

			void ne::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			lt::lt(bool in_place) : binary_operator(in_place)
			{
				m_type = "lt";
			}

			std::shared_ptr<tensor>lt::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::less_than_spv, sizeof(shaders::less_than_spv), x, w, m_param, Format::kFormatBool);
			}

			void lt::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			le::le(bool in_place) : binary_operator(in_place)
			{
				m_type = "le";
			}

			std::shared_ptr<tensor>le::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::less_eq_spv, sizeof(shaders::less_eq_spv), x, w, m_param, Format::kFormatBool);
			}

			void le::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			gt::gt(bool in_place) : binary_operator(in_place)
			{
				m_type = "gt";
			}

			std::shared_ptr<tensor>gt::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::greater_than_spv, sizeof(shaders::greater_than_spv), x, w, m_param, Format::kFormatBool);
			}

			void gt::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			ge::ge(bool in_place) : binary_operator(in_place)
			{
				m_type = "greater_eq";
			}

			std::shared_ptr<tensor>ge::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				return layer_construct_forward(shaders::greater_eq_spv, sizeof(shaders::greater_eq_spv), x, w, m_param, Format::kFormatBool);
			}

			void ge::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			xr::xr(bool in_place) : binary_operator(in_place)
			{
				m_type = "xor";
			}

			std::shared_ptr<tensor>xr::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>w)
			{
				if (x->getFormat() != Format::kFormatBool && w->getFormat() != Format::kFormatBool)
				{
					std::cerr << "XOR KERNEL REQUIRES BOTH INPUTS BE BOOLEAN VALUES" << std::endl;
					return nullptr;
				}

				return layer_construct_forward(shaders::xor_spv, sizeof(shaders::xor_spv), x, w, m_param, Format::kFormatBool);
			}

			void xr::backward()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}
		}
	}
}