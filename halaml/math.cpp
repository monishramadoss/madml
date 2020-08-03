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
			abs::abs(bool in_place) : unary_operator(in_place)
			{
				m_type = "abs";
			}

			std::shared_ptr<tensor>& abs::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::abs_spv, sizeof(shaders::abs_spv), x, this->m_param);
			}

			ceil::ceil(bool in_place) : unary_operator(in_place)
			{
				m_type = "ceil";
			}

			std::shared_ptr<tensor>& ceil::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::ceil_spv, sizeof(shaders::ceil_spv), x, this->m_param);
			}

			clip::clip(float min, float max, bool in_place) : unary_operator(in_place), m_param({ 0, min, max })
			{
				m_type = "clip";
			}

			std::shared_ptr<tensor>& clip::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward<clip_operator_param>(shaders::clip_spv, sizeof(shaders::clip_spv), x, this->m_param);
			}

			exp::exp(bool in_place) : unary_operator(in_place)
			{
				m_type = "exp";
			}

			std::shared_ptr<tensor>& exp::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::exp_spv, sizeof(shaders::exp_spv), x, this->m_param);
			}

			floor::floor(bool in_place) : unary_operator(in_place)
			{
				m_type = "floor";
			}

			std::shared_ptr<tensor>& floor::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::floor_spv, sizeof(shaders::floor_spv), x, this->m_param);
			}

			ln::ln(bool in_place) : unary_operator(in_place)
			{
				m_type = "ln";
			}

			std::shared_ptr<tensor>& ln::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::ln_spv, sizeof(shaders::ln_spv), x, this->m_param);
			}

			round::round(bool in_place) : unary_operator(in_place)
			{
				m_type = "round";
			}

			std::shared_ptr<tensor>& round::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::round_spv, sizeof(shaders::round_spv), x, this->m_param);
			}

			sqrt::sqrt(bool in_place) : unary_operator(in_place)
			{
				m_type = "sqrt";
			}

			std::shared_ptr<tensor>& sqrt::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::sqrt_spv, sizeof(shaders::sqrt_spv), x, this->m_param);
			}

			acos::acos(bool in_place) : unary_operator(in_place)
			{
				m_type = "acos";
			}

			std::shared_ptr<tensor>& acos::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::acos_spv, sizeof(shaders::acos_spv), x, this->m_param);
			}

			acosh::acosh(bool in_place) : unary_operator(in_place)
			{
				m_type = "acosh";
			}

			std::shared_ptr<tensor>& acosh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::acosh_spv, sizeof(shaders::acosh_spv), x, this->m_param);
			}

			asin::asin(bool in_place) : unary_operator(in_place)
			{
				m_type = "asin";
			}

			std::shared_ptr<tensor>& asin::hook(const std::shared_ptr<tensor>&)
			{
				return layer_construct_forward(shaders::asin_spv, sizeof(shaders::asin_spv), x, this->m_param);
			}

			asinh::asinh(bool in_place) : unary_operator(in_place)
			{
				m_type = "asinh";
			}

			std::shared_ptr<tensor>& asinh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::asinh_spv, sizeof(shaders::asinh_spv), x, this->m_param);
			}

			atan::atan(bool in_place) : unary_operator(in_place)
			{
				m_type = "atan";
			}

			std::shared_ptr<tensor>& atan::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::atan_spv, sizeof(shaders::atan_spv), x, this->m_param);
			}

			atanh::atanh(bool in_place) : unary_operator(in_place)
			{
				m_type = "atan";
			}

			std::shared_ptr<tensor>& atanh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::atanh_spv, sizeof(shaders::atanh_spv), x, this->m_param);
			}

			cos::cos(bool in_place) : unary_operator(in_place)
			{
				m_type = "cos";
			}

			std::shared_ptr<tensor>& cos::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::cos_spv, sizeof(shaders::cos_spv), x, this->m_param);
			}

			cosh::cosh(bool in_place) : unary_operator(in_place)
			{
				m_type = "cosh";
			}

			std::shared_ptr<tensor>& cosh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::cosh_spv, sizeof(shaders::cosh_spv), x, this->m_param);
			}

			sin::sin(bool in_place) : unary_operator(in_place)
			{
				m_type = "sin";
			}

			std::shared_ptr<tensor>& sin::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::sin_spv, sizeof(shaders::sin_spv), x, this->m_param);
			}

			sinh::sinh(bool in_place) : unary_operator(in_place)
			{
				m_type = "sinh";
			}

			std::shared_ptr<tensor>& sinh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::sinh_spv, sizeof(shaders::sinh_spv), x, this->m_param);
			}

			tan::tan(bool in_place) : unary_operator(in_place)
			{
				m_type = "tan";
			}

			std::shared_ptr<tensor>& tan::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::tan_spv, sizeof(shaders::tan_spv), x, this->m_param);
			}

			tanh::tanh(bool in_place) : unary_operator(in_place)
			{
				m_type = "tanh";
			}

			std::shared_ptr<tensor>& tanh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::tanh_spv, sizeof(shaders::tanh_spv), x, this->m_param);
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

			std::shared_ptr<tensor>& add::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::add_spv, sizeof(shaders::add_spv), x, w, this->m_param);
			}

			sub::sub(bool in_place) : binary_operator(in_place)
			{
				m_type = "sub";
			}

			std::shared_ptr<tensor>& sub::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::sub_spv, sizeof(shaders::sub_spv), x, w, this->m_param);
			}

			mul::mul(bool in_place) : binary_operator(in_place)
			{
				m_type = "mul";
			}

			std::shared_ptr<tensor>& mul::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::mul_spv, sizeof(shaders::mul_spv), x, w, this->m_param);
			}

			div::div(bool in_place) : binary_operator(in_place)
			{
				m_type = "div";
			}

			std::shared_ptr<tensor>& div::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::div_spv, sizeof(shaders::div_spv), x, w, this->m_param);
			}

			mod::mod(bool in_place) : binary_operator(in_place)
			{
				m_type = "mod";
			}

			std::shared_ptr<tensor>& mod::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::mod_spv, sizeof(shaders::mod_spv), x, w, this->m_param);
			}

			pow::pow(bool in_place) : binary_operator(in_place)
			{
				m_type = "pow";
			}

			std::shared_ptr<tensor>& pow::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::pow_spv, sizeof(shaders::pow_spv), x, w, this->m_param);
			}

			max::max(bool in_place) : binary_operator(in_place)
			{
				m_type = "max";
			}

			std::shared_ptr<tensor>& max::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::max_spv, sizeof(shaders::max_spv), x, w, this->m_param);
			}

			min::min(bool in_place) : binary_operator(in_place)
			{
				m_type = "min";
			}

			std::shared_ptr<tensor>& min::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::min_spv, sizeof(shaders::min_spv), x, w, this->m_param);
			}

			eq::eq(bool in_place) : binary_operator(in_place)
			{
				m_type = "eq";
			}

			std::shared_ptr<tensor>& eq::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::equal_spv, sizeof(shaders::equal_spv), x, w, this->m_param, Format::kFormatBool);
			}

			ne::ne(bool in_place) : binary_operator(in_place)
			{
				m_type = "ne";
			}

			std::shared_ptr<tensor>& ne::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::nequal_spv, sizeof(shaders::nequal_spv), x, w, this->m_param, Format::kFormatBool);
			}

			lt::lt(bool in_place) : binary_operator(in_place)
			{
				m_type = "lt";
			}

			std::shared_ptr<tensor>& lt::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::less_than_spv, sizeof(shaders::less_than_spv), x, w, this->m_param, Format::kFormatBool);
			}

			le::le(bool in_place) : binary_operator(in_place)
			{
				m_type = "le";
			}

			std::shared_ptr<tensor>& le::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::less_eq_spv, sizeof(shaders::less_eq_spv), x, w, this->m_param, Format::kFormatBool);
			}

			gt::gt(bool in_place) : binary_operator(in_place)
			{
				m_type = "gt";
			}

			std::shared_ptr<tensor>& gt::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::greater_than_spv, sizeof(shaders::greater_than_spv), x, w, this->m_param, Format::kFormatBool);
			}

			ge::ge(bool in_place) : binary_operator(in_place)
			{
				m_type = "greater_eq";
			}

			std::shared_ptr<tensor>& ge::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::greater_eq_spv, sizeof(shaders::greater_eq_spv), x, w, this->m_param, Format::kFormatBool);
			}

			xr::xr(bool in_place) : binary_operator(in_place)
			{
				m_type = "xor";
			}

			std::shared_ptr<tensor>& xr::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				if (x->getFormat() != Format::kFormatBool && w->getFormat() != Format::kFormatBool)
				{
					std::cerr << "XOR KERNEL REQUIRES BOTH INPUTS BE BOOLEAN VALUES" << std::endl;
				}
				return layer_construct_forward(shaders::xor_spv, sizeof(shaders::xor_spv), x, w, this->m_param, Format::kFormatBool);
			}
		}
	}
}

namespace kernel
{
	namespace layers
	{
		namespace derivative
		{
			namespace math
			{
				abs::abs(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_abs";
				}

				std::shared_ptr<tensor>& abs::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_abs_spv, sizeof(shaders::d_abs_spv), x, this->m_param);
				}

				ceil::ceil(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_ceil";
				}

				std::shared_ptr<tensor>& ceil::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::ceil_spv, sizeof(shaders::ceil_spv), x, this->m_param);
				}

				clip::clip(float min, float max, bool in_place) : unary_operator(in_place), m_param({ 0, min, max })
				{
					m_type = "d_clip";
				}

				std::shared_ptr<tensor>& clip::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward<clip_operator_param>(shaders::clip_spv, sizeof(shaders::clip_spv), x, this->m_param);
				}

				exp::exp(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_exp";
				}

				std::shared_ptr<tensor>& exp::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_exp_spv, sizeof(shaders::d_exp_spv), x, this->m_param);
				}

				floor::floor(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_floor";
				}

				std::shared_ptr<tensor>& floor::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::floor_spv, sizeof(shaders::floor_spv), x, this->m_param);
				}

				ln::ln(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_ln";
				}

				std::shared_ptr<tensor>& ln::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_ln_spv, sizeof(shaders::d_ln_spv), x, this->m_param);
				}

				round::round(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_round";
				}

				std::shared_ptr<tensor>& round::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::round_spv, sizeof(shaders::round_spv), x, this->m_param);
				}

				sqrt::sqrt(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_sqrt";
				}

				std::shared_ptr<tensor>& sqrt::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_sqrt_spv, sizeof(shaders::d_sqrt_spv), x, this->m_param);
				}

				acos::acos(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_acos";
				}

				std::shared_ptr<tensor>& acos::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_acos_spv, sizeof(shaders::d_acos_spv), x, this->m_param);
				}

				acosh::acosh(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_acosh";
				}

				std::shared_ptr<tensor>& acosh::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_acosh_spv, sizeof(shaders::d_acosh_spv), x, this->m_param);
				}

				asin::asin(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_asin";
				}

				std::shared_ptr<tensor>& asin::hook(const std::shared_ptr<tensor>&)
				{
					return layer_construct_forward(shaders::d_asin_spv, sizeof(shaders::d_asin_spv), x, this->m_param);
				}

				asinh::asinh(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_asinh";
				}

				std::shared_ptr<tensor>& asinh::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_asinh_spv, sizeof(shaders::d_asinh_spv), x, this->m_param);
				}

				atan::atan(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_atan";
				}

				std::shared_ptr<tensor>& atan::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_atan_spv, sizeof(shaders::d_atan_spv), x, this->m_param);
				}

				atanh::atanh(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_atanh";
				}

				std::shared_ptr<tensor>& atanh::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_atanh_spv, sizeof(shaders::d_atanh_spv), x, this->m_param);
				}

				cos::cos(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_cos";
				}

				std::shared_ptr<tensor>& cos::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_cos_spv, sizeof(shaders::d_cos_spv), x, this->m_param);
				}

				cosh::cosh(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_cosh";
				}

				std::shared_ptr<tensor>& cosh::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_cosh_spv, sizeof(shaders::d_cosh_spv), x, this->m_param);
				}

				sin::sin(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_sin";
				}

				std::shared_ptr<tensor>& sin::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_sin_spv, sizeof(shaders::d_sin_spv), x, this->m_param);
				}

				sinh::sinh(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_sinh";
				}

				std::shared_ptr<tensor>& sinh::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_sinh_spv, sizeof(shaders::d_sinh_spv), x, this->m_param);
				}

				tan::tan(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_tan";
				}

				std::shared_ptr<tensor>& tan::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_tan_spv, sizeof(shaders::d_tan_spv), x, this->m_param);
				}

				tanh::tanh(bool in_place) : unary_operator(in_place)
				{
					m_type = "d_tanh";
				}

				std::shared_ptr<tensor>& tanh::hook(const std::shared_ptr<tensor>& x)
				{
					return layer_construct_forward(shaders::d_tanh_spv, sizeof(shaders::d_tanh_spv), x, this->m_param);
				}
			}
		}
	}
}

namespace kernel
{
	namespace layers
	{
		namespace derivative
		{
			namespace math
			{
				add::add(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_add";
				}

				std::shared_ptr<tensor>& add::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), x, w, this->m_param);
				}

				sub::sub(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_sub";
				}

				std::shared_ptr<tensor>& sub::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), x, w, this->m_param);
				}

				mul::mul(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_mul";
				}

				std::shared_ptr<tensor>& mul::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::mul_spv, sizeof(shaders::mul_spv), x, w, this->m_param);
				}

				div::div(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_div";
				}

				std::shared_ptr<tensor>& div::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::div_spv, sizeof(shaders::div_spv), x, w, this->m_param);
				}

				mod::mod(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_mod";
				}

				std::shared_ptr<tensor>& mod::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::mod_spv, sizeof(shaders::mod_spv), x, w, this->m_param);
				}

				pow::pow(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_pow";
				}

				std::shared_ptr<tensor>& pow::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::pow_spv, sizeof(shaders::pow_spv), x, w, this->m_param);
				}

				max::max(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_max";
				}

				std::shared_ptr<tensor>& max::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::max_spv, sizeof(shaders::max_spv), x, w, this->m_param);
				}

				min::min(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_min";
				}

				std::shared_ptr<tensor>& min::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::min_spv, sizeof(shaders::min_spv), x, w, this->m_param);
				}

				eq::eq(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_eq";
				}

				std::shared_ptr<tensor>& eq::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::equal_spv, sizeof(shaders::equal_spv), x, w, this->m_param, Format::kFormatBool);
				}

				ne::ne(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_ne";
				}

				std::shared_ptr<tensor>& ne::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::nequal_spv, sizeof(shaders::nequal_spv), x, w, this->m_param, Format::kFormatBool);
				}

				lt::lt(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_lt";
				}

				std::shared_ptr<tensor>& lt::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::less_than_spv, sizeof(shaders::less_than_spv), x, w, this->m_param, Format::kFormatBool);
				}

				le::le(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_le";
				}

				std::shared_ptr<tensor>& le::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::less_eq_spv, sizeof(shaders::less_eq_spv), x, w, this->m_param, Format::kFormatBool);
				}

				gt::gt(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_gt";
				}

				std::shared_ptr<tensor>& gt::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::greater_than_spv, sizeof(shaders::greater_than_spv), x, w, this->m_param, Format::kFormatBool);
				}

				ge::ge(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_ge";
				}

				std::shared_ptr<tensor>& ge::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					return layer_construct_forward(shaders::greater_eq_spv, sizeof(shaders::greater_eq_spv), x, w, this->m_param, Format::kFormatBool);
				}

				xr::xr(bool in_place) : binary_operator(in_place)
				{
					m_type = "d_xor";
				}

				std::shared_ptr<tensor>& xr::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
				{
					if (x->getFormat() != Format::kFormatBool && w->getFormat() != Format::kFormatBool)
					{
						std::cerr << "XOR KERNEL REQUIRES BOTH INPUTS BE BOOLEAN VALUES" << std::endl;
					}
					return layer_construct_forward(shaders::xor_spv, sizeof(shaders::xor_spv), x, w, this->m_param, Format::kFormatBool);
				}
			}
		}
	}
}