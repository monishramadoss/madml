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
				return layer_construct_forward(shaders::add_spv, sizeof(shaders::add_spv), x, w, this->m_param);;
			}

			void add::set_derivative()
			{
				f_prime = new derivative::math::add();
				dx = f_prime->hook(dy, dw);
			}

			sub::sub(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::sub();
				m_type = "sub";
			}

			std::shared_ptr<tensor>& sub::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::sub_spv, sizeof(shaders::sub_spv), x, w, this->m_param);
			}

			mul::mul(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::mul();
				m_type = "mul";
			}

			std::shared_ptr<tensor>& mul::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::mul_spv, sizeof(shaders::mul_spv), x, w, this->m_param);
			}

			div::div(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::div();
				m_type = "div";
			}

			std::shared_ptr<tensor>& div::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::div_spv, sizeof(shaders::div_spv), x, w, this->m_param);
			}

			mod::mod(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::mod();
				m_type = "mod";
			}

			std::shared_ptr<tensor>& mod::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::mod_spv, sizeof(shaders::mod_spv), x, w, this->m_param);
			}

			pow::pow(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::pow();
				m_type = "pow";
			}

			std::shared_ptr<tensor>& pow::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::pow_spv, sizeof(shaders::pow_spv), x, w, this->m_param);
			}

			max::max(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::max();
				m_type = "max";
			}

			std::shared_ptr<tensor>& max::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::max_spv, sizeof(shaders::max_spv), x, w, this->m_param);
			}

			min::min(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::min();
				m_type = "min";
			}

			std::shared_ptr<tensor>& min::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::min_spv, sizeof(shaders::min_spv), x, w, this->m_param);
			}

			eq::eq(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::eq();
				m_type = "eq";
			}

			std::shared_ptr<tensor>& eq::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::equal_spv, sizeof(shaders::equal_spv), x, w, this->m_param, Format::kFormatBool);
			}

			ne::ne(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::ne();
				m_type = "ne";
			}

			std::shared_ptr<tensor>& ne::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::nequal_spv, sizeof(shaders::nequal_spv), x, w, this->m_param, Format::kFormatBool);
			}

			lt::lt(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::lt();
				m_type = "lt";
			}

			std::shared_ptr<tensor>& lt::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::less_than_spv, sizeof(shaders::less_than_spv), x, w, this->m_param, Format::kFormatBool);
			}

			le::le(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::le();
				m_type = "le";
			}

			std::shared_ptr<tensor>& le::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::less_eq_spv, sizeof(shaders::less_eq_spv), x, w, this->m_param, Format::kFormatBool);
			}

			gt::gt(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::gt();
				m_type = "gt";
			}

			std::shared_ptr<tensor>& gt::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::greater_than_spv, sizeof(shaders::greater_than_spv), x, w, this->m_param, Format::kFormatBool);
			}

			ge::ge(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::ge();
				m_type = "ge";
			}

			std::shared_ptr<tensor>& ge::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::greater_eq_spv, sizeof(shaders::greater_eq_spv), x, w, this->m_param, Format::kFormatBool);
			}

			xr::xr(bool in_place) : binary_operator(in_place)
			{
				f_prime = new derivative::math::xr();
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