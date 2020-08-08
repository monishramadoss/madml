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
			abs::abs(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "abs";
				bck_shader = shaders::d_abs_spv;
				bck_codeSize = sizeof(shaders::d_abs_spv);
			}

			std::shared_ptr<tensor>& abs::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::abs_spv, sizeof(shaders::abs_spv), x);
			}

			ceil::ceil(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "ceil";
				bck_shader = shaders::unary_operator_spv;
				bck_codeSize = sizeof(shaders::unary_operator_spv);
			}

			std::shared_ptr<tensor>& ceil::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::ceil_spv, sizeof(shaders::ceil_spv), x);
			}

			clip::clip(float min, float max, bool in_place) : Base_Layer<clip_operator_param>(2, in_place)
			{
				m_type = "clip";
				m_param = { 0, min, max };
				bck_shader = shaders::unary_operator_spv;
				bck_codeSize = sizeof(shaders::unary_operator_spv);
			}

			std::shared_ptr<tensor>& clip::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::clip_spv, sizeof(shaders::clip_spv), x);
			}

			exp::exp(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "exp";
				bck_shader = shaders::d_exp_spv;
				bck_codeSize = sizeof(shaders::d_exp_spv);
			}

			std::shared_ptr<tensor>& exp::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::exp_spv, sizeof(shaders::exp_spv), x);
			}

			floor::floor(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "floor";
				bck_shader = shaders::unary_operator_spv;
				bck_codeSize = sizeof(shaders::unary_operator_spv);
			}

			std::shared_ptr<tensor>& floor::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::floor_spv, sizeof(shaders::floor_spv), x);
			}

			ln::ln(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "ln";
				bck_shader = shaders::d_ln_spv;
				bck_codeSize = sizeof(shaders::d_ln_spv);
			}

			std::shared_ptr<tensor>& ln::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::ln_spv, sizeof(shaders::ln_spv), x);
			}

			round::round(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "round";
				bck_shader = shaders::unary_operator_spv;
				bck_codeSize = sizeof(shaders::unary_operator_spv);
			}

			std::shared_ptr<tensor>& round::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::round_spv, sizeof(shaders::round_spv), x);
			}

			sqrt::sqrt(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "sqrt";
				bck_shader = shaders::d_sqrt_spv;
				bck_codeSize = sizeof(shaders::d_sqrt_spv);
			}

			std::shared_ptr<tensor>& sqrt::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::sqrt_spv, sizeof(shaders::sqrt_spv), x);
			}

			acos::acos(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "acos";
				bck_shader = shaders::d_acos_spv;
				bck_codeSize = sizeof(shaders::d_acos_spv);
			}

			std::shared_ptr<tensor>& acos::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::acos_spv, sizeof(shaders::acos_spv), x);
			}

			acosh::acosh(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "acosh";
				bck_shader = shaders::d_acosh_spv;
				bck_codeSize = sizeof(shaders::d_acosh_spv);
			}

			std::shared_ptr<tensor>& acosh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::acosh_spv, sizeof(shaders::acosh_spv), x);
			}

			asin::asin(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "asin";
				bck_shader = shaders::d_asin_spv;
				bck_codeSize = sizeof(shaders::d_asin_spv);
			}

			std::shared_ptr<tensor>& asin::hook(const std::shared_ptr<tensor>&)
			{
				return layer_construct_forward(shaders::asin_spv, sizeof(shaders::asin_spv), x);
			}

			asinh::asinh(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "asinh";
				bck_shader = shaders::d_asinh_spv;
				bck_codeSize = sizeof(shaders::d_asinh_spv);
			}

			std::shared_ptr<tensor>& asinh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::asinh_spv, sizeof(shaders::asinh_spv), x);
			}

			atan::atan(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "atan";
				bck_shader = shaders::d_atan_spv;
				bck_codeSize = sizeof(shaders::d_atan_spv);
			}

			std::shared_ptr<tensor>& atan::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::atan_spv, sizeof(shaders::atan_spv), x);
			}

			atanh::atanh(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "atan";
				bck_shader = shaders::d_atanh_spv;
				bck_codeSize = sizeof(shaders::d_atanh_spv);
			}

			std::shared_ptr<tensor>& atanh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::atanh_spv, sizeof(shaders::atanh_spv), x);
			}

			cos::cos(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "cos";
				bck_shader = shaders::d_cos_spv;
				bck_codeSize = sizeof(shaders::d_cos_spv);
			}

			std::shared_ptr<tensor>& cos::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::cos_spv, sizeof(shaders::cos_spv), x);
			}

			cosh::cosh(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "cosh";
				bck_shader = shaders::d_cosh_spv;
				bck_codeSize = sizeof(shaders::d_cosh_spv);
			}

			std::shared_ptr<tensor>& cosh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::cosh_spv, sizeof(shaders::cosh_spv), x);
			}

			sin::sin(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "sin";
				bck_shader = shaders::d_sin_spv;
				bck_codeSize = sizeof(shaders::d_sin_spv);
			}

			std::shared_ptr<tensor>& sin::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::sin_spv, sizeof(shaders::sin_spv), x);
			}

			sinh::sinh(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "sinh";
				bck_shader = shaders::d_sinh_spv;
				bck_codeSize = sizeof(shaders::d_sinh_spv);
			}

			std::shared_ptr<tensor>& sinh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::sinh_spv, sizeof(shaders::sinh_spv), x);
			}

			tan::tan(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "tan";
				bck_shader = shaders::d_tan_spv;
				bck_codeSize = sizeof(shaders::d_tan_spv);
			}

			std::shared_ptr<tensor>& tan::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::tan_spv, sizeof(shaders::tan_spv), x);
			}

			tanh::tanh(bool in_place) : Base_Layer<>(2, in_place)
			{
				m_type = "tanh";
				bck_shader = shaders::d_tanh_spv;
				bck_codeSize = sizeof(shaders::d_tanh_spv);
			}

			std::shared_ptr<tensor>& tanh::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::tanh_spv, sizeof(shaders::tanh_spv), x);
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
			add::add(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "add";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& add::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::add_spv, sizeof(shaders::add_spv), x, w);
			}

			sub::sub(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "sub";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& sub::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::sub_spv, sizeof(shaders::sub_spv), x, w);
			}

			mul::mul(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "mul";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& mul::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::mul_spv, sizeof(shaders::mul_spv), x, w);
			}

			div::div(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "div";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& div::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::div_spv, sizeof(shaders::div_spv), x, w);
			}

			mod::mod(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "mod";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& mod::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::mod_spv, sizeof(shaders::mod_spv), x, w);
			}

			pow::pow(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "pow";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& pow::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::pow_spv, sizeof(shaders::pow_spv), x, w);
			}

			max::max(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "max";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& max::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::max_spv, sizeof(shaders::max_spv), x, w);
			}

			min::min(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "min";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& min::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::min_spv, sizeof(shaders::min_spv), x, w);
			}

			eq::eq(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "eq";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& eq::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::equal_spv, sizeof(shaders::equal_spv), x, w, Format::kFormatBool);
			}

			ne::ne(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "ne";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& ne::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::nequal_spv, sizeof(shaders::nequal_spv), x, w, Format::kFormatBool);
			}

			lt::lt(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "lt";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& lt::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::less_than_spv, sizeof(shaders::less_than_spv), x, w, Format::kFormatBool);
			}

			le::le(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "le";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& le::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::less_eq_spv, sizeof(shaders::less_eq_spv), x, w, Format::kFormatBool);
			}

			gt::gt(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "gt";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& gt::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::greater_than_spv, sizeof(shaders::greater_than_spv), x, w, Format::kFormatBool);
			}

			ge::ge(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "ge";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& ge::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				return layer_construct_forward(shaders::greater_eq_spv, sizeof(shaders::greater_eq_spv), x, w, Format::kFormatBool);
			}

			xr::xr(bool in_place) : Base_Layer<>(3, in_place)
			{
				m_type = "xor";
				bck_shader = shaders::binary_operator_spv;
				bck_codeSize = sizeof(shaders::binary_operator_spv);
			}

			std::shared_ptr<tensor>& xr::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
			{
				if (x->getFormat() != Format::kFormatBool && w->getFormat() != Format::kFormatBool)
				{
					std::cerr << "XOR KERNEL REQUIRES BOTH INPUTS BE BOOLEAN VALUES" << std::endl;
				}
				return layer_construct_forward(shaders::xor_spv, sizeof(shaders::xor_spv), x, w, Format::kFormatBool);
			}
		}
	}
}