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
			unary_operator::unary_operator(bool in_place, bool as_module) : Base_Layer(2, -1, in_place, as_module)
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
							
			void unary_operator::update_weight()
			{
			}

			binary_operator::binary_operator(bool in_place, bool as_module) : Base_Layer(3, -1, in_place, as_module)
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

			
			void binary_operator::update_weight()
			{
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
			abs::abs(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "abs";
			}

			tensor* abs::forward(tensor* x)
			{
				return layer_construct_forward(shaders::abs_spv, sizeof(shaders::abs_spv), x, m_param);
			}

			void abs::back_propagate() 
			{
				layer_construct_backward(shaders::d_abs_spv, sizeof(shaders::d_abs_spv), m_param);
			}

			ceil::ceil(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "ceil";
			}

			tensor* ceil::forward(tensor* x)
			{
				return layer_construct_forward(shaders::ceil_spv, sizeof(shaders::ceil_spv), x, m_param);
			}

			void ceil::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			clip::clip(float min, float max, bool in_place, bool as_module) : unary_operator(in_place, as_module), m_param({0, min, max})
			{
				m_type = "clip";
			}

			tensor* clip::forward(tensor* x)
			{
				return layer_construct_forward<clip_operator_param>(shaders::clip_spv, sizeof(shaders::clip_spv), x, m_param);
			}

			void clip::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			exp::exp(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "exp";
			}

			tensor* exp::forward(tensor* x)
			{
				return layer_construct_forward(shaders::exp_spv, sizeof(shaders::exp_spv), x, m_param);
			}

			void exp::back_propagate() 
			{
				layer_construct_backward(shaders::d_exp_spv, sizeof(shaders::d_exp_spv), m_param);
			}

			floor::floor(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "floor";
			}

			tensor* floor::forward(tensor* x)
			{
				return layer_construct_forward(shaders::floor_spv, sizeof(shaders::floor_spv), x, m_param);
			}

			void floor::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			ln::ln(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "ln";
			}

			tensor* ln::forward(tensor* x)
			{
				return layer_construct_forward(shaders::ln_spv, sizeof(shaders::ln_spv), x, m_param);
			}

			void ln::back_propagate() 
			{
				layer_construct_backward(shaders::d_ln_spv, sizeof(shaders::d_ln_spv), m_param);
			}

			round::round(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "round";
			}

			tensor* round::forward(tensor* x)
			{
				return layer_construct_forward(shaders::round_spv, sizeof(shaders::round_spv), x, m_param);
			}

			void round::back_propagate() 
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			sqrt::sqrt(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "sqrt";
			}
		
			tensor* sqrt::forward(tensor* x)
			{
				return layer_construct_forward(shaders::sqrt_spv, sizeof(shaders::sqrt_spv), x, m_param);
			}

			void sqrt::back_propagate() 
			{
				layer_construct_backward(shaders::d_sqrt_spv, sizeof(shaders::d_sqrt_spv), m_param);
			}

			acos::acos(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "acos";
			}

			tensor* acos::forward(tensor* x)
			{
				return layer_construct_forward(shaders::acos_spv, sizeof(shaders::acos_spv), x, m_param);
			}

			void acos::back_propagate() {
				layer_construct_backward(shaders::d_acos_spv, sizeof(shaders::d_acos_spv), m_param);
			}

			acosh::acosh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "acosh";
			}

			tensor* acosh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::acosh_spv, sizeof(shaders::acosh_spv), x, m_param);
			}

			void acosh::back_propagate() {
				layer_construct_backward(shaders::d_acosh_spv, sizeof(shaders::d_acosh_spv), m_param);
			}

			asin::asin(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "asin";
			}

			tensor* asin::forward(tensor* x)
			{
				return layer_construct_forward(shaders::asin_spv, sizeof(shaders::asin_spv), x, m_param);
			}

			void asin::back_propagate() 
			{
				layer_construct_backward(shaders::d_asin_spv, sizeof(shaders::d_asin_spv), m_param);
			}

			asinh::asinh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "asinh";
			}

			tensor* asinh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::asinh_spv, sizeof(shaders::asinh_spv), x, m_param);
			}

			void asinh::back_propagate() 
			{
				layer_construct_backward(shaders::d_asinh_spv, sizeof(shaders::d_asinh_spv), m_param);
			}

			atan::atan(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "atan";
			}

			tensor* atan::forward(tensor* x)
			{
				return layer_construct_forward(shaders::atan_spv, sizeof(shaders::atan_spv), x, m_param);
			}

			void atan::back_propagate() {
				layer_construct_backward(shaders::d_atan_spv, sizeof(shaders::d_atan_spv), m_param);
			}

			atanh::atanh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "atan";
			}

			tensor* atanh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::atanh_spv, sizeof(shaders::atanh_spv), x, m_param);
			}

			void atanh::back_propagate() {
				layer_construct_backward(shaders::d_atanh_spv, sizeof(shaders::d_atanh_spv), m_param);
			}

			cos::cos(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "cos";
			}

			tensor* cos::forward(tensor* x)
			{
				return layer_construct_forward(shaders::cos_spv, sizeof(shaders::cos_spv), x, m_param);
			}

			void cos::back_propagate() {
				layer_construct_backward(shaders::d_cos_spv, sizeof(shaders::d_cos_spv), m_param);
			}

			cosh::cosh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "cosh";
			}

			tensor* cosh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::cosh_spv, sizeof(shaders::cosh_spv), x, m_param);
			}

			void cosh::back_propagate() {
				layer_construct_backward(shaders::d_cosh_spv, sizeof(shaders::d_cosh_spv), m_param);
			}

			sin::sin(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "sin";
			}

			tensor* sin::forward(tensor* x)
			{
				return layer_construct_forward(shaders::sin_spv, sizeof(shaders::sin_spv), x, m_param);
			}

			void sin::back_propagate() {
				layer_construct_backward(shaders::d_sin_spv, sizeof(shaders::d_sin_spv), m_param);
			}

			sinh::sinh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "sinh";
			}

			tensor* sinh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::sinh_spv, sizeof(shaders::sinh_spv), x, m_param);
			}

			void sinh::back_propagate() {
				layer_construct_backward(shaders::d_sinh_spv, sizeof(shaders::d_sinh_spv), m_param);
			}

			tan::tan(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "tan";
			}

			tensor* tan::forward(tensor* x)
			{
				return layer_construct_forward(shaders::tan_spv, sizeof(shaders::tan_spv), x, m_param);
			}

			void tan::back_propagate() {
				layer_construct_backward(shaders::d_tan_spv, sizeof(shaders::d_tan_spv), m_param);
			}

			tanh::tanh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "tanh";
			}

			tensor* tanh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::tanh_spv, sizeof(shaders::tanh_spv), x, m_param);
			}

			void tanh::back_propagate() {
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
			add::add(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "add";
			}

			tensor* add::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::add_spv, sizeof(shaders::add_spv), x, w, m_param);
			}

			void add::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			sub::sub(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "sub";
			}

			tensor* sub::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::sub_spv, sizeof(shaders::sub_spv), x, w, m_param);
			}

			void sub::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			mul::mul(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "mul";
			}

			tensor* mul::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::mul_spv, sizeof(shaders::mul_spv), x, w, m_param);
			}

			void mul::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}


			div::div(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "div";
			}

			tensor* div::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::div_spv, sizeof(shaders::div_spv), x, w, m_param);
			}

			void div::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			mod::mod(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "mod";
			}

			tensor* mod::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::mod_spv, sizeof(shaders::mod_spv), x, w, m_param);
			}

			void mod::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			pow::pow(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "pow";
			}

			tensor* pow::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::pow_spv, sizeof(shaders::pow_spv), x, w, m_param);
			}

			void pow::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			max::max(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "max";
			}

			tensor* max::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::max_spv, sizeof(shaders::max_spv), x, w, m_param);
			}

			void max::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}


			min::min(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "min";
			}

			tensor* min::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::min_spv, sizeof(shaders::min_spv), x, w, m_param);
			}

			void min::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			eq::eq(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "eq";
			}

			tensor* eq::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::equal_spv, sizeof(shaders::equal_spv), x, w, m_param, kFormatBool);
			}

			void eq::back_propagate() 
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			ne::ne(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "ne";
			}

			tensor* ne::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::nequal_spv, sizeof(shaders::nequal_spv), x, w, m_param, kFormatBool);
			}

			void ne::back_propagate() 
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			lt::lt(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "lt";
			}

			tensor* lt::forward(tensor* x, tensor* w)
			{				
				return layer_construct_forward(shaders::less_than_spv, sizeof(shaders::less_than_spv), x, w, m_param, kFormatBool);
			}

			void lt::back_propagate() 
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			le::le(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "le";
			}

			tensor* le::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::less_eq_spv, sizeof(shaders::less_eq_spv), x, w, m_param, kFormatBool);
			}

			void le::back_propagate() 
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			gt::gt(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "gt";
			}

			tensor* gt::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::greater_than_spv, sizeof(shaders::greater_than_spv), x, w, m_param, kFormatBool);
			}


			void gt::back_propagate() 
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			ge::ge(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "greater_eq";
			}

			tensor* ge::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::greater_eq_spv, sizeof(shaders::greater_eq_spv), x, w, m_param, kFormatBool);
			}


			void ge::back_propagate() 
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}

			xr::xr(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "xor";
			}

			tensor* xr::forward(tensor* x, tensor* w)
			{
				if (x->getFormat() != Format::kFormatBool && w->getFormat() != Format::kFormatBool)
				{
					std::cerr << "XOR KERNEL REQUIRES BOTH INPUTS BE BOOLEAN VALUES" << std::endl;
					return nullptr;
				}

				return layer_construct_forward(shaders::xor_spv, sizeof(shaders::xor_spv), x, w, m_param, kFormatBool);
				
			}

			void xr::back_propagate() {
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv), m_param);
			}
		}
	}
}
