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
			unary_operator::unary_operator(bool in_place, bool as_module) : as_module(as_module), m_inplace(in_place),
			                                                                m_param({0})
			{
				initVulkanThing(2);
			}

			void unary_operator::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.total, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
					m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
				m_group_y = 1;
				m_group_z = 1;
			}

			tensor* unary_operator::layer_construct_forward(const uint32_t* shader, size_t codeSize, tensor* x)
			{
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape());
				
				if (m_pipeline_forward == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModuleForward(shader, codeSize);
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, y, 1, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(operator_param));
				
				inputs.push_back(x->getId());
				outputs.push_back(y->getId());
				layers.push_back(this);
				if (as_module)
					add_module(this);
				return y;
			}

			void unary_operator::layer_construct_backward(const uint32_t* shader, size_t codeSize)
			{
				tensor* x = get_grad(inputs[0]);
				tensor* y = get_grad(outputs[0]);
				
				if (m_pipeline_forward == nullptr)
				{
					m_param = { x->count() };
					computeGroupCount();
					createShaderModuleForward(shader, codeSize);
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, y, 0, m_descriptor_set_backward);
				bindTensor(m_device, x, 1, m_descriptor_set_backward);

				recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(operator_param));
		
			}
						
			void unary_operator::update_weight()
			{
			}

			binary_operator::binary_operator(bool in_place, bool as_module) : as_module(as_module), m_inplace(in_place),
			                                                                  m_param({0})
			{
				initVulkanThing(6);
			}

			void binary_operator::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.total, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
					m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
				m_group_y = 1;
				m_group_z = 1;
			}

			tensor* binary_operator::layer_construct_forward(const uint32_t* shader, size_t codeSize, tensor* x, tensor* w)
			{				
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape());
			
				if (m_pipeline_forward == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModuleForward(shader, codeSize);
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, w, 1, m_descriptor_set_forward);
				bindTensor(m_device, y, 2, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(operator_param));

				inputs.push_back(x->getId());
				inputs.push_back(w->getId());
				outputs.push_back(y->getId());
				layers.push_back(this);
				if (as_module)
					add_module(this);
				return y;
			}

			void binary_operator::layer_construct_backward(const uint32_t* shader, size_t codeSize)
			{
				tensor* x = get_grad(inputs[0]);
				tensor* w = get_grad(inputs[1]);
				tensor* y = get_grad(outputs[0]);
						
				if (m_pipeline_forward == nullptr)
				{
					m_param = { x->count() };
					computeGroupCount();
					createShaderModuleForward(shader, codeSize);
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, y, 0, m_descriptor_set_backward);
				bindTensor(m_device, w, 1, m_descriptor_set_backward);
				bindTensor(m_device, x, 2, m_descriptor_set_backward);

				recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(operator_param));			
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
				return layer_construct_forward(shaders::abs_spv, sizeof(shaders::abs_spv), x);
			}

			void abs::back_propagate() 
			{
				layer_construct_backward(shaders::d_abs_spv, sizeof(shaders::d_abs_spv));
			}

			ceil::ceil(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "ceil";
			}

			tensor* ceil::forward(tensor* x)
			{
				return layer_construct_forward(shaders::ceil_spv, sizeof(shaders::ceil_spv), x);
			}

			void ceil::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			clip::clip(float min, float max, bool in_place, bool as_module) : unary_operator(in_place, as_module), m_min(min),
			                                                                  m_max(max)
			{
				m_type = "clip";
			}

			tensor* clip::forward(tensor* x)
			{
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape());
				
				if (m_pipeline_forward == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModuleForward(shaders::clip_spv, sizeof(shaders::clip_spv));
					createPipelineForward(sizeof(clip_operator_param));
				}

				clip_operator_param param = {m_param.total, m_min, m_max};
				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, y, 1, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&param), sizeof(clip_operator_param));

				inputs.push_back(x->getId());
				outputs.push_back(y->getId());
				layers.push_back(this);
				return y;
			}

			void clip::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			exp::exp(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "exp";
			}

			tensor* exp::forward(tensor* x)
			{
				return layer_construct_forward(shaders::exp_spv, sizeof(shaders::exp_spv), x);
			}

			void exp::back_propagate() 
			{
				layer_construct_backward(shaders::d_exp_spv, sizeof(shaders::d_exp_spv));
			}

			floor::floor(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "floor";
			}

			tensor* floor::forward(tensor* x)
			{
				return layer_construct_forward(shaders::floor_spv, sizeof(shaders::floor_spv), x);
			}

			void floor::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			ln::ln(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "ln";
			}

			tensor* ln::forward(tensor* x)
			{
				return layer_construct_forward(shaders::ln_spv, sizeof(shaders::ln_spv), x);
			}

			void ln::back_propagate() 
			{
				layer_construct_backward(shaders::d_ln_spv, sizeof(shaders::d_ln_spv));
			}

			round::round(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "round";
			}

			tensor* round::forward(tensor* x)
			{
				return layer_construct_forward(shaders::round_spv, sizeof(shaders::round_spv), x);
			}

			void round::back_propagate() 
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			sqrt::sqrt(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "sqrt";
			}
		
			tensor* sqrt::forward(tensor* x)
			{
				return layer_construct_forward(shaders::sqrt_spv, sizeof(shaders::sqrt_spv), x);
			}

			void sqrt::back_propagate() 
			{
				layer_construct_backward(shaders::d_sqrt_spv, sizeof(shaders::d_sqrt_spv));
			}

			acos::acos(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "acos";
			}

			tensor* acos::forward(tensor* x)
			{
				return layer_construct_forward(shaders::acos_spv, sizeof(shaders::acos_spv), x);
			}

			void acos::back_propagate() {
				layer_construct_backward(shaders::d_acos_spv, sizeof(shaders::d_acos_spv));
			}

			acosh::acosh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "acosh";
			}

			tensor* acosh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::acosh_spv, sizeof(shaders::acosh_spv), x);
			}

			void acosh::back_propagate() {
				layer_construct_backward(shaders::d_acosh_spv, sizeof(shaders::d_acosh_spv));
			}

			asin::asin(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "asin";
			}

			tensor* asin::forward(tensor* x)
			{
				return layer_construct_forward(shaders::asin_spv, sizeof(shaders::asin_spv), x);
			}

			void asin::back_propagate() 
			{
				layer_construct_backward(shaders::d_asin_spv, sizeof(shaders::d_asin_spv));
			}

			asinh::asinh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "asinh";
			}

			tensor* asinh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::asinh_spv, sizeof(shaders::asinh_spv), x);
			}

			void asinh::back_propagate() 
			{
				layer_construct_backward(shaders::d_asinh_spv, sizeof(shaders::d_asinh_spv));
			}

			atan::atan(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "atan";
			}

			tensor* atan::forward(tensor* x)
			{
				return layer_construct_forward(shaders::atan_spv, sizeof(shaders::atan_spv), x);
			}

			void atan::back_propagate() {
				layer_construct_backward(shaders::d_atan_spv, sizeof(shaders::d_atan_spv));
			}

			atanh::atanh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "atan";
			}

			tensor* atanh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::atanh_spv, sizeof(shaders::atanh_spv), x);
			}

			void atanh::back_propagate() {
				layer_construct_backward(shaders::d_atanh_spv, sizeof(shaders::d_atanh_spv));
			}

			cos::cos(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "cos";
			}

			tensor* cos::forward(tensor* x)
			{
				return layer_construct_forward(shaders::cos_spv, sizeof(shaders::cos_spv), x);
			}

			void cos::back_propagate() {
				layer_construct_backward(shaders::d_cos_spv, sizeof(shaders::d_cos_spv));
			}

			cosh::cosh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "cosh";
			}

			tensor* cosh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::cosh_spv, sizeof(shaders::cosh_spv), x);
			}

			void cosh::back_propagate() {
				layer_construct_backward(shaders::d_cosh_spv, sizeof(shaders::d_cosh_spv));
			}

			sin::sin(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "sin";
			}

			tensor* sin::forward(tensor* x)
			{
				return layer_construct_forward(shaders::sin_spv, sizeof(shaders::sin_spv), x);
			}

			void sin::back_propagate() {
				layer_construct_backward(shaders::d_sin_spv, sizeof(shaders::d_sin_spv));
			}

			sinh::sinh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "sinh";
			}

			tensor* sinh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::sinh_spv, sizeof(shaders::sinh_spv), x);
			}

			void sinh::back_propagate() {
				layer_construct_backward(shaders::d_sinh_spv, sizeof(shaders::d_sinh_spv));
			}

			tan::tan(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "tan";
			}

			tensor* tan::forward(tensor* x)
			{
				return layer_construct_forward(shaders::tan_spv, sizeof(shaders::tan_spv), x);
			}

			void tan::back_propagate() {
				layer_construct_backward(shaders::d_tan_spv, sizeof(shaders::d_tan_spv));
			}

			tanh::tanh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "tanh";
			}

			tensor* tanh::forward(tensor* x)
			{
				return layer_construct_forward(shaders::tanh_spv, sizeof(shaders::tanh_spv), x);
			}

			void tanh::back_propagate() {
				layer_construct_backward(shaders::d_tanh_spv, sizeof(shaders::d_tanh_spv));
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
				return layer_construct_forward(shaders::add_spv, sizeof(shaders::add_spv), x, w);
			}

			void add::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
			}

			sub::sub(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "sub";
			}

			tensor* sub::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::sub_spv, sizeof(shaders::sub_spv), x, w);
			}

			void sub::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
			}

			mul::mul(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "mul";
			}

			tensor* mul::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::mul_spv, sizeof(shaders::mul_spv), x, w);
			}

			void mul::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
			}


			div::div(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "div";
			}

			tensor* div::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::div_spv, sizeof(shaders::div_spv), x, w);
			}

			void div::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
			}

			mod::mod(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "mod";
			}

			tensor* mod::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::mod_spv, sizeof(shaders::mod_spv), x, w);
			}

			void mod::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
			}

			pow::pow(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "pow";
			}

			tensor* pow::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::pow_spv, sizeof(shaders::pow_spv), x, w);
			}

			void pow::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
			}

			max::max(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "max";
			}

			tensor* max::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::max_spv, sizeof(shaders::max_spv), x, w);
			}

			void max::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
			}


			min::min(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "min";
			}

			void min::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
			}

			tensor* min::forward(tensor* x, tensor* w)
			{
				return layer_construct_forward(shaders::min_spv, sizeof(shaders::min_spv), x, w);
			}

			void min::back_propagate()
			{
				layer_construct_backward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
			}

			eq::eq(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "eq";
			}

			tensor* eq::forward(tensor* x, tensor* w)
			{
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
			
				if (m_pipeline_forward == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModuleForward(shaders::equal_spv, sizeof(shaders::equal_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, w, 1, m_descriptor_set_forward);
				bindTensor(m_device, y, 2, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(operator_param));

				inputs.push_back(x->getId());
				inputs.push_back(w->getId());
				outputs.push_back(y->getId());
				layers.push_back(this);
				return y;
			}

			void eq::back_propagate() {
				tensor* x = get_grad(inputs[0]);
				tensor* w = get_grad(inputs[1]);
				tensor* y = get_grad(outputs[0]);

				if (m_pipeline_forward == nullptr)
				{
					m_param = { x->count() };
					computeGroupCount();
					createShaderModuleForward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 2, m_descriptor_set_backward);
				bindTensor(m_device, w, 1, m_descriptor_set_backward);
				bindTensor(m_device, y, 0, m_descriptor_set_backward);

				recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(operator_param));
			}

			ne::ne(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "ne";
			}

			tensor* ne::forward(tensor* x, tensor* w)
			{
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
				
				if (m_pipeline_forward == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModuleForward(shaders::nequal_spv, sizeof(shaders::nequal_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, w, 1, m_descriptor_set_forward);
				bindTensor(m_device, y, 2, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(operator_param));

				inputs.push_back(x->getId());
				inputs.push_back(w->getId());
				outputs.push_back(y->getId());
				layers.push_back(this);
				return y;
			}

			void ne::back_propagate() {
				tensor* x = get_grad(inputs[0]);
				tensor* w = get_grad(inputs[1]);
				tensor* y = get_grad(outputs[0]);

				if (m_pipeline_forward == nullptr)
				{
					m_param = { x->count() };
					computeGroupCount();
					createShaderModuleForward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 2, m_descriptor_set_backward);
				bindTensor(m_device, w, 1, m_descriptor_set_backward);
				bindTensor(m_device, y, 0, m_descriptor_set_backward);

				recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(operator_param));
			}

			lt::lt(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "lt";
			}

			tensor* lt::forward(tensor* x, tensor* w)
			{				
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
				
				if (m_pipeline_forward == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModuleForward(shaders::less_than_spv, sizeof(shaders::less_than_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, w, 1, m_descriptor_set_forward);
				bindTensor(m_device, y, 2, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(operator_param));

				inputs.push_back(x->getId());
				inputs.push_back(w->getId());
				outputs.push_back(y->getId());
				layers.push_back(this);
				return y;
			}

			void lt::back_propagate() {
				tensor* x = get_grad(inputs[0]);
				tensor* w = get_grad(inputs[1]);
				tensor* y = get_grad(outputs[0]);

				if (m_pipeline_forward == nullptr)
				{
					m_param = { x->count() };
					computeGroupCount();
					createShaderModuleForward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 2, m_descriptor_set_backward);
				bindTensor(m_device, w, 1, m_descriptor_set_backward);
				bindTensor(m_device, y, 0, m_descriptor_set_backward);

				recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(operator_param));
			}

			le::le(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "le";
			}

			tensor* le::forward(tensor* x, tensor* w)
			{
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
			
				if (m_pipeline_forward == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModuleForward(shaders::less_eq_spv, sizeof(shaders::less_eq_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, w, 1, m_descriptor_set_forward);
				bindTensor(m_device, y, 2, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(operator_param));

				inputs.push_back(x->getId());
				inputs.push_back(w->getId());
				outputs.push_back(y->getId());
				layers.push_back(this);
				return y;
			}

			void le::back_propagate() {
				tensor* x = get_grad(inputs[0]);
				tensor* w = get_grad(inputs[1]);
				tensor* y = get_grad(outputs[0]);

				if (m_pipeline_forward == nullptr)
				{
					m_param = { x->count() };
					computeGroupCount();
					createShaderModuleForward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 2, m_descriptor_set_backward);
				bindTensor(m_device, w, 1, m_descriptor_set_backward);
				bindTensor(m_device, y, 0, m_descriptor_set_backward);

				recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(operator_param));
			}

			gt::gt(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "gt";
			}

			tensor* gt::forward(tensor* x, tensor* w)
			{
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
				
				if (m_pipeline_forward == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModuleForward(shaders::greater_than_spv, sizeof(shaders::greater_than_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, w, 1, m_descriptor_set_forward);
				bindTensor(m_device, y, 2, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(operator_param));

				inputs.push_back(x->getId());
				inputs.push_back(w->getId());
				outputs.push_back(y->getId());
				layers.push_back(this);
				return y;
			}


			void gt::back_propagate() {
				tensor* x = get_grad(inputs[0]);
				tensor* w = get_grad(inputs[1]);
				tensor* y = get_grad(outputs[0]);

				if (m_pipeline_forward == nullptr)
				{
					m_param = { x->count() };
					computeGroupCount();
					createShaderModuleForward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 2, m_descriptor_set_backward);
				bindTensor(m_device, w, 1, m_descriptor_set_backward);
				bindTensor(m_device, y, 0, m_descriptor_set_backward);

				recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(operator_param));
			}

			ge::ge(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "greater_eq";
			}

			tensor* ge::forward(tensor* x, tensor* w)
			{
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
			
				if (m_pipeline_forward == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModuleForward(shaders::greater_eq_spv, sizeof(shaders::greater_eq_spv));
					createPipelineForward(sizeof(operator_param));
				}
				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, w, 1, m_descriptor_set_forward);
				bindTensor(m_device, y, 2, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(operator_param));

				inputs.push_back(x->getId());
				inputs.push_back(w->getId());
				outputs.push_back(y->getId());
				layers.push_back(this);
				return y;
			}


			void ge::back_propagate() {
				tensor* x = get_grad(inputs[0]);
				tensor* w = get_grad(inputs[1]);
				tensor* y = get_grad(outputs[0]);

				if (m_pipeline_forward == nullptr)
				{
					m_param = { x->count() };
					computeGroupCount();
					createShaderModuleForward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 2, m_descriptor_set_backward);
				bindTensor(m_device, w, 1, m_descriptor_set_backward);
				bindTensor(m_device, y, 0, m_descriptor_set_backward);

				recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(operator_param));
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

				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);

				if (m_pipeline_forward == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModuleForward(shaders::xor_spv, sizeof(shaders::xor_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, w, 1, m_descriptor_set_forward);
				bindTensor(m_device, y, 2, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(operator_param));

				inputs.push_back(x->getId());
				inputs.push_back(w->getId());
				outputs.push_back(y->getId());
				layers.push_back(this);
				return y;
			}

			void xr::back_propagate() {
				tensor* x = get_grad(inputs[0]);
				tensor* w = get_grad(inputs[1]);
				tensor* y = get_grad(outputs[0]);

				if (m_pipeline_forward == nullptr)
				{
					m_param = { x->count() };
					computeGroupCount();
					createShaderModuleForward(shaders::binary_operator_spv, sizeof(shaders::binary_operator_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 2, m_descriptor_set_backward);
				bindTensor(m_device, w, 1, m_descriptor_set_backward);
				bindTensor(m_device, y, 0, m_descriptor_set_backward);

				recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(operator_param));
			}
		}
	}
}
