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

			tensor* unary_operator::layer_construct(const uint32_t* shader, size_t codeSize, tensor* x)
			{
				m_input.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape());
				m_output.push_back(x->getId());

				if (m_pipeline == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModule(shader, codeSize);
					createPipeline(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, y, 1, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(operator_param));
				layers.push_back(this);
				if (as_module)
					add_module(this);
				return y;
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

			tensor* binary_operator::layer_construct(const uint32_t* shader, size_t codeSize, tensor* x, tensor* w)
			{
				m_input.push_back(x->getId());
				m_input.push_back(w->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape());
				m_output.push_back(y->getId());

				if (m_pipeline == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModule(shader, codeSize);
					createPipeline(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, w, 1, m_descriptor_set);
				bindTensor(m_device, y, 2, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(operator_param));
				layers.push_back(this);
				if (as_module)
					add_module(this);
				return y;
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
				return layer_construct(shaders::abs_spv, sizeof(shaders::abs_spv), x);
			}

			ceil::ceil(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "ceil";
			}

			tensor* ceil::forward(tensor* x)
			{
				return layer_construct(shaders::ceil_spv, sizeof(shaders::ceil_spv), x);
			}

			clip::clip(float min, float max, bool in_place, bool as_module) : unary_operator(in_place, as_module), m_min(min),
			                                                                  m_max(max)
			{
				m_type = "clip";
			}

			tensor* clip::forward(tensor* x)
			{
				m_input.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape());
				m_output.push_back(x->getId());

				if (m_pipeline == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModule(shaders::clip_spv, sizeof(shaders::clip_spv));
					createPipeline(sizeof(clip_operator_param));
				}
				clip_operator_param param = {m_param.total, m_min, m_max};
				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, y, 1, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&param), sizeof(clip_operator_param));
				layers.push_back(this);
				return y;
			}

			exp::exp(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "exp";
			}

			tensor* exp::forward(tensor* x)
			{
				return layer_construct(shaders::exp_spv, sizeof(shaders::exp_spv), x);
			}

			floor::floor(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "floor";
			}

			tensor* floor::forward(tensor* x)
			{
				return layer_construct(shaders::floor_spv, sizeof(shaders::floor_spv), x);
			}

			log::log(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "log";
			}

			tensor* log::forward(tensor* x)
			{
				return layer_construct(shaders::log_spv, sizeof(shaders::log_spv), x);
			}

			round::round(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "round";
			}

			tensor* round::forward(tensor* x)
			{
				return layer_construct(shaders::round_spv, sizeof(shaders::round_spv), x);
			}

			sqrt::sqrt(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "sqrt";
			}

			tensor* sqrt::forward(tensor* x)
			{
				return layer_construct(shaders::sqrt_spv, sizeof(shaders::sqrt_spv), x);
			}

			acos::acos(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "acos";
			}

			tensor* acos::forward(tensor* x)
			{
				return layer_construct(shaders::acos_spv, sizeof(shaders::acos_spv), x);
			}

			acosh::acosh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "acosh";
			}

			tensor* acosh::forward(tensor* x)
			{
				return layer_construct(shaders::acosh_spv, sizeof(shaders::acosh_spv), x);
			}

			asin::asin(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "asin";
			}

			tensor* asin::forward(tensor* x)
			{
				return layer_construct(shaders::asin_spv, sizeof(shaders::asin_spv), x);
			}

			asinh::asinh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "asinh";
			}

			tensor* asinh::forward(tensor* x)
			{
				return layer_construct(shaders::asinh_spv, sizeof(shaders::asinh_spv), x);
			}

			atan::atan(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "atan";
			}

			tensor* atan::forward(tensor* x)
			{
				return layer_construct(shaders::atan_spv, sizeof(shaders::atan_spv), x);
			}

			atanh::atanh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "atan";
			}

			tensor* atanh::forward(tensor* x)
			{
				return layer_construct(shaders::atanh_spv, sizeof(shaders::atanh_spv), x);
			}

			cos::cos(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "cos";
			}

			tensor* cos::forward(tensor* x)
			{
				return layer_construct(shaders::cos_spv, sizeof(shaders::cos_spv), x);
			}

			cosh::cosh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "cosh";
			}

			tensor* cosh::forward(tensor* x)
			{
				return layer_construct(shaders::cosh_spv, sizeof(shaders::cosh_spv), x);
			}

			sin::sin(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "sin";
			}

			tensor* sin::forward(tensor* x)
			{
				return layer_construct(shaders::sin_spv, sizeof(shaders::sin_spv), x);
			}

			sinh::sinh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "sinh";
			}

			tensor* sinh::forward(tensor* x)
			{
				return layer_construct(shaders::sinh_spv, sizeof(shaders::sinh_spv), x);
			}

			tan::tan(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "tan";
			}

			tensor* tan::forward(tensor* x)
			{
				return layer_construct(shaders::tan_spv, sizeof(shaders::tan_spv), x);
			}

			tanh::tanh(bool in_place, bool as_module) : unary_operator(in_place, as_module)
			{
				m_type = "tanh";
			}

			tensor* tanh::forward(tensor* x)
			{
				return layer_construct(shaders::tanh_spv, sizeof(shaders::tanh_spv), x);
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
				return layer_construct(shaders::add_spv, sizeof(shaders::add_spv), x, w);
			}

			sub::sub(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "sub";
			}

			tensor* sub::forward(tensor* x, tensor* w)
			{
				return layer_construct(shaders::sub_spv, sizeof(shaders::sub_spv), x, w);
			}

			mul::mul(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "mul";
			}

			tensor* mul::forward(tensor* x, tensor* w)
			{
				return layer_construct(shaders::mul_spv, sizeof(shaders::mul_spv), x, w);
			}

			div::div(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "div";
			}

			tensor* div::forward(tensor* x, tensor* w)
			{
				return layer_construct(shaders::div_spv, sizeof(shaders::div_spv), x, w);
			}

			mod::mod(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "mod";
			}

			tensor* mod::forward(tensor* x, tensor* w)
			{
				return layer_construct(shaders::mod_spv, sizeof(shaders::mod_spv), x, w);
			}

			pow::pow(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "pow";
			}

			tensor* pow::forward(tensor* x, tensor* w)
			{
				return layer_construct(shaders::pow_spv, sizeof(shaders::pow_spv), x, w);
			}

			max::max(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "max";
			}

			tensor* max::forward(tensor* x, tensor* w)
			{
				return layer_construct(shaders::max_spv, sizeof(shaders::max_spv), x, w);
			}

			min::min(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "min";
			}

			tensor* min::forward(tensor* x, tensor* w)
			{
				return layer_construct(shaders::min_spv, sizeof(shaders::min_spv), x, w);
			}

			eq::eq(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "eq";
			}

			tensor* eq::forward(tensor* x, tensor* w)
			{
				m_input.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
				m_output.push_back(x->getId());

				if (m_pipeline == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModule(shaders::equal_spv, sizeof(shaders::equal_spv));
					createPipeline(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, y, 1, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(operator_param));
				layers.push_back(this);
				return y;
			}

			ne::ne(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "ne";
			}

			tensor* ne::forward(tensor* x, tensor* w)
			{
				m_input.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
				m_output.push_back(x->getId());

				if (m_pipeline == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModule(shaders::nequal_spv, sizeof(shaders::nequal_spv));
					createPipeline(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, y, 1, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(operator_param));
				layers.push_back(this);
				return y;
			}

			lt::lt(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "lt";
			}

			tensor* lt::forward(tensor* x, tensor* w)
			{
				m_input.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
				m_output.push_back(x->getId());

				if (m_pipeline == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModule(shaders::less_than_spv, sizeof(shaders::less_than_spv));
					createPipeline(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, y, 1, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(operator_param));
				layers.push_back(this);
				return y;
			}

			le::le(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "le";
			}

			tensor* le::forward(tensor* x, tensor* w)
			{
				m_input.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
				m_output.push_back(x->getId());

				if (m_pipeline == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModule(shaders::less_eq_spv, sizeof(shaders::less_eq_spv));
					createPipeline(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, y, 1, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(operator_param));
				layers.push_back(this);
				return y;
			}

			gt::gt(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "gt";
			}

			tensor* gt::forward(tensor* x, tensor* w)
			{
				m_input.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
				m_output.push_back(x->getId());

				if (m_pipeline == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModule(shaders::greater_than_spv, sizeof(shaders::greater_than_spv));
					createPipeline(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, y, 1, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(operator_param));
				layers.push_back(this);
				return y;
			}

			ge::ge(bool in_place, bool as_module) : binary_operator(in_place, as_module)
			{
				m_type = "greater_eq";
			}

			tensor* ge::forward(tensor* x, tensor* w)
			{
				m_input.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
				m_output.push_back(x->getId());

				if (m_pipeline == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModule(shaders::greater_eq_spv, sizeof(shaders::greater_eq_spv));
					createPipeline(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, y, 1, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(operator_param));
				layers.push_back(this);
				return y;
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

				m_input.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape(), Format::kFormatBool);
				m_output.push_back(x->getId());

				if (m_pipeline == nullptr)
				{
					m_param = {x->count()};
					computeGroupCount();
					createShaderModule(shaders::xor_spv, sizeof(shaders::xor_spv));
					createPipeline(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, y, 1, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(operator_param));
				layers.push_back(this);
				return y;
			}
		}
	}
}
