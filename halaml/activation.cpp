#include "common.h"
#include "utils.h"
#include "activation.h"
#define LOCAL_SZ_X 1024
#define MAX_COMPUTE_WORK_GROUP_COUNT 65535

namespace kernel
{
	namespace layers
	{
		namespace activation
		{
			unary_operator::unary_operator(float alpha, bool in_place, bool as_module) : as_module(as_module),
			                                                                             m_inplace(in_place),
																						 m_alpha(alpha),
																						 m_param({0, m_alpha})
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
					m_param.total = x->count();
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

			celu::celu(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module)
			{
				m_type = "celu";
			}

			tensor* celu::forward(tensor* x)
			{
				tensor* y;
				tensor* alpha;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape());
				alpha = new tensor(1.0, x->getShape());

				if (m_pipeline_forward == nullptr)
				{
					m_param.total = x->count();
					computeGroupCount();
					createShaderModuleForward(shaders::celu_spv, sizeof(shaders::celu_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, alpha, 1, m_descriptor_set_forward);
				bindTensor(m_device, y, 2, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(operator_param));

				inputs.push_back(x->getId());
				inputs.push_back(alpha->getId());
				outputs.push_back(y->getId());
				layers.push_back(this);
				if (as_module)
					add_module(this);

				return y;
			}

			void celu::back_propagate()
			{
				tensor* x = get_grad(inputs[0]);
				tensor* alpha = get_grad(inputs[1]);
				tensor* y = get_grad(outputs[0]);

				if (m_pipeline_forward == nullptr)
				{
					m_param.total = x->count();
					computeGroupCount();
					createShaderModuleForward(shaders::d_celu_spv, sizeof(shaders::d_celu_spv));
					createPipelineForward(sizeof(operator_param));
				}

				bindTensor(m_device, x, 2, m_descriptor_set_backward);
				bindTensor(m_device, alpha, 1, m_descriptor_set_backward);
				bindTensor(m_device, y, 0, m_descriptor_set_backward);

				recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(operator_param));
			}

			elu::elu(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module)
			{
				m_type = "elu";
			}

			tensor* elu::forward(tensor* x)
			{
				return layer_construct_forward(shaders::elu_spv, sizeof(shaders::elu_spv), x);
			}

			void elu::back_propagate()
			{
				layer_construct_backward(shaders::d_elu_spv, sizeof(shaders::d_elu_spv));
			}

			hardshrink::hardshrink(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module)
			{
				m_type = "hardshrink";
			}

			tensor* hardshrink::forward(tensor* x)
			{
				return layer_construct_forward(shaders::hardshrink_spv, sizeof(shaders::hardshrink_spv), x);
			}

			void hardshrink::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			hardtanh::hardtanh(float min_val, float max_val, bool in_place, bool as_module) :
				unary_operator(0, in_place, as_module), m_param({0, min_val, max_val})
			{
				m_type = "hardtanh";
			}

			tensor* hardtanh::forward(tensor* x)
			{
				inputs.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape());
				outputs.push_back(x->getId());

				if (m_pipeline_forward == nullptr)
				{
					m_param.total = x->count();
					computeGroupCount();
					createShaderModuleForward(shaders::hardtanh_spv, sizeof(shaders::hardtanh_spv));
					createPipelineForward(sizeof(two_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set_forward);
				bindTensor(m_device, y, 1, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(two_param));
				layers.push_back(this);
				if (as_module)
					add_module(this);
				return y;
			}

			void hardtanh::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			leakyrelu::leakyrelu(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module)
			{
				m_type = "leakyrelu";
			}

			tensor* leakyrelu::forward(tensor* x)
			{
				return layer_construct_forward(shaders::leakyrelu_spv, sizeof(shaders::leakyrelu_spv), x);
			}

			void leakyrelu::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			logsigmoid::logsigmoid(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module)
			{
				m_type = "logsigmoid";
			}

			tensor* logsigmoid::forward(tensor* x)
			{
				return layer_construct_forward(shaders::logsigmoid_spv, sizeof(shaders::logsigmoid_spv), x);
			}

			void logsigmoid::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			prelu::prelu(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module)
			{
				m_type = "prelu";
			}

			tensor* prelu::forward(tensor* x)
			{
				return layer_construct_forward(shaders::prelu_spv, sizeof(shaders::prelu_spv), x);
			}

			void prelu::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			relu::relu(bool in_place, bool as_module) : unary_operator(0, in_place, as_module)
			{
				m_type = "relu";
			}

			tensor* relu::forward(tensor* x)
			{
				return layer_construct_forward(shaders::relu_spv, sizeof(shaders::relu_spv), x);
			}

			void relu::back_propagate() 
			{
				layer_construct_backward(shaders::d_relu_spv, sizeof(shaders::d_relu_spv));
			}

			relu6::relu6(bool in_place, bool as_module) : unary_operator(0, in_place, as_module)
			{
				m_type = "relu6";
			}

			tensor* relu6::forward(tensor* x)
			{
				return layer_construct_forward(shaders::relu6_spv, sizeof(shaders::relu6_spv), x);
			}

			void relu6::back_propagate()
			{
				layer_construct_backward(shaders::d_relu6_spv, sizeof(shaders::d_relu6_spv));
			}

			selu::selu(bool in_place, bool as_module) : unary_operator(0, in_place, as_module)
			{
				m_type = "selu";
			}

			tensor* selu::forward(tensor* x)
			{
				return layer_construct_forward(shaders::selu_spv, sizeof(shaders::selu_spv), x);
			}

			void selu::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			sigmoid::sigmoid(bool in_place, bool as_module) : unary_operator(0, in_place, as_module)
			{
				m_type = "sigmoid";
			}

			tensor* sigmoid::forward(tensor* x)
			{
				return layer_construct_forward(shaders::sigmoid_spv, sizeof(shaders::sigmoid_spv), x);
			}

			void sigmoid::back_propagate() 
			{
				layer_construct_backward(shaders::d_sigmoid_spv, sizeof(shaders::d_sigmoid_spv));
			}

			softplus::softplus(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module)
			{
				m_type = "softplus";
			}

			tensor* softplus::forward(tensor* x)
			{
				return layer_construct_forward(shaders::softplus_spv, sizeof(shaders::softplus_spv), x);
			}

			void softplus::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			softshrink::softshrink(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module)
			{
				m_type = "softshrink";
			}

			tensor* softshrink::forward(tensor* x)
			{
				return layer_construct_forward(shaders::softshrink_spv, sizeof(shaders::softshrink_spv), x);
			}

			void softshrink::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			softsign::softsign(bool in_place, bool as_module) : unary_operator(0, in_place, as_module)
			{
				m_type = "softsign";
			}

			tensor* softsign::forward(tensor* x)
			{
				return layer_construct_forward(shaders::softsign_spv, sizeof(shaders::softsign_spv), x);
			}

			void softsign::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}

			tanhshrink::tanhshrink(bool in_place, bool as_module) : unary_operator(0, in_place, as_module)
			{
				m_type = "tanhshrink";
			}

			tensor* tanhshrink::forward(tensor* x)
			{
				return layer_construct_forward(shaders::tanhshrink_spv, sizeof(shaders::tanhshrink_spv), x);
			}

			void tanhshrink::back_propagate()
			{
				layer_construct_backward(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv));
			}
		}
	}
}
