#include "common.h"
#include "utils.h"
#include "activation.h"
#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535
/*/
struct noParam
{
	int m_total;
};

struct singleParam
{
	int m_total;
	float alpha;
};

namespace kernel
{
	namespace layers
	{
		namespace nn
		{
			namespace activation
			{
				std::vector<Module*>* ActivationFn::get_module()
				{
					return &Module::module_list;
				}

				ActivationFn::ActivationFn()
				{
					initVulkanThing(2);
					m_type = "Activation";
					m_total = 0;
				}

				void ActivationFn::reshapeOutTensor(tensor* x, tensor* z)
				{
					Shape shape = x->getShape();
					z = &(z->reshape(nullptr, shape));
				}

				bool ActivationFn::forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs)
				{
					return forward(ins[0], outs[0]);
				}

				bool ActivationFn::computeGroupCount()
				{
					m_group_x = static_cast<int>(alignSize(m_total, LOCAL_SZ_X)) / LOCAL_SZ_X;
					if (m_group_x > maxComputeWorkGroupCount)
						m_group_x = maxComputeWorkGroupCount;
					m_group_y = 1;
					m_group_z = 1;
					return true;
				}
			}
		}
	}
}

namespace kernel
{
	namespace layers
	{
		namespace nn
		{
			namespace activation
			{
				elu::elu(float alpha) : m_const(alpha)
				{
					initVulkanThing(2);
					m_type = "elu";
				}

				bool elu::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::elu_spv, sizeof(shaders::elu_spv));
						createPipeline(sizeof(singleParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					singleParam param = { static_cast<int>(m_total), m_const };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(singleParam));
					return true;
				}

				relu::relu()
				{
					initVulkanThing(2);
					m_type = "relu";
				}

				bool relu::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::relu_spv, sizeof(shaders::relu_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				sigmoid::sigmoid()
				{
					initVulkanThing(2);
					m_type = "sigmoid";
				}

				bool sigmoid::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::sigmoid_spv, sizeof(shaders::sigmoid_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}
			}
		}
	}
}

namespace kernel
{
	namespace layers
	{
		namespace nn
		{
			namespace activation
			{
				acos::acos()
				{
					initVulkanThing(2);
					m_type = "acos";
				}

				bool acos::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::acos_spv, sizeof(shaders::acos_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				acosh::acosh()
				{
					initVulkanThing(2);
					m_type = "acosh";
				}

				bool acosh::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::acosh_spv, sizeof(shaders::acosh_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				asin::asin()
				{
					initVulkanThing(2);
					m_type = "asin";
				}

				bool asin::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::asin_spv, sizeof(shaders::asin_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				asinh::asinh()
				{
					initVulkanThing(2);
					m_type = "asinh";
				}

				bool asinh::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::asinh_spv, sizeof(shaders::asinh_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				atan::atan()
				{
					initVulkanThing(2);
					m_type = "atan";
				}

				bool atan::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::atan_spv, sizeof(shaders::atan_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				atanh::atanh()
				{
					initVulkanThing(2);
					m_type = "atanh";
				}

				bool atanh::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::atanh_spv, sizeof(shaders::atanh_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				cos::cos()
				{
					initVulkanThing(2);
					m_type = "acos";
				}

				bool cos::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::cos_spv, sizeof(shaders::cos_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				cosh::cosh()
				{
					initVulkanThing(2);
					m_type = "cosh";
				}

				bool cosh::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::cosh_spv, sizeof(shaders::cosh_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				sin::sin()
				{
					initVulkanThing(2);
					m_type = "asin";
				}

				bool sin::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::sin_spv, sizeof(shaders::sin_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				sinh::sinh()
				{
					initVulkanThing(2);
					m_type = "sinh";
				}

				bool sinh::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::sinh_spv, sizeof(shaders::sinh_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				tan::tan()
				{
					initVulkanThing(2);
					m_type = "tan";
				}

				bool tan::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::tan_spv, sizeof(shaders::tan_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				tanh::tanh()
				{
					initVulkanThing(4);
					m_type = "tanh";
				}

				bool tanh::forward(tensor* x, tensor* y)
				{
					if (m_pipeline == nullptr)
					{
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::tanh_spv, sizeof(shaders::tanh_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
					return true;
				}

				void tanh::backward(tensor* d_output, tensor* d_input)
				{
					if (m_pipeline == nullptr)
					{
						m_total = d_output->count();
						computeGroupCount();
						createShaderModule(shaders::d_tanh_spv, sizeof(shaders::d_tanh_spv));
						createPipeline(sizeof(noParam));
					}
					bindTensor(m_device, d_output, 2, m_descriptor_set);
					bindTensor(m_device, d_input, 3, m_descriptor_set);
					noParam param = { static_cast<int>(m_total) };
					recordCommandBuffer(static_cast<void*>(&param), sizeof(noParam));
				}
			}
		}
	}
}
*/
