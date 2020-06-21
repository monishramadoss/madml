#include "common.h"
#include "utils.h"
#include "activation.h"
#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

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
				ActivationFn::ActivationFn()
				{
					initVulkanThing(2);
					m_type = "Activation";
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
						createShaderModule(shaders::elu_spv, sizeof(shaders::elu_spv));
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
