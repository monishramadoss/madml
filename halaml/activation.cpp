#include "common.h"
#include "utils.h"
#include "activation.h"
#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

struct noParam {
	int m_total;
};

struct singleParam {
	int m_total;
	float alpha;
};

namespace kernel {
	namespace layers {
		namespace nn {

			namespace activation {
				Activationfn::Activationfn() {
					initVulkanThing(2);
					m_type = "Activation";
				}

				void Activationfn::reshapeOutTensor(tensor* x, tensor* z) {
					Shape shape = x->getShape();
					z = &(z->reshape(nullptr, shape));
				}

				bool Activationfn::forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) {
					return forward(ins[0], outs[0]);
				}

				bool Activationfn::computeGroupCount() {
					m_group_x = (int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
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


namespace kernel {
	namespace layers {
		namespace nn {
			namespace activation {
				elu::elu(float alpha) : m_const(alpha) {
					initVulkanThing(2);
					m_type = "elu";
				}

				bool elu::forward(tensor* x, tensor* y) {
					if (m_pipeline == VK_NULL_HANDLE) {
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::elu_spv, sizeof(shaders::elu_spv));
						createPipeline(sizeof(singleParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					singleParam param = { (int)m_total, m_const };
					recordCommandBuffer((void*)&param, sizeof(singleParam));
					return true;
				}

				relu::relu() {
					initVulkanThing(2);
					m_type = "relu";
				}

				bool relu::forward(tensor* x, tensor* y) {
					if (m_pipeline == VK_NULL_HANDLE) {
						m_total = x->count();
						computeGroupCount();
						createShaderModule(shaders::elu_spv, sizeof(shaders::elu_spv));
						createPipeline(sizeof(noParam));
					}

					bindTensor(m_device, x, 0, m_descriptor_set);
					bindTensor(m_device, y, 1, m_descriptor_set);
					noParam param = { (int)m_total };
					recordCommandBuffer((void*)&param, sizeof(noParam));
					return true;
				}
			}
		}
	}
}

