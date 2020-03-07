#include "common.h"
#include "utils.h"
#include "operators.h"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct tensorParam {
			int total;
		};

		operators::operators(int op_id) {
			layer::initVulkanThing(3);
			m_type = "operators";
			m_op = op_id;
		}

		void operators::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool operators::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], ins[1], outs[0]);
		}

		bool operators::forward(tensor& in, tensor& in2, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				switch (m_op) {
				case 0:
					createShaderModule(shaders::add_spv, sizeof(shaders::add_spv));
					break;
				case 1:
					createShaderModule(shaders::sub_spv, sizeof(shaders::sub_spv));
					break;
				case 2:
					createShaderModule(shaders::mul_spv, sizeof(shaders::mul_spv));
					break;
				case 3:
					createShaderModule(shaders::add_spv, sizeof(shaders::div_spv));
					break;
				case 4:
					createShaderModule(shaders::mod_spv, sizeof(shaders::mod_spv));
					break;
				}
				createPipeline(sizeof(tensorParam));			
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, in2, 1, m_descriptor_set);
			bindTensor(m_device, out, 2, m_descriptor_set);			
			tensorParam param = { m_total };
			recordCommandBuffer((void*)&param, sizeof(tensorParam));	

			//cmd_layer.push_back(this); //TODO check if obj in cmd_layer

			return true;
		}

		bool operators::run(){
			runCommandBuffer();
			return true;
		}

		bool operators::computeGroupCount() {
			m_group_x = (int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}