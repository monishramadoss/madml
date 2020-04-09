#include "common.h"
#include "utils.h"
#include "matmul.h"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct matmulParams {
			int m;
			int n;
			int k;
			int use_bias;
		};

		matmul::matmul(bool use_bias) {
			layer::initVulkanThing(3);
			m_type = "matmul";
			m_use_bias = use_bias;
		}

		void matmul::reshapeOutTensor(tensor& x, tensor& z) {
			Shape shape = x.getShape();
			z = z.reshape(nullptr, shape);
		}

		bool matmul::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], ins[1], ins[2], outs[0]);
		}

		bool matmul::forward(tensor& x, tensor& y, tensor& z, tensor& w) {
			if (m_pipeline == VK_NULL_HANDLE) {				
				computeGroupCount();
				createShaderModule(shaders::gemm_spv, sizeof(shaders::gemm_spv));
				createPipeline(sizeof(matmulParams));
				m_m = x.getShape()[0];
				m_n = y.getShape()[1];
				m_k = x.getShape()[1];
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);
			bindTensor(m_device, z, 2, m_descriptor_set);
			bindTensor(m_device, w, 3, m_descriptor_set);
			matmulParams param = { m_m, m_n, m_k, m_use_bias };
			recordCommandBuffer((void*)&param, sizeof(matmulParams));
			return true;
		}

		bool matmul::run() {
			runCommandBuffer();
			return 1;
		}
		
		bool matmul::computeGroupCount() {
			m_group_x = 16;
			m_group_y = 16;
			m_group_z = 1;
			return true;
		}
	}
}