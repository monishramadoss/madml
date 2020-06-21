#include "common.h"
#include "utils.h"
#include "matmul.h"
#define maxComputeWorkGroupCount 1024

#define TSM 128                     // The tile-size in dvolension M
#define TSN 128                     // The tile-size in dvolension N
#define TSK 16                      // The tile-size in dvolension K
#define WPTM 4                      // The amount of work-per-thread in dvolension M
#define WPTN 4                      // The amount of work-per-thread in dvolension N
#define LPTA ((TSK*WPTM*WPTN)/(TSN)) // The amount of loads-per-thread for A
#define LPTB ((TSK*WPTM*WPTN)/(TSM)) // The amount of loads-per-thread for B
#define LOCAL_SZ_X 16    // The reduced tile-size in dvolension M (TSM/WPTM number of threads)
#define LOCAL_SZ_Y 16    // The reduced tile-size in dvolension N (TSN/WPTN number of threads)

namespace kernel
{
	namespace layers
	{
		struct matmulParams
		{
			int m;
			int n;
			int k;
		};

		std::vector<Module*>* matmul::get_module()
		{
			return &Module::module_list;
		}

		matmul::matmul()
		{
			initVulkanThing(3);
			m_type = "matmul";
			m_n = 0;
			m_k = 0;
			m_m = 0;
		}

		void matmul::reshapeOutTensor(tensor* x, tensor* z)
		{
			Shape shape = x->getShape();
			*z = z->reshape(nullptr, shape);
		}

		bool matmul::forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs)
		{
			return forward(ins[0], ins[1], outs[0]);
		}

		bool matmul::forward(tensor* x, tensor* y, tensor* z)
		{
			if (m_pipeline == nullptr)
			{
				m_m = x->getShape()[0];
				m_n = y->getShape()[1];
				m_k = x->getShape()[1];
				if (m_k != y->getShape()[0])
					std::cout << "MATML ERROR" << std::endl;
				computeGroupCount();
				createShaderModule(shaders::gemm_spv, sizeof(shaders::gemm_spv));
				createPipeline(sizeof(matmulParams));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);
			bindTensor(m_device, z, 2, m_descriptor_set);

			matmulParams param = { m_m, m_n, m_k };
			recordCommandBuffer(static_cast<void*>(&param), sizeof(matmulParams));
			return true;
		}

		bool matmul::computeGroupCount()
		{
			m_group_x = static_cast<int>(alignSize(m_m, LOCAL_SZ_X)) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = static_cast<int>(alignSize(m_n, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
			if (m_group_y > maxComputeWorkGroupCount)
				m_group_y = maxComputeWorkGroupCount;
			m_group_z = 1;
			return true;
		}
	}
}
