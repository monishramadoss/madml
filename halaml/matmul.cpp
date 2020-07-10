#include "common.h"
#include "utils.h"
#include "matmul.h"
#define MAX_COMPUTE_WORK_GROUP_COUNT 1024

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
		matmul::matmul()
		{
			initVulkanThing(3);
			m_type = "matmul";
		}

		void matmul::computeGroupCount()
		{
			m_group_x = static_cast<int>(alignSize(m_param.m, LOCAL_SZ_X)) / LOCAL_SZ_X;
			if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
				m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
			m_group_y = static_cast<int>(alignSize(m_param.n, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
			if (m_group_y > MAX_COMPUTE_WORK_GROUP_COUNT)
				m_group_y = MAX_COMPUTE_WORK_GROUP_COUNT;
			m_group_z = 1;
		}

		tensor* matmul::forward(tensor* x, tensor* w)
		{
			if (x->getShape()[1] != w->getShape()[0])
				std::cerr << "Mat mul dim ERROR" << std::endl;
			m_input.push_back(x->getId());
			m_input.push_back(w->getId());
			auto* y = new tensor(0.0, std::vector<int>{x->getShape()[0], w->getShape()[1]});
			m_output.push_back(y->getId());

			if (m_pipeline == nullptr)
			{
				m_param = {x->getShape()[0], w->getShape()[1], x->getShape()[1]};
				computeGroupCount();
				createShaderModule(shaders::gemm_spv, sizeof(shaders::gemm_spv));
				createPipeline(sizeof(matmul_param));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, w, 1, m_descriptor_set);
			bindTensor(m_device, y, 2, m_descriptor_set);

			recordCommandBuffer(static_cast<void*>(&m_param), sizeof(matmul_param));
			layers.push_back(this);
			return y;
		}
	}
}
