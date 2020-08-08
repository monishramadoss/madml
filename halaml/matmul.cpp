#include "common.h"
#include "utils.h"
#include "matmul.h"
#define MAX_COMPUTE_WORK_GROUP_COUNT 1024

#define TSM 128                     // The tile-size in dimension M
#define TSN 128                     // The tile-size in dimension N
#define TSK 16                      // The tile-size in dimension K
#define WPTM 4                      // The amount of work-per-thread in dimension M
#define WPTN 4                      // The amount of work-per-thread in dimension N
#define LPTA ((TSK*WPTM*WPTN)/(TSN)) // The amount of loads-per-thread for A
#define LPTB ((TSK*WPTM*WPTN)/(TSM)) // The amount of loads-per-thread for B
#define LOCAL_SZ_X 16    // The reduced tile-size in dimension M (TSM/WPTM number of threads)
#define LOCAL_SZ_Y 16    // The reduced tile-size in dimension N (TSN/WPTN number of threads)

namespace kernel
{
	namespace layers
	{
		matmul::matmul() : Base_Layer<matmul_param>(3)
		{
			m_type = "matmul";
			m_param = {0, 0, 0};
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

		std::shared_ptr<tensor>& matmul::hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
		{
			if (x->getShape()[1] != w->getShape()[0])
				std::cerr << "Mat mul dim ERROR" << std::endl;
			m_param = {0, x->getShape()[0], w->getShape()[1], x->getShape()[1]};
			y = layer_construct_forward(shaders::gemm_spv, sizeof(shaders::gemm_spv), x, w, Format::kFormatFp32,
			                            std::vector<int>{x->getShape()[0], w->getShape()[1]});
			return y;
		}
	}
}
