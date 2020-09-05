#include "common.h"
#include "utils.h"
#include "matmul.h"

#define GEMM_2
#ifdef GEMM_2
#define TSM 512                     // The tile-size in dvolension M
#define TSN 512                     // The tile-size in dvolension N
#define TSK 64                      // The tile-size in dvolension K
#define WPTM 8                      // The amount of work-per-thread in dvolension M
#define WPTN 8                     // The amount of work-per-thread in dvolension N
#define RTSM (TSM/WPTM)    // The reduced tile-size in dvolension M (TSM/WPTM number of threads)
#define RTSN (TSN/WPTN)   // The reduced tile-size in dvolension N (TSN/WPTN number of threads)
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B
#endif
#ifdef GEMM_3
#define TS 32u
#define WPT 8u
#define RTS (TS/WPT)
#endif

namespace layers
{
	bool is_power_of_two(uint32_t x)
	{
		return ((x != 0) && ((x & (~x + 1)) == x));
	}

	matmul::matmul() : Base_Layer<matmul_param>(3)
	{
		m_type = "matmul";
		m_param = { 0, 0, 0 };
		bck_codeSize = sizeof(kernel::shaders::d_gemm_spv);
		bck_shader = kernel::shaders::d_gemm_spv;
		bck_shader = kernel::shaders::d_gemm_spv;
	}

	void matmul::computeGroupCount()
	{
		m_group_x = static_cast<int>(alignSize(m_param.m, TSM)) / TSM; //256 -> 2
		m_group_y = static_cast<int>(alignSize(m_param.n, TSN)) / TSN; //256 -> 64
		m_group_z = static_cast<int>(alignSize(m_param.batchsize, 1)) / 1;

		if (m_group_x > max_compute_work_group_count)
			m_group_x = max_compute_work_group_count - 1;
		if (m_group_y > max_compute_work_group_count)
			m_group_y = max_compute_work_group_count - 1;
		if (m_group_z > max_compute_work_group_count)
			m_group_z = max_compute_work_group_count - 1;
	}

	std::shared_ptr<tensor>& matmul::operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w)
	{
		if (x->getShape().size() == w->getShape().size() + 1)
		{
			if (x->getShape()[x->getShape().size() - 1] != w->getShape()[0])
				std::cerr << "Mat mul dim ERROR" << std::endl;
			m_param.batchsize = x->getShape()[0];
			m_param.m = x->getShape()[1];
			m_param.k = x->getShape()[2];
			m_param.n = w->getShape()[1];
			m_param.is_power_of_two = is_power_of_two(m_param.m) && is_power_of_two(m_param.n) && is_power_of_two(m_param.k);
			return layer_construct_forward(kernel::shaders::gemm_spv, sizeof(kernel::shaders::gemm_spv), x, w,
				Format::kFormatFp32,
				std::vector<int>{m_param.batchsize, m_param.m, m_param.n});
		}

		if (x->getShape()[x->getShape().size() - 1] != w->getShape()[0])
			std::cerr << "Mat mul dim ERROR" << std::endl;
		m_param.total = 0;
		m_param.batchsize = 1;
		m_param.m = x->getShape()[0];
		m_param.k = x->getShape()[1];
		m_param.n = w->getShape()[1];
		m_param.is_power_of_two = is_power_of_two(m_param.m) && is_power_of_two(m_param.n) && is_power_of_two(m_param.k);
		return layer_construct_forward(kernel::shaders::gemm_spv, sizeof(kernel::shaders::gemm_spv), x, w, Format::kFormatFp32,
			std::vector<int>{m_param.m, m_param.n});
	}

	int matmul::set_backward()
	{
		// dx = dy * w.T  // MxK = MxN NxK
		// dw = I.T * dy  // KxN = KxM MxN
		// MxK KxN = MxN
		if (!dx)
			dx = std::make_shared<tensor>(tensor(0., x->getShape()));
		if (!dw)
			dw = std::make_shared<tensor>(tensor(0., w->getShape()));
		if (!dy)
			dy = std::make_shared<tensor>(tensor(0., y->getShape()));

		if (derivative->m_pipeline == nullptr)
		{
			m_param.total = x->count();
			derivative->createShaderModule(kernel::shaders::d_gemm_spv, sizeof(kernel::shaders::d_gemm_spv));
			derivative->createPipeline(sizeof(matmul_param));
		}

		derivative->bindTensor(x, 0);
		derivative->bindTensor(w, 1);
		derivative->bindTensor(dy, 2);
		derivative->bindTensor(dw, 3);
		derivative->bindTensor(dx, 4);

		derivative->recordCommandBuffer(static_cast<void*>(&m_param), sizeof(matmul_param));
		derivative->runCommandBuffer();
		return dy->getId();
	}
}