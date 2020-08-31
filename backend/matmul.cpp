#include "common.h"
#include "utils.h"
#include "matmul.h"

#define TSM 128                     // The tile-size in dvolension M
#define TSN 128                     // The tile-size in dvolension N
#define TSK 16                      // The tile-size in dvolension K
#define WPTM 8                      // The amount of work-per-thread in dvolension M
#define WPTN 8                      // The amount of work-per-thread in dvolension N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B
#define RTSM (TSM/WPTM)    // The reduced tile-size in dvolension M (TSM/WPTM number of threads)
#define RTSN (TSN/WPTN)   // The reduced tile-size in dvolension N (TSN/WPTN number of threads)

namespace layers
{
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
		m_group_x = static_cast<int>(alignSize(m_param.m, RTSM)) / RTSM;
		if (m_group_x > max_compute_work_group_count)
			m_group_x = max_compute_work_group_count;
		m_group_y = static_cast<int>(alignSize(m_param.n, RTSN)) / RTSN;
		if (m_group_y > max_compute_work_group_count)
			m_group_y = max_compute_work_group_count;
		m_group_z = static_cast<int>(alignSize(m_param.batchsize, 1)) / 1;
		if (m_group_z > max_compute_work_group_count)
			m_group_z = max_compute_work_group_count;
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
			return layer_construct_forward(kernel::shaders::gemm_spv, sizeof(kernel::shaders::gemm_spv), x, w, Format::kFormatFp32,
				std::vector<int>{m_param.batchsize, m_param.m, m_param.n});
		}
		else
		{
			if (x->getShape().size() != w->getShape().size())
				std::cerr << "Mat mul dim ERROR" << std::endl;
			if (x->getShape()[x->getShape().size() - 1] != w->getShape()[0])
				std::cerr << "Mat mul dim ERROR" << std::endl;
			m_param.total = 0;
			m_param.batchsize = 1;
			m_param.m = x->getShape()[0];
			m_param.k = x->getShape()[1];
			m_param.n = w->getShape()[1];
			return layer_construct_forward(kernel::shaders::gemm_spv, sizeof(kernel::shaders::gemm_spv), x, w, Format::kFormatFp32,
				std::vector<int>{m_param.m, m_param.n});
		}
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