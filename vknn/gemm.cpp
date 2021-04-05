#include "../engine/common.h"
#include "../engine/utils.h"

#include "gemm.h"
#include <future>        


gemm::gemm(float alpha, float beta, bool use_bias, bool transpose_x, bool transpose_w) : m_transpose_w(transpose_w), m_transpose_x(transpose_x)
{
	m_future = std::async(&gemm::initVulkanThing, &*this, 4);
	m_type = "gemm";	
	m_param.alpha = alpha;
	m_param.beta = beta;
	m_param.use_bias = use_bias;
	m_futures.resize(4);
   
}

void gemm::forward(tensor& y,tensor& x, tensor& w, tensor& b)
{
	
	if (m_pipeline == nullptr)
	{
		if (x.getShape().size() == w.getShape().size() + 1)
		{
			m_param.batchsize = x.getShape()[0];
			m_param.m = x.getShape()[1];
			m_param.k = x.getShape()[2];
			m_param.n = w.getShape()[1];
		}
		else
		{
			m_param.batchsize = x.getShape()[0];
			m_param.m = 1; 
			m_param.k = x.getShape()[1];
			m_param.n = w.getShape()[1];
		}
		m_param.total = w.count();

		m_group_x = static_cast<int>(alignSize(m_param.m, 16)) / 16;
		m_group_y = static_cast<int>(alignSize(m_param.n, 16)) / 16;
		m_group_z = static_cast<int>(alignSize(m_param.batchsize, 2)) / 2;

		if (m_group_x > max_compute_work_group_count)
			m_group_x = max_compute_work_group_count - 1;
		if (m_group_y > max_compute_work_group_count)
			m_group_y = max_compute_work_group_count - 1;
		if (m_group_z > max_compute_work_group_count)
			m_group_z = max_compute_work_group_count - 1;
		m_future.wait();
		if (m_transpose_x)
			createShaderModule(xt_gemm_spv, sizeof(xt_gemm_spv));
		else if (m_transpose_w)
			createShaderModule(wt_gemm_spv, sizeof(wt_gemm_spv));
		else
			createShaderModule(gemm_spv, sizeof(gemm_spv));
		createPipeline(sizeof(gemm_param));
	}

	m_futures[0] = std::async(&gemm::bindtensor, &*this, x, 0);
	m_futures[1] = std::async(&gemm::bindtensor, &*this, w, 1);
	m_futures[2] = std::async(&gemm::bindtensor, &*this, b, 2);
	m_futures[3] = std::async(&gemm::bindtensor, &*this, y, 3);

	m_future = std::async(&gemm::recordCommandBuffer, &*this, static_cast<void*>(&m_param), sizeof(gemm_param));
	runCommandBuffer();
	return;
}
