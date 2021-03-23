#include "../kernel/common.h"
#include "../kernel/utils.h"

#include "gemm.h"



gemm::gemm(float alpha, float beta, bool use_bias)
{
	initVulkanThing(4);
	m_type = "gemm";
	m_param.alpha = alpha;
	m_param.beta = beta;
	m_param.use_bias = use_bias;
   
}

void gemm::forward(std::shared_ptr<tensor>& y, const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w, const std::shared_ptr<tensor>& b)
{
	if (m_pipeline == nullptr)
	{
		if (x->getShape().size() == w->getShape().size() + 1)
		{
			m_param.batchsize = x->getShape()[0];
			m_param.m = x->getShape()[1];
			m_param.k = x->getShape()[2];
			m_param.n = w->getShape()[1];
		}
		else
		{
			m_param.batchsize = x->getShape()[0];
			m_param.m = 1; // x->getShape()[0];
			m_param.k = x->getShape()[1];
			m_param.n = w->getShape()[1];
		}
		m_param.total = w->count();

		m_group_x = static_cast<int>(alignSize(m_param.m, 16)) / 16;
		m_group_y = static_cast<int>(alignSize(m_param.n, 16)) / 16;
		m_group_z = static_cast<int>(alignSize(m_param.batchsize, 4)) / 4;

		if (m_group_x > max_compute_work_group_count)
			m_group_x = max_compute_work_group_count - 1;
		if (m_group_y > max_compute_work_group_count)
			m_group_y = max_compute_work_group_count - 1;
		if (m_group_z > max_compute_work_group_count)
			m_group_z = max_compute_work_group_count - 1;

		createShaderModule(gemm_spv, sizeof(gemm_spv));
		createPipeline(sizeof(gemm_param));
	}


	bindtensor(*x, 0);
	bindtensor(*w, 1);
	bindtensor(*b, 2);
	bindtensor(*y, 3);

	recordCommandBuffer(static_cast<void*>(&m_param), sizeof(gemm_param));
	runCommandBuffer();


	auto* tmp = (float*)y->toHost();
	for (int i = 0; i < y->size() / sizeof(float); ++i)
		std::cout << tmp[i] << ", " << std::endl;
	std::cout << std::endl;
	return;
}