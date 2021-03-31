#include "../engine/common.h"
#include "../engine/utils.h"

#include "gemm.h"



gemm::gemm(float alpha, float beta, bool use_bias)
{
	initVulkanThing(4);
	m_type = "gemm";	
	m_param.use_bias = use_bias;
   
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

		createShaderModule(gemm_spv, sizeof(gemm_spv));
		createPipeline(sizeof(gemm_param));
	}
	bindtensor(x, 0);
	bindtensor(w, 1);
	bindtensor(b, 2);
	bindtensor(y, 3);

	recordCommandBuffer(static_cast<void*>(&m_param), sizeof(gemm_param));
	runCommandBuffer();


	auto* tmp = (float*)y.toHost();
	for (int i = 0; i < y.size() / sizeof(float); ++i)
		std::cout << tmp[i] << ", ";
	std::cout << std::endl;
	return;
}

void test_gemm()
{
	const std::vector<int> shape{ 10 , 10 };

	auto x = tensor(1.0, shape);
	auto w = tensor(1.0, shape);
	auto b = tensor(1.0, shape);
	auto y = tensor(0.0, shape);

	auto m = gemm(1.0, 1.0, false);
	m.forward(y, x, w, b);
}