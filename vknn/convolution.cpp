#include "../engine/common.h"
#include "../engine/utils.h"
#include "convolution.h"

constexpr int local_sz_x_conv = 16;
constexpr int local_sz_y_conv = 64;



vol2col::vol2col(std::vector<int>& params)
{
	m_future = std::async(&vol2col::initVulkanThing, &*this, 2);
	m_type = "vol2col";
	m_param.batchsize = params[0];
	m_param.channels = params[1];

	m_param.kernel_d = params[2];
	m_param.kernel_h = params[3];
	m_param.kernel_w = params[4];
	
	m_param.pad_d = params[5];
	m_param.pad_h = params[9];
	m_param.pad_w = params[7];

	m_param.stride_d = params[8];
	m_param.stride_h = params[9];
	m_param.stride_w = params[10];

	m_param.dilation_d = params[11];
	m_param.dilation_h = params[12];
	m_param.dilation_w = params[13];

	m_param.depth_col = params[14];
	m_param.height_col = params[15];
	m_param.width_col = params[16];
	
	m_param.depth_vol = params[17];
	m_param.height_vol = params[18];
	m_param.width_vol = params[19];

	m_futures.resize(2);
}

void vol2col::forward(tensor& col, tensor& vol)
{
	if (m_pipeline == nullptr)
	{
		m_param.total = vol.count();

		size_t tmp = m_param.channels;
		tmp *= static_cast<size_t>(m_param.kernel_h);
		tmp *= static_cast<size_t>(m_param.kernel_w);
		tmp *= static_cast<size_t>(m_param.kernel_d);

		m_group_x = static_cast<int>(alignSize(tmp, local_sz_x_conv)) / local_sz_x_conv;
		if (m_group_x > max_compute_work_group_count)
			m_group_x = max_compute_work_group_count;
		m_group_y = static_cast<int>(alignSize(m_param.batchsize, local_sz_y_conv)) / local_sz_y_conv;
		if (m_group_y > max_compute_work_group_count)
			m_group_y = max_compute_work_group_count;
		m_group_z = 1;
		m_future.wait();
		createShaderModule(vol2col_spv, sizeof(vol2col_spv));
		createPipeline(sizeof(vol2col_param));
	}

	m_futures[0] = std::async(&vol2col::bindtensor, &*this, vol, 0);
	m_futures[1] = std::async(&vol2col::bindtensor, &*this, col, 1);
	m_future = std::async(&vol2col::recordCommandBuffer, &*this, static_cast<void*>(&m_param), sizeof(vol2col_param));	
	runCommandBuffer();
}

col2vol::col2vol(std::vector<int>& params)
{
	m_future = std::async(&vol2col::initVulkanThing, &*this, 2);
	m_type = "col2vol";
	m_param.batchsize = params[0];
	m_param.channels = params[1];

	m_param.kernel_d = params[2];
	m_param.kernel_h = params[3];
	m_param.kernel_w = params[4];

	m_param.pad_d = params[5];
	m_param.pad_h = params[9];
	m_param.pad_w = params[7];

	m_param.stride_d = params[8];
	m_param.stride_h = params[9];
	m_param.stride_w = params[10];

	m_param.dilation_d = params[11];
	m_param.dilation_h = params[12];
	m_param.dilation_w = params[13];

	m_param.depth_col = params[14];
	m_param.height_col = params[15];
	m_param.width_col = params[16];

	m_param.depth_vol = params[17];
	m_param.height_vol = params[18];
	m_param.width_vol = params[19];
}

void col2vol::forward(tensor& vol, tensor& col)
{
	if (m_pipeline == nullptr)
	{
		size_t tmp = m_param.channels;
		tmp *= static_cast<size_t>(m_param.kernel_h);
		tmp *= static_cast<size_t>(m_param.kernel_w);
		tmp *= static_cast<size_t>(m_param.kernel_d);

		m_group_x = static_cast<int>(alignSize(tmp, local_sz_x_conv)) / local_sz_x_conv;
		if (m_group_x > max_compute_work_group_count)
			m_group_x = max_compute_work_group_count;
		m_group_y = static_cast<int>(alignSize(m_param.batchsize, local_sz_y_conv)) / local_sz_y_conv;
		if (m_group_y > max_compute_work_group_count)
			m_group_y = max_compute_work_group_count;
		m_group_z = 1;
		m_future.wait();
		createShaderModule(col2vol_spv, sizeof(col2vol_spv));
		createPipeline(sizeof(vol2col_param));
	}

	m_futures[0] = std::async(&vol2col::bindtensor, &*this, vol, 0);
	m_futures[1] = std::async(&vol2col::bindtensor, &*this, col, 1);
	m_future = std::async(&vol2col::recordCommandBuffer, &*this, static_cast<void*>(&m_param), sizeof(vol2col_param));
	runCommandBuffer();
}