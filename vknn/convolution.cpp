#include "../kernel/common.h"
#include "../kernel/utils.h"
#include "convolution.h"

constexpr int local_sz_x_conv = 16;
constexpr int local_sz_y_conv = 64;

std::string vol2col_string = R"(
#version 450

layout(push_constant) uniform pushBlock {
		int total;
		int batchsize;
		int channels;
		int kernel_h;
		int kernel_w;
		int kernel_d;
		int pad_h;
		int pad_w;
		int pad_d;
		int stride_h;
		int stride_w;
		int stride_d;
		int dilation_h;
		int dilation_w;
		int dilation_d;
		int height_col; // height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
		int width_col;  // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
		int depth_col;  // depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1
		int height_vol;
		int width_vol;
		int depth_vol;
};

layout(binding = 0) buffer buf1 { float A[]; };

layout(binding = 1) buffer buf2 { float B[]; };

layout (local_size_x = 16, local_size_y = 64, local_size_z = 1) in;

void vol2col(){
	float n_output_plane = channels * kernel_h * kernel_w * kernel_d;
	uint elt = gl_GlobalInvocationID.y;
	if(elt >= batchsize)
		return;

	float data_vol = elt * channels * height_vol * width_vol * depth_vol;
	float data_col = elt * n_output_plane * height_col * width_col * depth_col;

	uint c_col = gl_GlobalInvocationID.x;
	if(c_col >= n_output_plane)
		return;

	uint w_offset = c_col % uint(kernel_w);
	uint h_offset = uint(c_col / kernel_w) % uint(kernel_h);
	uint d_offset = uint(c_col / kernel_w / kernel_h) % uint(kernel_d);
	uint c_vol = uint(c_col / kernel_w / kernel_h / kernel_d);

	for (uint d_col = 0; d_col < depth_col; ++d_col){
		float d_vol = d_col * stride_d - pad_d + d_offset * dilation_d;
		for (uint h_col = 0; h_col < height_col; ++h_col){
			float h_vol = h_col * stride_h - pad_h + h_offset * dilation_h;
			for (uint w_col = 0; w_col < width_col; ++w_col){
				float w_vol = w_col * stride_w - pad_w + w_offset * dilation_w;
				if (0 <= h_vol && h_vol < height_vol && 0 <= w_vol && w_vol < width_vol && 0 <= d_vol && d_vol < depth_vol){
					uint data_col_idx = uint(floor(data_col + ((c_col * depth_col + d_col) * height_col + h_col) * width_col + w_col));
					uint data_vol_idx = uint(floor(data_vol + ((c_vol * depth_vol + d_vol) * height_vol + h_vol) * width_vol + w_vol));
					B[data_col_idx] = A[data_vol_idx];
				}
			}
		}
	}
}

void main(){
   vol2col();
}
)";

const std::string col2vol_shader = R"(
#version 460

layout(push_constant) uniform pushBlock {
		int total;
		int batchsize;
		int channels;
		int kernel_h;
		int kernel_w;
		int kernel_d;
		int pad_h;
		int pad_w;
		int pad_d;
		int stride_h;
		int stride_w;
		int stride_d;
		int dilation_h;
		int dilation_w;
		int dilation_d;

		int height_col; 
		int width_col;  
		int depth_col; 

		int height_vol;
		int width_vol;
		int depth_vol;
};
layout(binding = 0) buffer buf1 { float A[]; };

layout(binding = 1) buffer buf2 { float B[]; };

layout (local_size_x = 16, local_size_y = 64, local_size_z = 1) in;

void col2vol(){
	uint n_output_plane = channels *kernel_h * kernel_w * kernel_d;
	uint channels_col = channels * kernel_h * kernel_w * kernel_d;

	uint elt = gl_GlobalInvocationID.y;
	if(elt >= batchsize)
		return;

	uint data_vol = elt * channels * height_col * width_col * depth_col;
	uint data_col = elt * n_output_plane * height_vol * width_vol * depth_vol;

	uint c_col = gl_GlobalInvocationID.x;
	if(c_col >= channels_col)
		return;

	uint w_offset = c_col % kernel_w;
	uint h_offset = (c_col / kernel_w) % kernel_h;
	uint d_offset = (c_col / kernel_w / kernel_h) % kernel_d;
	uint c_vol = c_col / kernel_w / kernel_h / kernel_d;

	for (uint d_col = 0; d_col < depth_col; ++d_col){
		uint d_vol = d_col * stride_d - pad_d + d_offset * dilation_d;
		for (uint h_col = 0; h_col < height_col; ++h_col){
			uint h_vol = h_col * stride_h - pad_h + h_offset * dilation_h;
			for (uint w_col = 0; w_col < width_col; ++w_col){
				uint w_vol = w_col * stride_w - pad_w + w_offset * dilation_w;

				if (0 <= h_vol && h_vol < height_vol && 0 <= w_vol && w_vol < width_vol && 0 <= d_vol && d_vol < depth_vol){
					B[data_vol + ((c_vol * depth_vol + d_vol) * height_vol + h_vol) * width_vol + w_vol] +=
						A[data_col + ((c_col * depth_col + d_col) * height_col + h_col) * width_col + w_col];
				}
			}
		}
	}
}

void main(){
   col2vol();
}
)";


vol2col::vol2col(std::vector<int>& params)
{
	initVulkanThing(2);
	m_type = "vol2col";
	m_param = {
		0, 1, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10],
		params[11], params[12], 0, 0, 0, 0, 0, 0
	};
}

void vol2col::forward(std::shared_ptr<tensor>& col, const std::shared_ptr<tensor>& vol)
{
	if (m_pipeline == nullptr)
	{
		m_param.total = vol->count();
		const int depth = static_cast<int>(vol->getShape()[vol->getShape().size() - 3]);
		const int height = static_cast<int>(vol->getShape()[vol->getShape().size() - 2]);
		const int width = static_cast<int>(vol->getShape()[vol->getShape().size() - 1]);
		m_param.batchsize = vol->getShape()[0];
		m_param.channels = vol->getShape()[1];
		m_param.depth_vol = depth;
		m_param.height_vol = height;
		m_param.width_vol = width;
		m_param.depth_col = (depth + 2 * m_param.pad_d - (m_param.dilation_d * (m_param.kernel_d - 1) - 1)) / m_param.stride_d + 1;
		m_param.height_col = (height + 2 * m_param.pad_h - (m_param.dilation_h * (m_param.kernel_h - 1) - 1)) / m_param.stride_h + 1;
		m_param.width_col = (width + 2 * m_param.pad_w - (m_param.dilation_w * (m_param.kernel_w - 1) - 1)) / m_param.stride_w + 1;
		const int n_out_plane = static_cast<int>(m_param.channels * m_param.kernel_d * m_param.kernel_h * m_param.kernel_w);
		const int output_length = static_cast<int>(m_param.batchsize * m_param.depth_col * m_param.height_col * m_param.width_col);

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
		if (!col)
			col = std::make_shared<tensor>(tensor(0., std::vector<int>{n_out_plane, output_length}));

		createShaderModule(nullptr, 0, vol2col_string);
		createPipeline(sizeof(vol2col_param));
	}

	bindtensor(vol, 0);
	bindtensor(col, 1);

	recordCommandBuffer(static_cast<void*>(&m_param), sizeof(vol2col_param));
	runCommandBuffer();
}

col2vol::col2vol(std::vector<int>& params)
{
	m_param = {
		0, 1, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10],
		params[11], params[12], 0, 0, 0, 0, 0, 0
	};
}

void col2vol::forward(std::shared_ptr<tensor>& vol, const std::shared_ptr<tensor>& col)
{
	if (m_pipeline == nullptr)
	{
		m_param.total = vol->count();
		const int depth = static_cast<int>(vol->getShape()[vol->getShape().size() - 3]);
		const int height = static_cast<int>(vol->getShape()[vol->getShape().size() - 2]);
		const int width = static_cast<int>(vol->getShape()[vol->getShape().size() - 1]);
		m_param.batchsize = vol->getShape()[0];
		m_param.channels = vol->getShape()[1];
		m_param.depth_col = depth;
		m_param.height_col = height;
		m_param.width_col = width;
		m_param.depth_vol = (depth - 1) * m_param.stride_d - 2 * m_param.pad_d + m_param.dilation_d * (m_param.kernel_d - 1) + m_param.pad_d + 1;
		m_param.height_vol = (height - 1) * m_param.stride_h - 2 * m_param.pad_h + m_param.dilation_h * (m_param.kernel_h - 1) + m_param.pad_h + 1;
		m_param.width_vol = (width - 1) * m_param.stride_w - 2 * m_param.pad_w + m_param.dilation_w * (m_param.kernel_w - 1) + m_param.pad_w + 1;
		if (m_param.depth_vol != vol->getShape()[2] || m_param.height_col != vol->getShape()[3] || m_param.width_vol != vol->getShape()[4])
			return;

		const int n_out_plane = static_cast<int>(m_param.channels * m_param.kernel_d * m_param.kernel_h * m_param.kernel_w);
		const int output_length = static_cast<int>(m_param.batchsize * m_param.depth_col * m_param.height_col * m_param.width_col);

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

		createShaderModule(nullptr, 0, col2vol_shader);
		createPipeline(sizeof(vol2col_param));
	}

	bindtensor(vol, 0);
	bindtensor(col, 1);

	recordCommandBuffer(static_cast<void*>(&m_param), sizeof(vol2col_param));
	runCommandBuffer();
}