#include "common.h"
#include "utils.h"
#include "transform.h"

#define LOCAL_SZ_X 16
#define LOCAL_SZ_Y 64
#define maxComputeWorkGroupCount 65535

namespace kernel
{
	namespace layers
	{
		vol2col::vol2col(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation) :
			m_param({ 1, channels, kernel.h, kernel.w, kernel.d, pad.h, pad.w, pad.d, stride.h, stride.w, stride.d, dilation.h, dilation.w, dilation.d,
				0, 0, 0, 0, 0, 0 })
		{
			initVulkanThing(2);
			m_type = "vol2col";
		}

		void vol2col::computeGroupCount()
		{
			size_t tmp = m_param.channels;
			tmp *= m_param.kernel_h;
			tmp *= m_param.kernel_w;
			tmp *= m_param.kernel_d;

			m_group_x = static_cast<int>(alignSize(tmp, LOCAL_SZ_X)) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1; // static_cast<int>(alignSize(1, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
			if (m_group_y > maxComputeWorkGroupCount)
				m_group_y = maxComputeWorkGroupCount;
			m_group_z = 1;
		}

		tensor* vol2col::forward(tensor* x)
		{
			m_input.push_back(x->getId());

			if (m_pipeline == nullptr)
			{
				const int depth = x->getShape()[x->getShape().size() - 3];
				const int height = x->getShape()[x->getShape().size() - 2];
				const int width = x->getShape()[x->getShape().size() - 1];
				m_param.batchsize = 1;
				m_param.depth_vol = depth;
				m_param.height_vol = height;
				m_param.width_vol = width;
				m_param.depth_col = (depth + 2 * m_param.pad_d - (m_param.dilation_d * (m_param.kernel_d - 1) + 1)) / m_param.stride_d + 1;
				m_param.height_col = (height + 2 * m_param.pad_h - (m_param.dilation_h * (m_param.kernel_h - 1) + 1)) / m_param.stride_h + 1;
				m_param.width_col = (width + 2 * m_param.pad_w - (m_param.dilation_w * (m_param.kernel_w - 1) + 1)) / m_param.stride_w + 1;
				computeGroupCount();
				createShaderModule(shaders::vol2col_spv, sizeof(shaders::vol2col_spv));
				createPipeline(sizeof(vol2col_param));
			}

			const int n_out_plane = m_param.channels * m_param.kernel_d * m_param.kernel_h * m_param.kernel_w;
			const int output_length = m_param.depth_col * m_param.height_col * m_param.width_col;
			auto* y = new tensor(0.0, std::vector<int>{output_length* n_out_plane});
			m_output.push_back(y->getId());
			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);

			recordCommandBuffer(static_cast<void*>(&m_param), sizeof(vol2col_param));
			layers.push_back(this);
			y->reshape(std::vector<int>{n_out_plane, output_length});
			return y;
		}

		std::vector<int> vol2col::output_shape() const
		{
			return std::vector<int>{m_param.depth_col, m_param.height_col, m_param.width_col};
		}

		col2vol::col2vol(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation) :
			m_param({ 1, channels, kernel.h, kernel.w, kernel.d, pad.h, pad.w, pad.d, stride.h, stride.w, stride.d, dilation.h, dilation.w, dilation.d, 0, 0, 0, 0, 0, 0 })
		{
			initVulkanThing(2);
			m_type = "col2vol";
		}

		void col2vol::computeGroupCount()
		{
			size_t tmp = m_param.channels;
			tmp *= m_param.kernel_h;
			tmp *= m_param.kernel_w;
			tmp *= m_param.kernel_d;

			m_group_x = static_cast<int>(alignSize(1, LOCAL_SZ_X)) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = static_cast<int>(alignSize(tmp, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
			if (m_group_y > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_z = 1;
		}

		tensor* col2vol::forward(tensor* x)
		{
			m_input.push_back(x->getId());

			if (m_pipeline == nullptr)
			{
				const int depth = x->getShape()[x->getShape().size() - 3];
				const int height = x->getShape()[x->getShape().size() - 2];
				const int width = x->getShape()[x->getShape().size() - 3];
				m_param.batchsize = 1;
				m_param.depth_col = depth;
				m_param.height_col = height;
				m_param.width_col = width;
				m_param.depth_vol = (depth - 1) * m_param.stride_d - 2 * m_param.pad_d + m_param.dilation_d * (m_param.kernel_d - 1) + m_param.pad_d + 1;
				m_param.height_vol = (height - 1) * m_param.stride_h - 2 * m_param.pad_h + m_param.dilation_h * (m_param.kernel_h - 1) + m_param.pad_h + 1;
				m_param.width_vol = (depth - 1) * m_param.stride_w - 2 * m_param.pad_w + m_param.dilation_w * (m_param.kernel_w - 1) + m_param.pad_w + 1;
				computeGroupCount();
				createShaderModule(shaders::col2vol_spv, sizeof(shaders::col2vol_spv));
				createPipeline(sizeof(vol2col_param));
			}
			int n_out_plane = x->getShape()[0] * (m_param.kernel_d * m_param.kernel_h * m_param.kernel_w);
			int output_length = m_param.depth_vol * m_param.height_vol * m_param.width_vol;
			auto* y = new tensor(0.0, std::vector<int>{n_out_plane* (m_param.depth_vol* m_param.height_vol* m_param.width_vol) });
			m_output.push_back(y->getId());
			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);

			recordCommandBuffer(static_cast<void*>(&m_param), sizeof(vol2col_param));
			layers.push_back(this);
			y->reshape(std::vector<int>{n_out_plane, output_length});
			return y;
		}

		std::vector<int> col2vol::output_shape() const
		{
			return std::vector<int> {m_param.depth_vol, m_param.height_vol, m_param.width_vol};
		}
	}
}
