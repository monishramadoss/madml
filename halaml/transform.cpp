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
		struct vol2colParam
		{
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
			int width_col; // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
			int depth_col; // depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1

			int height_im;
			int width_im;
			int depth_im;
		};

		std::vector<Module*>* vol2col::get_module()
		{
			return &Module::module_list;
		}

		vol2col::vol2col(int channels, int kernel[3], int pad[3], int stride[3], int dilation[3]) :
			channels(channels),
			kernel_h(kernel[0]), kernel_w(kernel[1]), kernel_d(kernel[2]),
			pad_h(pad[0]), pad_w(pad[1]), pad_d(pad[2]),
			stride_h(stride[0]), stride_w(stride[1]), stride_d(stride[2]),
			dilation_h(dilation[0]), dilation_w(dilation[1]), dilation_d(dilation[2])
		{
			initVulkanThing(2);
			m_type = "vol2col";
		}

		void vol2col::reshapeOutTensor(tensor* x, tensor* z)
		{
			Shape shape = x->getShape();
			z = &(z->reshape(nullptr, shape));
		}

		bool vol2col::forward(tensor* x, tensor* y)
		{
			if (m_pipeline == nullptr)
			{
				batch_size = 1;
				height_im = x->getShape()[x->getShape().size() - 1];
				width_im = x->getShape()[x->getShape().size() - 2];
				depth_im = x->getShape()[x->getShape().size() - 3];
				height_col = (height_im + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
				width_col = (width_im + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
				depth_col = (depth_im + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;
				computeGroupCount();
				createShaderModule(shaders::vol2col_spv, sizeof(shaders::vol2col_spv));
				createPipeline(sizeof(vol2colParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);
			vol2colParam param = {
				1, channels,
				kernel_h, kernel_w, kernel_d,
				pad_h, pad_w, pad_d,
				stride_h, stride_w, stride_d,
				dilation_h, dilation_w, dilation_d,
				height_col, width_col, depth_col,
				height_im, width_im, depth_im
			};

			recordCommandBuffer(static_cast<void*>(&param), sizeof(vol2colParam));
			return true;
		}

		bool vol2col::forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs)
		{
			return forward(ins[0], outs[0]);
		}

		bool vol2col::computeGroupCount()
		{
			size_t tmp = channels;
			tmp *= kernel_h;
			tmp *= kernel_w;
			tmp *= kernel_d;

			m_group_x = static_cast<int>(alignSize(1, LOCAL_SZ_X)) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_x = static_cast<int>(alignSize(tmp, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_z = 1;
			return true;
		}

		std::vector<Module*>* col2vol::get_module()
		{
			return &Module::module_list;
		}

		col2vol::col2vol(int channels, int kernel[3], int pad[3], int stride[3], int dilation[3]) :
			channels(channels),
			kernel_h(kernel[0]), kernel_w(kernel[1]), kernel_d(kernel[2]),
			pad_h(pad[0]), pad_w(pad[1]), pad_d(pad[2]),
			stride_h(stride[0]), stride_w(stride[1]), stride_d(stride[2]),
			dilation_h(dilation[0]), dilation_w(dilation[1]), dilation_d(dilation[2])
		{
			initVulkanThing(2);
			m_type = "col2vol";
		}

		void col2vol::reshapeOutTensor(tensor* x, tensor* z)
		{
			Shape shape = x->getShape();
			z = &(z->reshape(nullptr, shape));
		}

		bool col2vol::forward(tensor* x, tensor* y)
		{
			if (m_pipeline == nullptr)
			{
				batch_size = 1;
				height_im = x->getShape()[x->getShape().size() - 1];
				width_im = x->getShape()[x->getShape().size() - 2];
				depth_im = x->getShape()[x->getShape().size() - 3];
				height_col = (height_im + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
				width_col = (width_im + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
				depth_col = (depth_im + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;
				computeGroupCount();
				createShaderModule(shaders::col2vol_spv, sizeof(shaders::col2vol_spv));
				createPipeline(sizeof(vol2colParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);

			vol2colParam param = {
				batch_size, channels,
				kernel_h, kernel_w, kernel_d,
				pad_h, pad_w, pad_d,
				stride_h, stride_w, stride_d,
				dilation_h, dilation_w, dilation_d,
				height_col, width_col, depth_col,
				height_im, width_im, depth_im
			};

			recordCommandBuffer(static_cast<void*>(&param), sizeof(vol2colParam));
			return true;
		}

		bool col2vol::forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs)
		{
			return forward(ins[0], outs[0]);
		}

		bool col2vol::computeGroupCount()
		{
			size_t tmp = channels;
			tmp *= kernel_h;
			tmp *= kernel_w;
			tmp *= kernel_d;

			m_group_x = static_cast<int>(alignSize(1, LOCAL_SZ_X)) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_x = static_cast<int>(alignSize(tmp, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_z = 1;
			return true;
		}
	}
}
