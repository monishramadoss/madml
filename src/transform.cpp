#include "common.h"
#include "utils.h"
#include "transform.h"
#include <algorithm>

#define LOCAL_SZ_X 16
#define LOCAL_SZ_Y 64
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct im2colParam {
            int batchsize;
            int channels;
            int kernel_h;
            int kernel_w;
            int pad_h;
            int pad_w;
            int stride_h;
            int stride_w;
            int dilation_h;
            int dilation_w;

            int height_col; // height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
            int width_col;  // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
            int height_im;
            int width_im;
		};
		
        im2col::im2col(int batchsize, int channels, int kernel[2], int pad[2], int stride[2], int dilation[2]):
            batchsize(batchsize), channels(channels), kernel_h(kernel[0]), kernel_w(kernel[1]), pad_h(pad[0]), pad_w(pad[1]), stride_h(stride[0]), stride_w(stride[1]), dilation_h(dilation[0]), dilation_w(dilation[1])
        {
            initVulkanThing(2);
            m_type = "im2col";

        }

        bool im2col::forward(tensor& x, tensor& y) {
            if (m_pipeline == VK_NULL_HANDLE) {
                height_im = x.getShape()[x.getShape().size() - 1];
                width_im = x.getShape()[x.getShape().size() - 2];
                height_col = (height_im + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
                width_col = (width_im + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
                computeGroupCount();
                createShaderModule(shaders::im2col_spv, sizeof(shaders::im2col_spv));
                createPipeline(sizeof(im2colParam));
            }

            bindTensor(m_device, x, 0, m_descriptor_set);
            bindTensor(m_device, y, 1, m_descriptor_set);
            im2colParam param = { batchsize, channels, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, height_im, width_im };
            recordCommandBuffer((void*)&param, sizeof(im2colParam));
            return true;
        }

        bool im2col::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
            return forward(ins[0], outs[0]);
        }

        bool im2col::computeGroupCount() {
            m_group_x = (int)alignSize(batchsize, LOCAL_SZ_X) / LOCAL_SZ_X;
            if (m_group_x > maxComputeWorkGroupCount)
                m_group_x = maxComputeWorkGroupCount;
            m_group_x = (int)alignSize(channels * kernel_h * kernel_w, LOCAL_SZ_X) / LOCAL_SZ_X;
            if (m_group_x > maxComputeWorkGroupCount)
                m_group_x = maxComputeWorkGroupCount;
            m_group_z = 1;
            return true;
        }

        col2im::col2im(int batchsize, int channels, int kernel[2], int pad[2], int stride[2], int dilation[2]) :
            batchsize(batchsize), channels(channels), kernel_h(kernel[0]), kernel_w(kernel[1]), pad_h(pad[0]), pad_w(pad[1]), stride_h(stride[0]), stride_w(stride[1]), dilation_h(dilation[0]), dilation_w(dilation[1])
        {
            initVulkanThing(2);
            m_type = "col2im";
        }

        bool col2im::forward(tensor& x, tensor& y) {
            if (m_pipeline == VK_NULL_HANDLE) {
                height_im = x.getShape()[x.getShape().size() - 1];
                width_im = x.getShape()[x.getShape().size() - 2];
                height_col = (height_im + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
                width_col = (width_im + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
                computeGroupCount();
                createShaderModule(shaders::col2im_spv, sizeof(shaders::col2im_spv));
                createPipeline(sizeof(im2colParam));
            }

            bindTensor(m_device, x, 0, m_descriptor_set);
            bindTensor(m_device, y, 1, m_descriptor_set);
            im2colParam param = { batchsize, channels, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, height_im, width_im };
            recordCommandBuffer((void*)&param, sizeof(im2colParam));
            return true;
        }

        bool col2im::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
            return forward(ins[0], outs[0]);
        }

        bool col2im::computeGroupCount() {
            m_group_x = (int)alignSize(batchsize, LOCAL_SZ_X) / LOCAL_SZ_X;
            if (m_group_x > maxComputeWorkGroupCount)
                m_group_x = maxComputeWorkGroupCount;
            m_group_x = (int)alignSize(channels * kernel_h * kernel_w, LOCAL_SZ_X) / LOCAL_SZ_X;
            if (m_group_x > maxComputeWorkGroupCount)
                m_group_x = maxComputeWorkGroupCount;
            m_group_z = 1;
            return true;
        }

	}
}
