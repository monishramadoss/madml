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
  
            int height_im;
            int width_im;
            int depth_im;

		};
		
        im2col::im2col( int channels, int kernel[3], int pad[3], int stride[3], int dilation[3]) :
            batchsize(1), channels(channels),
            kernel_h(kernel[0]), kernel_w(kernel[1]), kernel_d(kernel[2]),
            pad_h(pad[0]), pad_w(pad[1]), pad_d(pad[2]),
            stride_h(stride[0]), stride_w(stride[1]), stride_d(stride[2]),
            dilation_h(dilation[0]), dilation_w(dilation[1]), dilation_d(dilation[2])
        {
            initVulkanThing(2);
            m_type = "im2col";

        }

        void im2col::reshapeOutTensor(tensor* x, tensor* z) {
            Shape shape = x->getShape();
            z = &(z->reshape(nullptr, shape));
        }

        bool im2col::forward(tensor* x, tensor* y) {
            if (m_pipeline == VK_NULL_HANDLE) {
                height_im = x->getShape()[x->getShape().size() - 1];
                width_im = x->getShape()[x->getShape().size() - 2];
                depth_im = x->getShape()[x->getShape().size() - 3];
                height_col = (height_im + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
                width_col = (width_im + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
                depth_col = (depth_im + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;

                computeGroupCount();
                createShaderModule(shaders::im2col_spv, sizeof(shaders::im2col_spv));
                createPipeline(sizeof(im2colParam));
            }

            bindTensor(m_device, x, 0, m_descriptor_set);
            bindTensor(m_device, y, 1, m_descriptor_set);
            im2colParam param = {   
                                    batchsize, channels,
                                    kernel_h, kernel_w, kernel_d,
                                    pad_h, pad_w, pad_d,
                                    stride_h, stride_w, stride_d,
                                    dilation_h, dilation_w, dilation_d,
                                    height_col, width_col, depth_col,
                                    height_im, width_im, depth_im
                                };

            recordCommandBuffer((void*)&param, sizeof(im2colParam));
            return true;
        }

        bool im2col::forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) {
            return forward(ins[0], outs[0]);
        }

        bool im2col::computeGroupCount() {
            m_group_x = (int)alignSize(batchsize, LOCAL_SZ_X) / LOCAL_SZ_X;
            if (m_group_x > maxComputeWorkGroupCount)
                m_group_x = maxComputeWorkGroupCount;
            m_group_x = (int)alignSize(channels * kernel_h * kernel_w * kernel_d, LOCAL_SZ_Y) / LOCAL_SZ_Y;
            if (m_group_x > maxComputeWorkGroupCount)
                m_group_x = maxComputeWorkGroupCount;
            m_group_z = 1;
            return true;
        }

        col2im::col2im(int channels, int kernel[3], int pad[3], int stride[3], int dilation[3]):
            batchsize(1), channels(channels),
            kernel_h(kernel[0]), kernel_w(kernel[1]), kernel_d(kernel[2]),
            pad_h(pad[0]), pad_w(pad[1]), pad_d(pad[2]),
            stride_h(stride[0]), stride_w(stride[1]), stride_d(stride[2]),
            dilation_h(dilation[0]), dilation_w(dilation[1]), dilation_d(dilation[2])   
        {
            initVulkanThing(2);
            m_type = "col2im";
        }

        void col2im::reshapeOutTensor(tensor* x, tensor* z) {
            Shape shape = x->getShape();
            z = &(z->reshape(nullptr, shape));
        }

        bool col2im::forward(tensor* x, tensor* y) {
            if (m_pipeline == VK_NULL_HANDLE) {
                height_im = x->getShape()[x->getShape().size() - 1];
                width_im = x->getShape()[x->getShape().size() - 2];
                depth_im = x->getShape()[x->getShape().size() - 3];
                height_col = (height_im + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
                width_col = (width_im + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
                depth_col = (depth_im + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;
                computeGroupCount();
                createShaderModule(shaders::col2im_spv, sizeof(shaders::col2im_spv));
                createPipeline(sizeof(im2colParam));
            }

            bindTensor(m_device, x, 0, m_descriptor_set);
            bindTensor(m_device, y, 1, m_descriptor_set);

            im2colParam param = {   
                                    batchsize, channels,
                                    kernel_h, kernel_w, kernel_d,
                                    pad_h, pad_w, pad_d,
                                    stride_h, stride_w, stride_d,
                                    dilation_h, dilation_w, dilation_d,
                                    height_col, width_col, depth_col,
                                    height_im, width_im, depth_im
                                };

            recordCommandBuffer((void*)&param, sizeof(im2colParam));
            return true;
        }

        bool col2im::forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) {
            return forward(ins[0], outs[0]);
        }

        bool col2im::computeGroupCount() {
            m_group_x = (int)alignSize(batchsize, LOCAL_SZ_X) / LOCAL_SZ_X;
            if (m_group_x > maxComputeWorkGroupCount)
                m_group_x = maxComputeWorkGroupCount;
            m_group_x = (int)alignSize((int)(channels * kernel_h * kernel_w * kernel_d), LOCAL_SZ_Y) / LOCAL_SZ_Y;
            if (m_group_x > maxComputeWorkGroupCount)
                m_group_x = maxComputeWorkGroupCount;
            m_group_z = 1;
            return true;
        }

	}
}
