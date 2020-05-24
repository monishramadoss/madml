#include "common.h"
#include "utils.h"
#include "im2col.h"
#include <algorithm>

#define LOCAL_SZ_X 256
#define LOCAL_SZ_Y 1
#define maxComputeWorkGroupCount 65535



namespace kernel {
	namespace layers {
        struct im2colParam {

        };


        im2col::im2col(){
            layer::initVulkanThing(2);
            m_type = "im2col";
        }

        void im2col::reshapeOutTensor(tensor& x, tensor& z) {
			Shape shape = x.getShape();
			z = z.reshape(nullptr, shape);
		}

        bool im2col::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

        bool im2col::forward(tensor& x, tensor& y){
            if (m_pipeline == VK_NULL_HANDLE) {
                
                computeGroupCount();
				createShaderModule(shaders::im2col_spv, sizeof(shaders::im2col_spv));
				createPipeline(sizeof(im2colParam));
            }

            bindTensor(m_device, x, 0, m_descriptor_set);
            bindTensor(m_device, y, 1, m_descriptor_set);

        }

        bool matmul::computeGroupCount() {
			m_group_x = (int)alignSize(m_m, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = (int)alignSize(m_n, LOCAL_SZ_Y) / LOCAL_SZ_Y;
			if (m_group_y > maxComputeWorkGroupCount)
				m_group_y = maxComputeWorkGroupCount;
			m_group_z = 1;
			return true;
		}

    }
}



namespace kernel {
	namespace layers {

        struct col2imParam {
            int n;
            int num_axes;
            int im_shape;
            int col_shape;
            int kernel_shape;
            int pad;
            int stride;
            int dilation;
        };

        col2im::col2im(){
            layer::initVulkanThing(2);
            m_type = "col2im";
        }

        void col2im::reshapeOutTensor(tensor& x, tensor& z) {
			Shape shape = x.getShape();
			z = z.reshape(nullptr, shape);
		}

        bool col2im::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

        bool col2im::forward(tensor& x, tensor& y){
            if (m_pipeline == VK_NULL_HANDLE) {                
                computeGroupCount();
				createShaderModule(shaders::im2col_spv, sizeof(shaders::im2col_spv));
				createPipeline(sizeof(col2imParam));
            }

            bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);	

           

        }
        
        bool col2im::computeGroupCount() {
			m_group_x = (int)alignSize(120, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = (int)alignSize(120, LOCAL_SZ_Y) / LOCAL_SZ_Y;
			if (m_group_y > maxComputeWorkGroupCount)
				m_group_y = maxComputeWorkGroupCount;
			m_group_z = 1;
			return true;
		}

    }
}