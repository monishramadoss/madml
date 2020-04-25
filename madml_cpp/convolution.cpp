#include "common.h"
#include "utils.h"
#include "convolution.h"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define LOCAL_SZ_Y 1024
#define LOCAL_SZ_Z 64

#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct convParam {
			int dim;
			int batch_size;
			int input_channel;
			int input_size_x;
			int input_size_y;
			int input_size_z;
			int kernel_channel;
			int kernel_size_x;
			int kernel_size_y;
			int kernel_size_z;
			int stride_x;
			int stride_y;
			int stride_z;
			int dialation_x;
			int dialation_y;
			int dialation_z;
			int padding_x;
			int padding_y;
			int padding_z;
			int output_x;
			int output_y;
			int ouptut_z;
			int padding_type;
		};

		convolution::convolution(int kernel_size, int stride, int padding, int dialation) {
			layer::initVulkanThing(3);
			m_type = "conv";

			this->kernel_size_x = kernel_size;
			this->stride_x = stride;
			this->padding_x = padding;
			this->dialation_x = dialation;

			this->kernel_size_y = 0;
			this->stride_y = 0;
			this->padding_y = 0;
			this->dialation_y = 0;

			this->kernel_size_z = 0;
			this->stride_z = 0;
			this->padding_z = 0;
			this->dialation_z = 0;

			padding_type = 0;
			dim = 1;			
		}
		convolution::convolution(int kernel_size_x, int kernel_size_y, int stride_x, int stride_y, int padding_x, int padding_y, int dialation_x, int dialation_y) {
			layer::initVulkanThing(3);
			m_type = "conv";

			this->kernel_size_x = kernel_size_x;
			this->stride_x = stride_x;
			this->padding_x = padding_x;
			this->dialation_x = dialation_x;
			
			this->kernel_size_y = kernel_size_y;
			this->stride_y = stride_y;
			this->padding_y = padding_y;
			this->dialation_y = dialation_y;

			this->kernel_size_z = 0;
			this->stride_z = 0;
			this->padding_z = 0;
			this->dialation_z = 0;

			padding_type = 0;
			dim = 2;			
		}

		convolution::convolution(int kernel_size_x, int kernel_size_y, int kernel_size_z, int stride_x, int stride_y, int stride_z, int padding_x, int padding_y, int padding_z, int dialation_x, int dialation_y, int dialation_z) {
			layer::initVulkanThing(3);
			m_type = "conv";

			this->kernel_size_x = kernel_size_x;
			this->stride_x = stride_x;
			this->padding_x = padding_x;
			this->dialation_x = dialation_x;

			this->kernel_size_y = kernel_size_y;
			this->stride_y = stride_y;
			this->padding_y = padding_y;
			this->dialation_y = dialation_y;

			this->kernel_size_z = kernel_size_z;
			this->stride_z = stride_z;
			this->padding_z = padding_z;
			this->dialation_z = dialation_z;

			padding_type = 0;
			dim = 3;
		}


		void convolution::reshapeOutTensor(tensor& x, tensor& z) {
			Shape shape = x.getShape();
			z = z.reshape(nullptr, shape);
		}

		bool convolution::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], ins[1], outs[0]);
		}

		void convolution::output_shape(std::vector<int> out_shape, int input_x) {
			out_shape.push_back(input_x + 2 * padding_x - dialation_x * (kernel_size_x - 1) - 1);

		}
		
		bool convolution::forward(tensor& x, tensor& y, tensor& z) {
			convParam param;
			int input_x, input_y, input_z, output_x, output_y, output_z;
			size_t t;
			batch_size = x.getShape()[0];
			int input_channel = x.getShape()[1];
			kernel_channel = y.getShape()[0];
			switch (dim) {
			case 1:
				t = x.getShape().size();
				input_x = x.getShape()[t - 1];
				output_x = input_x + 2 * padding_x - dialation_x * (kernel_size_x - 1) - 1;
				output_x = output_x / stride_x + 1;
				param = {1, batch_size, input_channel, input_x, 0, 0, kernel_channel, kernel_size_x, 0, 0, stride_x, 0, 0, dialation_x, 1, 1, padding_x, 0, 0, output_x, 0, 0, padding_type};
				break;
			case 2:
				t = x.getShape().size();
				input_x = x.getShape()[t - 1];
				input_y = x.getShape()[t - 2];
				output_x = input_x + 2 * padding_x - dialation_x * (kernel_size_x - 1) - 1;
				output_x = output_x / stride_x + 1;
				output_y = input_y + 2 * padding_y - dialation_y * (kernel_size_y - 1) - 1;
				output_y = output_y / stride_y + 1;
				param = { 2, batch_size, input_channel, input_x, input_y, 0, kernel_channel,kernel_size_x, kernel_size_y, 0, stride_x, stride_y, 0, dialation_x, dialation_y, 1, padding_x, padding_y, 0, output_x, output_y, 0, padding_type };
				break;
			case 3:
				t = x.getShape().size();
				input_x = x.getShape()[t - 1];
				input_y = x.getShape()[t- 2];
				input_z = x.getShape()[t - 3];
				output_x = input_x + 2 * padding_x - dialation_x * (kernel_size_x - 1) - 1;
				output_x = output_x / stride_x + 1;
				output_y = input_y + 2 * padding_y - dialation_y * (kernel_size_y - 1) - 1;
				output_y = output_y / stride_y + 1;
				output_z = input_z + 2 * padding_z - dialation_z * (kernel_size_z - 1) - 1;
				output_z = output_z / stride_z + 1;
				param = { 3,batch_size,  input_channel, input_x, input_y, input_z, kernel_channel, kernel_size_x, kernel_size_y, kernel_size_z, stride_x, stride_y, stride_z, dialation_x, dialation_y, dialation_z, padding_x, padding_y, padding_z, output_x, output_y, output_z, padding_type };
				break;
			}
	
			if (m_pipeline == VK_NULL_HANDLE) {
				computeGroupCount();			
				createShaderModule(shaders::im2col_spv, sizeof(shaders::im2col_spv));
				createPipeline(sizeof(convParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);
			bindTensor(m_device, z, 2, m_descriptor_set);
			
			recordCommandBuffer((void*)&param, sizeof(convParam));
			return true;
		}

		bool convolution::computeGroupCount() {
			m_group_x = (int)alignSize(5, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}