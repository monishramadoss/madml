#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "madml.h"
#include "layer.h"

namespace kernel {
	namespace layers {
		class convolution : public layer
		{
		public:
			convolution(int kernel_size, int stride, int padding, int dialation);
			convolution(int kernel_size_x, int kernel_size_y, int stride_x, int stride_y, int padding_x, int padding, int dialation_x, int dialation_y);
			convolution(int kernel_size_x, int kernel_size_y, int kernel_size_z, int stride_x, int stride_y, int stride_z, int padding_x, int padding, int padding_z, int dialation_x, int dialation_y, int dialation_z);

			void output_shape(std::vector<int> out_shape, int input_x);

			bool forward(tensor& x, tensor& y, tensor& z);
			bool forward(tensor& x, tensor& y);
			void reshapeOutTensor(tensor& x, tensor& z);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int dim;
			int batch_size;
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
			int padding_type;
		};

	}
}

#endif
