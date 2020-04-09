#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "madml.h"
#include "layer.h"

namespace kernel {
	namespace layers {
		class convolution : public layer
		{
		public:
			convolution(int kernel_size, int stride, int padding, int dialation=1);
			convolution(int kernel_size_x, int kernel_size_y, int stride_x, int stride_y, int padding_x=0, int padding=0, int dialation_x = 1, int dialation_y = 1);
			convolution(int kernel_size_x, int kernel_size_y, int kernel_size_z, int stride_x, int stride_y, int stride_z, int padding_x = 0, int padding = 0, int padding_z=0, int dialation_x = 1, int dialation_y = 1, int dialation_z = 1);

			bool forward(tensor& x, tensor& y, tensor& z);
			bool forward(tensor& x, tensor& y);
			void reshapeOutTensor(tensor& x, tensor& z);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int dim;
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
