#include "common.h"
#include "utils.h"
#include "nn_layers.h"
#include <numeric>

namespace kernel {

	namespace nn {
		conv::conv(int kernel_size, int num_filters, bool bias, int stride, int padding, int dialation, int padding_type) {
			this->kernel_size = kernel_size;
			this->num_filters = num_filters;
			this->stride = stride;
			this->padding = padding;
			this->dialation = dialation;
			this->padding_type = padding_type;
		}

		std::vector<tensor*> conv::forward(std::vector<tensor*> x) {
			layer* kernel;
			int filter_size;
			if (x[0]->getShape().size() == 3) {
				std::vector<int> weight_shape{ num_filters, x[0]->getShape()[1], kernel_size};
				filter_size = std::accumulate(std::begin(weight_shape), std::end(weight_shape), 1, std::multiplies<int>());

			}
			if (x[0]->getShape().size() == 4) {
				std::vector<int> weight_shape{ num_filters, x[0]->getShape()[1], kernel_size, kernel_size };
				filter_size = std::accumulate(std::begin(weight_shape), std::end(weight_shape), 1, std::multiplies<int>());
			}
			if (x[0]->getShape().size() == 5) {
				std::vector<int> weight_shape{ num_filters, x[0]->getShape()[1], kernel_size, kernel_size, kernel_size };
				filter_size = std::accumulate(std::begin(weight_shape), std::end(weight_shape), 1, std::multiplies<int>());
			}

			layers.push_back(kernel);

			return x;

		}

		void conv::backward() {

		}


	}

	namespace nn {
		dense::dense(int size, bool bias) {
			this->size = size;
			this->bias = bias;
		}

		std::vector<tensor*> dense::forward(std::vector<tensor*> x) {
			float* data = new float[x[0]->getShape()[0] * size];
			float* k_data = new float[x[0]->getShape()[1] * size];
			for (int i = 0; i < x[0]->getShape()[0] * size; ++i)
				data[i] = 0;
			for (int i = 0; i < x[0]->getShape()[1] * size; ++i)
				k_data[i] = 0;

			std::vector<int> output_shape { x[0]->getShape()[0], size };
			output_tensor = new tensor((char*)data, output_shape, kFormatFp32);
			
			if (bias || !bias_tensor || !add_op) {
				add_op = new layers::operators(0);
				bias_tensor = new tensor((char*)data, output_shape, kFormatFp32);
			}	

		    mul_op = new layers::matmul();
			std::vector<int> weight_shape{ x[0]->getShape()[1], size };
			dense_tensor = new tensor((char*)k_data, weight_shape, kFormatFp32);

			mul_op->forward(*x[0], *dense_tensor, *output_tensor);

			if (bias)
				add_op->forward(*output_tensor, *bias_tensor, *output_tensor);
			
			return std::vector<tensor*> {output_tensor};
				
		}

		void dense::backward() {
			
		}

	}

}