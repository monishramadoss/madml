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
				kernel = new layers::convolution(kernel_size, stride, padding, dialation);
				std::vector<int> weight_shape{ num_filters, x[0]->getShape()[1], kernel_size};
				filter_size = std::accumulate(std::begin(weight_shape), std::end(weight_shape), 1, std::multiplies<int>());

			}
			if (x[0]->getShape().size() == 4) {
				kernel = new layers::convolution(kernel_size, kernel_size, stride, stride, padding, padding, dialation, dialation);
				std::vector<int> weight_shape{ num_filters, x[0]->getShape()[1], kernel_size, kernel_size };
				filter_size = std::accumulate(std::begin(weight_shape), std::end(weight_shape), 1, std::multiplies<int>());
			}
			if (x[0]->getShape().size() == 5) {
				kernel = new layers::convolution(kernel_size, kernel_size, kernel_size, stride, stride, stride, padding, padding, padding, dialation, dialation, dialation);
				std::vector<int> weight_shape{ num_filters, x[0]->getShape()[1], kernel_size, kernel_size, kernel_size };
				filter_size = std::accumulate(std::begin(weight_shape), std::end(weight_shape), 1, std::multiplies<int>());
			}

			layers.push_back(kernel);

			return x;

		}

		void conv::backward() {

		}


	}

}