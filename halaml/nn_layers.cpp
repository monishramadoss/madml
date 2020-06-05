#include "common.h"
#include "utils.h"
#include "nn_layers.h"
#include <numeric>

namespace kernel {
	namespace layers {
		namespace nn {
			dense::dense(int size, bool bias, bool debug) : size(size), bias(bias) {
				mul_op = new matmul();
				layers.push_back(mul_op);
				if (bias) {
					add_op = new operators(0);
					layers.push_back(add_op);
				}
				
				m_weight = nullptr;
				m_bias = nullptr;
				m_output = nullptr;
				input_tensor = nullptr;
				weight_tensor = nullptr;
				bias_tensor = nullptr;
				output_tensor = nullptr;
				m_debug = debug;

			}
			bool dense::operator()(tensor* x, tensor* y) {
				std::vector<layer*> layers;
				std::vector<tensor*> tensors;
				tensors.push_back(x);

				auto in_shape = x->getShape();
				
				if (m_weight == nullptr) {
					auto weight_shape = std::vector<int>{ in_shape[1], size };
					m_weight = fill_memory_shape<float>(weight_shape, 1);
					weight_tensor = new tensor(m_weight, weight_shape, kFormatFp32);
					tensors.push_back(weight_tensor);
				}

				if (m_bias == nullptr && bias) {
					auto bias_shape = std::vector<int>{ in_shape[0], size };
					m_bias = fill_memory_shape<float>(bias_shape, 1);
					bias_tensor = new tensor(m_bias, bias_shape, kFormatFp32);
					tensors.push_back(bias_tensor);
				}
				if (m_output == nullptr) {
					auto output_shape = std::vector<int>{ in_shape[0], size };
					m_output = fill_memory_shape<float>(output_shape, 0);
					output_tensor = new tensor(m_output, output_shape, kFormatFp32);
					tensors.push_back(output_tensor);
					*y = *output_tensor;
				}
				mul_op->forward(x, weight_tensor, y);
				if (bias) {
					add_op->forward(output_tensor, bias_tensor, output_tensor);
				}
				
				if (m_debug) {
					mul_op->run();
					if (bias)
						add_op->run();
					for(int i = 0; i < y->count(); ++i)
						std::cout << ((float*)y->toHost())[i] << " ";
					std::cout << std::endl;

				
				}				
				return true;
			}

		

			void dense::backward() {

			}

		}
	}
}