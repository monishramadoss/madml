#include "common.h"
#include "utils.h"
#include "nn_layers.h"
#include <numeric>

namespace kernel {
	namespace layers {
		namespace nn {
			dense::dense(int size, bool bias) : size(size), bias(bias) {
				mul_op = new matmul();
				if(bias)
					add_op = new operators(0);
				m_weight = nullptr;
				m_bias = nullptr;
				m_output = nullptr;
				weight_tensor = nullptr;
				bias_tensor = nullptr;
				output_tensor = nullptr;

			}
			bool dense::forward(tensor* x, tensor* y) {
				auto in_shape = x->getShape();

				if (m_weight == nullptr) {
					auto weight_shape = std::vector<int>{ in_shape[1], size };
					m_weight = fill_memory_shape<float>(weight_shape, 1);
					float* t = (float*)m_weight;
					weight_tensor = new tensor(m_weight, weight_shape, kFormatFp32);
				}
				if (m_bias == nullptr && bias) {
					auto bias_shape = std::vector<int>{ in_shape[0], size };
					m_bias = fill_memory_shape<float>(bias_shape, 1);
					bias_tensor = new tensor(m_bias, bias_shape, kFormatFp32);
				}
				if (m_output == nullptr) {
					auto output_shape = std::vector<int>{ in_shape[0], size };
					m_output = fill_memory_shape<float>(output_shape, 0);
					output_tensor = new tensor(m_output, output_shape, kFormatFp32);
					*y = *output_tensor;
				}
				mul_op->forward(x, weight_tensor, y);
				if (bias)
					add_op->forward(y, bias_tensor, y);

				return true;
			}

			void dense::run() {
				mul_op->run();
				if(bias)
					add_op->run();
			}

			void dense::backward() {

			}

		}
	}
}