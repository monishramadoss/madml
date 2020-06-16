#ifndef NN
#define NN

#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel {
	namespace layers{
		namespace nn {
			
			class conv : public Module {
				int kernel_size, num_filters, stride, padding, dialation, padding_type;
			public:
				conv(int kernel_size, int num_filters, bool bias, int stride, int padding, int dialation, int padding_type);

				bool operator()(tensor* x, tensor* y);
				void backward();
				void update_weight();
			};

			class dense : public Module
			{
			public:
				dense(int size, bool bias, bool debug=false);
				bool operator()(tensor* x, tensor* y);
				void backward();
				void update_weight();
				~dense() {
					if (m_weight != nullptr)
						delete[] m_weight;
					if (m_bias != nullptr)
						delete[] m_bias;
					
					if (weight_tensor != nullptr)
						delete weight_tensor;
					if (bias_tensor != nullptr)
						delete bias_tensor;

					layers.clear();
					tensors.clear();

					delete mul_op;
					delete add_op;
				}
			private:
				int size; bool bias;
				bool m_debug;
				char* m_weight;
				char* m_bias;
				char* m_output;
				tensor* input_tensor;
				tensor* weight_tensor;
				tensor* bias_tensor;
				tensor* output_tensor;
				layers::matmul* mul_op;
				layers::operators* add_op;

				static std::vector<Module*> module_list;
				virtual std::vector<Module*>* get_module();
			};

			class RNN : public Module {

			public:
				RNN(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
				bool operator()(tensor* x, tensor* y);
				void backward();
				void update_weight();

			};

			class LSTM : public Module {

			public:
				LSTM(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
				bool operator()(tensor* x, tensor* y);
				void backward();
				void update_weight();

			};

			class GRU : public Module {

			public:
				GRU(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
				bool operator()(tensor* x, tensor* y);
				void backward();
				void update_weight();
			};
		}
	}

}





#endif //!NN