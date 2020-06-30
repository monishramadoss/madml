#ifndef NN_H
#define NN_H

#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel
{
	namespace layers
	{
		namespace nn
		{
			class dense : public Module
			{
			public:
				dense(int size, bool use_bias);
				tensor* forward(tensor* x);
				void update_weight() override {};

			private:
				int m_size;
				bool USE_BIAS;
				matmul* m_mm;
				math::add* m_bias_op;
			};

			class conv : public Module
			{
			public:
				conv(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
					bool use_bias);
				tensor* forward(tensor* x);
				void update_weight() override {};

			private:
				int m_num_filters;
				dhw m_kernel_size, m_stride, m_padding, m_dilation;
				bool USE_BIAS;
				vol2col* m_kernel;
				matmul* m_mm;
				math::add* m_bias_op;
			};

			class convTranspose : public Module
			{
			public:
				convTranspose(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
					bool use_bias);
				tensor* forward(tensor* x);
				void update_weight() override {};

			private:
				int m_num_filters;
				dhw m_kernel_size, m_stride, m_padding, m_dilation;
				bool USE_BIAS;
				col2vol* m_kernel;
				matmul* m_mm;
				math::add* m_bias_op;
			};

			/*class RNNCell : public Module
			{
			public:
				RNNCell(int vocab_size, int hidden_size, int num_layers = 1, float dropout = 0.9, bool bidirectional = false, bool bias = false, std::string nonlinearity = "tanh");
				bool operator()(tensor* x, tensor* y) override { return false; };
				bool operator()(tensor* x, tensor* h, tensor* y, tensor* h_prime);
				void backward() override {};
				void update_weight() override {};

			private:
				int m_vocab_size;
				int m_hidden_size;
				int m_num_layers;
				float m_dropout;
				bool m_bidirectional;
				bool USE_BIAS;

				tensor* m_input;
				tensor* m_output;

				tensor* m_d1;
				tensor* m_d2;
				tensor* m_h_t;

				dense* m_dense_1;
				dense* m_dense_2;
				dense* m_dense_3;

				operators* m_input_output_add_layer;

				activation::tanh* m_tanh;
				activation::relu* m_relu;

				activation::tanh* m_tanh_beta;
			};

			class LSTMCell : public Module
			{
			public:
				LSTMCell(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
				bool operator()(tensor* x, tensor* y) override;
				bool operator()(tensor* x, tensor* y, tensor* z, tensor* h_t);
				void backward() override;
				void update_weight() override;
			private:
				bool USE_BIAS;

				tensor* m_input;
				tensor* m_weight;
				tensor* m_bias;
				tensor* m_output;
				tensor* d_weight;
				tensor* d_bias;
			};

			class GRUCell : public Module
			{
			public:
				GRUCell(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
				bool operator()(tensor* x, tensor* y) override;
				void backward() override;
				void update_weight() override;
			private:
				bool USE_BIAS;

				tensor* m_input;
				std::vector<tensor*> m_weight;
				tensor* m_bias;
				tensor* m_output;
				tensor* d_weight;
				tensor* d_bias;
			};*/
		}
	}
}

#endif //!NN
