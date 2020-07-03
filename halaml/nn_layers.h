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

			class RNN : public Module
			{
			public:
				RNN(int vocab_size, int hidden_size, int num_layers = 1, float dropout = 0.9, bool bidirectional = false, bool bias = false, std::string nonlinearity = "tanh");
				std::tuple<tensor*, tensor*> forward(tensor* x, tensor* h);
				void update_weight() override {};

			private:
				int m_vocab_size, m_hidden_size, m_num_layers, m_directions;
				bool USE_BIAS;
				matmul* ih_mm;
				matmul* hh_mm;
				matmul* oh_mm;
				math::add* bias;
				math::add* ht_add;
				math::tanh* tanh;
				math::exp* softmax;
			};

			struct RNN_cell_param {
				int vocab_size;
				int hidden_size;
				int num_layers;
			};
			
			class RNNCell : public layer, public Module {
			private:
				void computeGroupCount() override;
				RNN_cell_param m_param;
			public:
				RNNCell(int vocab_size, hidden_size, num_layers);
				tensor* forward(tensor* x, tensor* h, tensor* y);
				virtual void update_weight() override {};
			};
			/*
			class LSTM : public Module
			{
				LSTM(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
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

			class LSTMCell : public layer, public Module {
			private:
				void computeGroupCount() override;
				RNN_cell_param m_param;
			public:
				LSTMCell();
				tensor* forward(tensor* x, tensor* h, tensor* y);
				virtual void update_weight() override {};
			};

			class LSTM : public Module
			{
			public:
				LSTM(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
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
			};
			
			class GRUCell : public layer, public Module {
			private:
				void computeGroupCount() override;
				RNN_cell_param m_param;
			public:
				GRUCell();
				tensor* forward(tensor* x, tensor* h, tensor* y);
				virtual void update_weight() override {};
			};

			*/
		}
	}
}

#endif //!NN
