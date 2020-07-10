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

				void update_weight() override
				{
				};

			private:
				int m_size;
				bool USE_BIAS;
			};

			class conv : public Module
			{
			public:
				conv(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type, bool use_bias);
				tensor* forward(tensor* x);

				void update_weight() override
				{
				};

			private:
				int m_num_filters;
				dhw m_kernel_size, m_stride, m_padding, m_dilation;
				bool USE_BIAS;
			};

			class convTranspose : public Module
			{
			public:
				convTranspose(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
				              bool use_bias);
				tensor* forward(tensor* x);

				void update_weight() override
				{
				};

			private:
				int m_num_filters;
				dhw m_kernel_size, m_stride, m_padding, m_dilation;
				bool USE_BIAS;
			};

			struct RNN_cell_param
			{
				int vocab_size;
				int hidden_size;
				int output_size;
				int input_offset;
				int weight_offset;
				int output_offset;
			};

			class RNNCell : public layer, public Module
			{
			private:
				void computeGroupCount() override;
				RNN_cell_param m_param;
			public:
				RNNCell(int vocab_size, int hidden_size, int output_size = 0);
				void forward(tensor* x, tensor* h, tensor* y, tensor* hn, tensor* U, tensor* W, tensor* V, tensor* b1,
				             tensor* b2, int input_offset, int weight_offset, int output_offset);

				void update_weight() override
				{
				};
			};

			class RNN : public Module
			{
			public:
				RNN(int vocab_size, int hidden_size, int num_layers = 1, int seq_length = 16, bool bidirectional = false,
				    int output_size = 0, float dropout = 0.9, bool bias = false, std::string nonlinearity = "tanh");
				std::tuple<tensor*, tensor*> forward(tensor* x);

				void update_weight() override
				{
				};

			private:
				int m_vocab_size, m_hidden_size, m_num_layers, m_directions;
				int m_output_size, m_seq_length;
				bool USE_BIAS, bidirectional{};
				std::vector<RNNCell*> cells;
				std::vector<tensor*> weights_biases;
				std::vector<tensor*> cache;
			};

			class LSTMCell : public layer, public Module
			{
			private:
				void computeGroupCount() override;
				RNN_cell_param m_param;
			public:
				LSTMCell(int vocab_size, int hidden_size, int output_size);
				void forward(tensor* x, tensor* h, tensor* c, tensor* y, tensor* hn, tensor* cn, tensor* U, tensor* W,
				             tensor* V, tensor* b1, tensor* b2, int input_offset, int weight_offset, int output_offset);

				void update_weight() override
				{
				};
			};

			class LSTM : public Module
			{
			public:
				LSTM(int vocab_size, int hidden_size, int num_layers = 1, int seq_length = 16, bool bidirectional = false,
				     int output_size = 0, float dropout = 0.9, bool bias = false, std::string nonlinearity = "tanh");
				std::tuple<tensor*, tensor*, tensor*> forward(tensor* x);

				void update_weight() override
				{
				};
			private:
				int m_vocab_size, m_hidden_size, m_num_layers, m_directions;
				int m_output_size, m_seq_length;
				bool USE_BIAS, bidirectional{};
				std::vector<LSTMCell*> cells;
				std::vector<tensor*> weights_biases;
				std::vector<tensor*> cache;
				std::string nonlinearity_;
			};

			class GRUCell : public layer, public Module
			{
			private:
				void computeGroupCount() override;
				RNN_cell_param m_param;
			public:
				GRUCell(int vocab_size, int hidden_size, int output_size);
				void GRUCell::forward(tensor* x, tensor* h, tensor* y, tensor* hn, tensor* U, tensor* W, tensor* V, tensor* b1,
				                      tensor* b2, int input_offset, int weight_offset, int output_offset);

				void update_weight() override
				{
				};
			};

			class GRU : public Module
			{
			public:
				GRU(int vocab_size, int hidden_size, int num_layers = 1, int seq_length = 16, bool bidirectional = false,
				    int output_size = 0, float dropout = 0.9, bool bias = false, std::string nonlinearity = "tanh");
				std::tuple<tensor*, tensor*> forward(tensor* x);

				void update_weight() override
				{
				};
			private:
				int m_vocab_size, m_hidden_size, m_num_layers, m_directions;
				int m_output_size, m_seq_length;
				bool USE_BIAS, bidirectional{};
				std::vector<GRUCell*> cells;
				std::vector<tensor*> weights_biases;
				std::vector<tensor*> cache;
				std::string nonlinearity_;
			};
		}
	}
}

#endif //!NN
