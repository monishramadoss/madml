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
				std::shared_ptr<tensor>& forward(const std::shared_ptr<tensor>& x);
				void fwd_callback() override;
				void backward() override;
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
				std::shared_ptr<tensor>& forward(const std::shared_ptr<tensor>& x);
				void fwd_callback() override;
				void backward() override;
				void update_weight() override
				{
				};

			private:
				std::shared_ptr<tensor> ir_vol2col;
				int m_num_filters;
				dhw m_kernel_size, m_stride, m_padding, m_dilation;
				bool USE_BIAS;
			};

			class convTranspose : public Module
			{
			public:
				convTranspose(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
					bool use_bias);
				std::shared_ptr<tensor>& forward(const std::shared_ptr<tensor>& x);
				void fwd_callback() override;
				void backward() override;
				void update_weight() override
				{
				};

			private:
				std::shared_ptr<tensor> ir_col2vol;
				int m_num_filters;
				dhw m_kernel_size, m_stride, m_padding, m_dilation;
				bool USE_BIAS;
			};

			class RNN : public Module
			{
			public:
				RNN(int vocab_size, int hidden_size, int num_layers = 1, int seq_length = 16, bool bidirectional = false,
					int output_size = 0, float dropout = 0.9, bool bias = false, std::string nonlinearity = "tanh");
				std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> forward(const std::shared_ptr<tensor>& x);
				std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> forward(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h);
				void fwd_callback() override;
				void backward() override;
				void update_weight() override
				{
				};

			private:
				std::shared_ptr<tensor> x, h;
				int m_vocab_size, m_hidden_size, m_num_layers, m_directions;
				int m_output_size, m_seq_length;
				bool USE_BIAS, bidirectional{};
				std::vector<rnn::RNNCell*> cells;
				std::vector<std::shared_ptr<tensor>> weights_biases;
				std::vector<std::shared_ptr<tensor>> cache;
			};

			class LSTM : public Module
			{
			public:
				LSTM(int vocab_size, int hidden_size, int num_layers = 1, int seq_length = 16, bool bidirectional = false,
					int output_size = 0, float dropout = 0.9, bool bias = false, std::string nonlinearity = "tanh");
				std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> forward(const std::shared_ptr<tensor>& x);
				std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> forward(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h, const std::shared_ptr<tensor>& c);
				void fwd_callback() override;
				void backward() override;
				void update_weight() override
				{
				};
			private:
				std::shared_ptr<tensor> x, h, c;
				int m_vocab_size, m_hidden_size, m_num_layers, m_directions;
				int m_output_size, m_seq_length;
				bool USE_BIAS, bidirectional{};
				std::vector<rnn::LSTMCell*> cells;
				std::vector<std::shared_ptr<tensor>> weights_biases;
				std::vector<std::shared_ptr<tensor>> cache;
				std::string nonlinearity_;
			};

			class GRU : public Module
			{
			public:
				GRU(int vocab_size, int hidden_size, int num_layers = 1, int seq_length = 16, bool bidirectional = false,
					int output_size = 0, float dropout = 0.9, bool bias = false, std::string nonlinearity = "tanh");
				std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> forward(const std::shared_ptr<tensor>& x);
				std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> forward(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h);
				void fwd_callback() override;
				void backward() override;
				void update_weight() override
				{
				};
			private:
				std::shared_ptr<tensor> x, h;
				int m_vocab_size, m_hidden_size, m_num_layers, m_directions;
				int m_output_size, m_seq_length;
				bool USE_BIAS, bidirectional{};
				std::vector<rnn::GRUCell*> cells;
				std::vector<std::shared_ptr<tensor>> weights_biases;
				std::vector<std::shared_ptr<tensor>> cache;
				std::string nonlinearity_;
			};
		}
	}
}

#endif //!NN
