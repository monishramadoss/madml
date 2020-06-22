#ifndef NN_H
#define NN_H

#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel
{
	namespace layers
	{
		class gradient : public layer, public Module
		{
		public:
			gradient(float lr);
			bool forward(tensor* x, tensor* y, tensor* z);
			void reshapeOutTensor(tensor* x, tensor* z);
			bool forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) override;

			void backward() override
			{
			}

			void update_weight() override
			{
			}

			bool operator()(tensor* x, tensor* y) override { return forward(x, y, x); };

		private:
			float m_lr;
			int m_total;
			bool computeGroupCount();

			static std::vector<Module*> module_list;
			std::vector<Module*>* get_module() override;
		};

		namespace nn
		{
			class conv : public Module
			{
			public:
				conv(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
					bool use_bias);
				bool operator()(tensor* x, tensor* y) override;

				void backward() override
				{
				};
				void backward(tensor* d_output, tensor* d_input);
				void update_weight() override;

			private:
				dhw m_kernel_size, m_stride, m_padding, m_dilation;
				int m_padding_type, m_num_filters;
				bool USE_BIAS;

				tensor* m_input;
				tensor* m_weight;
				tensor* m_bias;
				tensor* m_output;
				tensor* d_weight;
				tensor* d_bias;

				tensor* m_input_t;
				vol2col* m_kernel;
				matmul* m_mm;
				operators* m_bias_op;

				static std::vector<Module*> module_list;
				std::vector<Module*>* get_module() override;
			};

			class convTranspose : public Module
			{
			public:
				convTranspose(int num_filters, int* kernel_size, int* stride, int* padding, int* dilation,
					int padding_type, bool use_bias);
				bool operator()(tensor* x, tensor* y) override;

				void backward() override
				{
				};
				void backward(tensor* d_output, tensor* d_input);
				void update_weight() override;

			private:
				int m_kernel_size[3], m_stride[3], m_padding[3], m_dilation[3], m_padding_type, m_num_filters;
				bool USE_BIAS;

				tensor* m_input;
				tensor* m_weight;
				tensor* m_bias;
				tensor* m_output;
				tensor* d_weight;
				tensor* d_bias;

				tensor* m_input_t;
				col2vol* m_kernel;
				matmul* m_mm;
				operators* m_bias_op;

				static std::vector<Module*> module_list;
				std::vector<Module*>* get_module() override;
			};

			class dense : public Module
			{
			public:
				dense(int size, bool use_bias);
				bool operator()(tensor* x, tensor* y) override;

				void backward() override
				{
				};
				void backward(tensor* d_output, tensor* d_input);
				void update_weight() override;

			private:
				int size;
				bool USE_BIAS;

				tensor* m_input;
				tensor* m_weight;
				tensor* m_bias;
				tensor* m_output;
				tensor* d_weight;
				tensor* d_bias;

				matmul* m_mm;
				operators* m_bias_op;

				static std::vector<Module*> module_list;
				std::vector<Module*>* get_module() override;
			};

			class RNNCell : public Module
			{
			public:
				RNNCell(int vocab_size, int hidden_size, int num_layers = 1, float dropout = 0.9, bool bidirectional = false, bool bias = false, std::string nonlinearity = "tanh");
				bool operator()(tensor* x, tensor* y) override { return false; };
				bool operator()(tensor* x, tensor* h, tensor* y);
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

				tensor* m_WIH;
				tensor* m_WHH;
				tensor* m_bi;
				tensor* m_bh;

				tensor* m_input_hidden;
				tensor* m_hidden_hidden;

				matmul* m_input_hidden_layer;
				operators* m_input_bias_layer;
				matmul* m_hidden_hidden_layer;
				operators* m_hidden_bias_layer;

				dense* m_dense_1;
				dense* m_dense_2;

				operators* m_input_output_add_layer;

				activation::tanh* m_tanh;
				activation::relu* m_relu;
			};

			class LSTMCell : public Module
			{
			public:
				LSTMCell(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
				bool operator()(tensor* x, tensor* y) override;
				bool operator()(tensor* x, tensor* y, tensor* z);
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
			};
		}
	}
}

#endif //!NN
