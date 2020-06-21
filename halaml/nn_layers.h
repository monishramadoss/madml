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
				conv(int num_filters, int* kernel_size, int* stride, int* padding, int* dialation, int padding_type,
					bool use_bias);
				bool operator()(tensor* x, tensor* y) override;

				void backward() override
				{
				};
				void backward(tensor* d_output, tensor* d_input);
				void update_weight() override;

			private:
				int m_kernel_size[3], m_stride[3], m_padding[3], m_dialation[3], m_padding_type, m_num_filters;
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
				convTranspose(int num_filters, int* kernel_size, int* stride, int* padding, int* dialation,
					int padding_type, bool use_bias);
				bool operator()(tensor* x, tensor* y) override;

				void backward() override
				{
				};
				void backward(tensor* d_output, tensor* d_input);
				void update_weight() override;

			private:
				int m_kernel_size[3], m_stride[3], m_padding[3], m_dialation[3], m_padding_type, m_num_filters;
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

			class RNN : public Module
			{
			public:
				RNN(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
				bool operator()(tensor* x, tensor* y) override;
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
				tensor* m_weight;
				tensor* m_bias;
				tensor* m_output;
				tensor* d_weight;
				tensor* d_bias;
			};

			class GRU : public Module
			{
			public:
				GRU(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
				bool operator()(tensor* x, tensor* y) override;
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
		}
	}
}

#endif //!NN
