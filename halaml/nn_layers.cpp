#include "common.h"
#include "utils.h"
#include "nn_layers.h"
#include <numeric>
#define maxComputeWorkGroupCount 1024
#define LOCAL_SZ_X 1024

namespace kernel
{
	namespace layers
	{
		namespace nn
		{
			dense::dense(int size, bool use_bias) : m_size(size), USE_BIAS(use_bias) {
				m_mm = new matmul();
				if (USE_BIAS) {
					m_bias_op = new math::add(true, false);
				}
			}

			tensor* dense::forward(tensor* x)
			{
				auto input_shape = x->getShape();
				auto* w = new tensor(1.0, std::vector<int> {input_shape[1], m_size});
				auto* y = m_mm->forward(x, w);
				layers.push_back(m_mm);
				m_input.insert(m_input.end(), m_mm->m_input.begin(), m_mm->m_input.end());
				m_output.insert(m_output.end(), m_mm->m_output.begin(), m_mm->m_output.end());

				if (USE_BIAS)
				{
					auto* b = new tensor(1.0, y->getShape());
					y = m_bias_op->forward(y, b);
					layers.push_back(m_bias_op);
					m_input.insert(m_input.end(), m_bias_op->m_input.begin(), m_bias_op->m_input.end());
					m_output.insert(m_output.end(), m_bias_op->m_output.begin(), m_bias_op->m_output.end());
				}
				add_module(this);
				return y;
			}
		}
	}
}

namespace kernel
{
	namespace layers
	{
		namespace nn
		{
			conv::conv(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
				bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size), m_stride(stride), m_padding(padding), m_dilation(dilation), USE_BIAS(use_bias)
			{
				m_kernel = nullptr;
				m_mm = new matmul();
				if (USE_BIAS)
				{
					m_bias_op = new math::add(true, false);
				}
			}

			tensor* conv::forward(tensor* x)
			{
				auto input_shape = x->getShape();
				m_kernel = new vol2col(input_shape[0], m_kernel_size, m_padding, m_stride, m_dilation);
				auto* ir_vol2col = m_kernel->forward(x); //27 9
				layers.push_back(m_kernel);
				m_input.insert(m_input.end(), m_kernel->m_input.begin(), m_kernel->m_input.end());
				m_output.insert(m_output.end(), m_kernel->m_output.begin(), m_kernel->m_output.end());

				auto* w = new tensor(1.0, std::vector<int> {m_num_filters, ir_vol2col->getShape()[0]});
				auto* y = m_mm->forward(w, ir_vol2col);
				layers.push_back(m_mm);
				m_input.insert(m_input.end(), m_mm->m_input.begin(), m_mm->m_input.end());
				m_output.insert(m_output.end(), m_mm->m_output.begin(), m_mm->m_output.end());
				auto out = m_kernel->output_shape();
				y->reshape(std::vector<int>{m_num_filters, out[0], out[1], out[2]}); //8,9

				if (USE_BIAS)
				{
					auto* b = new tensor(1.0, y->getShape());
					y = m_bias_op->forward(y, b);
					layers.push_back(m_bias_op);
					m_input.insert(m_input.end(), m_bias_op->m_input.begin(), m_bias_op->m_input.end());
					m_output.insert(m_output.end(), m_bias_op->m_output.begin(), m_bias_op->m_output.end());
				}
				add_module(this);
				return y;
			}

			convTranspose::convTranspose(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
				bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size), m_stride(stride), m_padding(padding), m_dilation(dilation), USE_BIAS(use_bias)
			{
				m_kernel = nullptr;
				m_mm = new matmul();
				if (USE_BIAS)
				{
					m_bias_op = new math::add(true, false);
				}
			}

			tensor* convTranspose::forward(tensor* x)
			{
				auto input_shape = x->getShape();
				m_kernel = new col2vol(input_shape[0], m_kernel_size, m_padding, m_stride, m_dilation);
				auto* ir_col2vol = m_kernel->forward(x);
				add_module(m_kernel);
				layers.push_back(m_kernel);
				
				auto* w = new tensor(1.0, std::vector<int> {m_num_filters, ir_col2vol->getShape()[0]});
				auto* y = m_mm->forward(w, ir_col2vol);
				layers.push_back(m_mm);
				auto out = m_kernel->output_shape();
				y->reshape(std::vector<int>{m_num_filters, out[0], out[1], out[2]}); //8,9

				if (USE_BIAS)
				{
					auto* b = new tensor(1.0, y->getShape());
					y = m_bias_op->forward(y, b);
					layers.push_back(m_bias_op);
					m_input.insert(m_input.end(), m_bias_op->m_input.begin(), m_bias_op->m_input.end());
					m_output.insert(m_output.end(), m_bias_op->m_output.begin(), m_bias_op->m_output.end());
				}
				add_module(this);
				return y;
			}

			RNN::RNN(int vocab_size, int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias, std::string nonlinearity) :
				m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), USE_BIAS(bias)
				/// requires batch first
			{
				ih_mm = new matmul();
				hh_mm = new matmul();
				oh_mm = new matmul();
				ht_add = new math::add(false, false);
				if(USE_BIAS)
					bias = new math::add(true, false);
				m_directions = 1;
				if (bidirectional)
					m_directions = 2;
				tanh = new math::tanh();
				softmax = new math::exp();
			}

			std::tuple<tensor*, tensor*> RNN::forward(tensor* x, tensor* h0) { 
				auto input_shape = x->getShape();//seq_len, input_size
				auto hidden_shape = h0->getShape(); //num_layers * num_directions, hidden_size
				
				auto* U = new tensor(1.0, std::vector<int>{m_hidden_size, m_vocab_size});
				auto* V = new tensor(1.0, std::vector<int>{m_vocab_size, m_hidden_size});
				auto* W = new tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size});
				
				auto* t1 = ih_mm->forward(U, x);
				auto* t2 = hh_mm->forward(W, h0);
				auto* t3 = ht_add->forward(t1, t2);				

				if (USE_BIAS) {
					auto b = new tensor(1.0, t3->getShape());
					t3 = bias->forward(t3, b);
				}
				
				auto* hn = tanh->forward(t3);
				
				auto* y = oh_mm->forward(V, hn);
				y = softmax->forward(y);
				//auto* y = new tensor(0.0, std::vector<int>{x->getShape()[0], m_directions * m_hidden_size});
				//auto* hn = new tensor(0.0, std::vector<int>{x->getShape()[0], m_num_layers * m_directions, m_hidden_size});

				return std::make_tuple(y, hn);
			}
			
			RNNCell::RNNCell() {
				
			}


		}
	}
}
