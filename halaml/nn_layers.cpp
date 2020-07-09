#include "common.h"
#include "utils.h"
#include "nn_layers.h"
#include <numeric>
#define maxComputeWorkGroupCount 1024
#define LOCAL_SZ_X 32
#define LOCAL_SZ_Y 32
namespace kernel
{
	namespace layers
	{
		namespace nn
		{
			dense::dense(int size, bool use_bias) : m_size(size), USE_BIAS(use_bias) {
			}

			tensor* dense::forward(tensor* x)
			{
				auto input_shape = x->getShape();
				auto* mm = new matmul();
				auto* w = new tensor(1.0, std::vector<int> {input_shape[1], m_size});
				auto* y = mm->forward(x, w);
				layers.push_back(mm);

				if (USE_BIAS)
				{
					auto* b = new tensor(1.0, y->getShape());
					auto* bias = new math::add();
					y = bias->forward(y, b);
					layers.push_back(bias);
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
			}

			tensor* conv::forward(tensor* x)
			{
				auto input_shape = x->getShape();
				auto* kernel = new vol2col(input_shape[0], m_kernel_size, m_padding, m_stride, m_dilation);
				auto* mm = new matmul();
				layers.push_back(kernel);
				layers.push_back(mm);

				auto* ir_vol2col = kernel->forward(x); //27 9
				auto* w = new tensor(1.0, std::vector<int> {m_num_filters, ir_vol2col->getShape()[0]});
				auto* y = mm->forward(w, ir_vol2col);

				auto out = kernel->output_shape();
				y->reshape(std::vector<int>{m_num_filters, out[0], out[1], out[2]}); //8,9

				set_io(kernel);
				set_io(mm);

				if (USE_BIAS)
				{
					auto* bias = new math::add();
					layers.push_back(bias);

					auto* b = new tensor(1.0, y->getShape());
					y = bias->forward(y, b);
					set_io(bias);
				}

				add_module(this);
				return y;
			}

			convTranspose::convTranspose(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
				bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size), m_stride(stride), m_padding(padding), m_dilation(dilation), USE_BIAS(use_bias)
			{
			}

			tensor* convTranspose::forward(tensor* x)
			{
				auto input_shape = x->getShape();
				auto* kernel = new col2vol(input_shape[0], m_kernel_size, m_padding, m_stride, m_dilation);
				auto* mm = new matmul();
				layers.push_back(kernel);
				layers.push_back(mm);

				auto* ir_col2vol = kernel->forward(x);
				auto* w = new tensor(1.0, std::vector<int> {m_num_filters, ir_col2vol->getShape()[0]});
				auto* y = mm->forward(w, ir_col2vol);

				auto out = kernel->output_shape();
				y->reshape(std::vector<int>{m_num_filters, out[0], out[1], out[2]}); //8,9

				set_io(kernel);
				set_io(mm);

				if (USE_BIAS)
				{
					auto* bias = new math::add();
					layers.push_back(bias);

					auto* b = new tensor(1.0, y->getShape());
					y = bias->forward(y, b);
					set_io(bias);
				}
				add_module(this);
				return y;
			}

			void rnn_constructor_helper(int seq, int input, int hidden, int output, std::vector<RNNCell*>& layers)
			{
				for (int i = 0; i < seq; ++i)
					layers.push_back(new RNNCell(input, hidden, output));
			}


			RNN::RNN(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size, float dropout, bool bias, std::string nonlinearity)	:
				m_vocab_size(vocab_size), m_hidden_size(hidden_size),  m_num_layers(num_layers), m_seq_length(seq_length), m_output_size(output_size), m_directions(1), USE_BIAS(bias)
			{
				if (bidirectional)
					m_directions = 2;
				if (output_size == 0)
					m_output_size = vocab_size;

				for (int dir = 0; dir < m_directions; ++dir) {
					for (int l = 0; l < m_num_layers; ++l) {
						int input = l == 0 ? m_vocab_size : m_hidden_size;
						int output = l == m_num_layers-1 ? m_output_size : m_hidden_size;

						weights_biases.push_back(new tensor(1.0, std::vector<int>{m_hidden_size, input}));
						weights_biases.push_back(new tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size}));
						weights_biases.push_back(new tensor(1.0, std::vector<int>{output, m_hidden_size}));
						if (USE_BIAS)
						{
							weights_biases.push_back(new tensor(1.0, std::vector<int>{m_hidden_size}));
							weights_biases.push_back(new tensor(1.0, std::vector<int>{output}));
						}
						else
						{
							weights_biases.push_back(new tensor(0.0, std::vector<int>{m_hidden_size}));
							weights_biases.push_back(new tensor(0.0, std::vector<int>{output}));
						}
						for (int i = 0; i < seq_length; ++i)
							cells.push_back(new RNNCell(input, m_hidden_size, output));
					}
				}
			
			}

			std::tuple<tensor*, tensor*> RNN::forward(tensor* x) {
				const auto input_shape = x->getShape();
				x->reshape(std::vector<int>{input_shape[0], 1, m_vocab_size});
				cache.push_back(x);
				cache.push_back(new tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
				for (int l = 0; l < m_num_layers; ++l) {
					int input = l == 0 ? m_vocab_size : m_hidden_size;
					int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;
					cache.push_back(new tensor(0.0, std::vector<int>{m_seq_length, m_directions, output}));
					cache.push_back(new tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
				}
			
				for (int dir = 0; dir < m_directions; ++dir) {
					for (int l = 0; l < m_num_layers; ++l) {
						for (int i = 0; i < input_shape[0]; ++i) {
							int weight_bias_idx = dir * m_num_layers * 5 + l * 5;
							int cache_idx = l * 2;
							int direction = dir == 1 ? input_shape[0] - i - 1 : i;								
							
							int input_offset = direction * cache[cache_idx]->getShape()[2];
							int weight_offset = direction * m_hidden_size;
							int output_offset = direction * cache[cache_idx + 2]->getShape()[2];
														
							if (dir == 1) {
								if (cache_idx != 0)
									input_offset += m_seq_length * cache[cache_idx]->getShape()[2];
								weight_offset += m_seq_length * m_hidden_size;
								output_offset += m_seq_length * cache[cache_idx + 2]->getShape()[2];
							}

							cells[dir * m_num_layers * m_seq_length + l * m_seq_length + i]->forward(
								cache[cache_idx + 0],
								cache[cache_idx + 1],
								cache[cache_idx + 2],
								cache[cache_idx + 3],
								weights_biases[weight_bias_idx + 0], 
								weights_biases[weight_bias_idx + 1], 
								weights_biases[weight_bias_idx + 2], 
								weights_biases[weight_bias_idx + 3],
								weights_biases[weight_bias_idx + 4],
								input_offset, weight_offset, output_offset
							);					
						}
					}
				}
			

				for (auto l : cells) {
					layers.push_back(l);
					set_io(l);
				}
				add_module(this);
				return std::make_tuple(cache[cache.size() - 2], cache[cache.size() - 1]);
			}


			LSTM::LSTM(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size, float dropout, bool bias, std::string nonlinearity) :
				m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_seq_length(seq_length), m_output_size(output_size), m_directions(1), USE_BIAS(bias)
			{
				if (bidirectional)
					m_directions = 2;
				if (output_size == 0)
					m_output_size = vocab_size;

				for (int dir = 0; dir < m_directions; ++dir) {
					for (int l = 0; l < m_num_layers; ++l) {
						int input = l == 0 ? m_vocab_size : m_hidden_size;
						int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;

						weights_biases.push_back(new tensor(1.0, std::vector<int>{m_hidden_size, input, 4}));
						weights_biases.push_back(new tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size, 4}));
						weights_biases.push_back(new tensor(1.0, std::vector<int>{output, m_hidden_size, 4}));
						if (USE_BIAS)
						{
							weights_biases.push_back(new tensor(1.0, std::vector<int>{m_hidden_size, 4}));
							weights_biases.push_back(new tensor(1.0, std::vector<int>{output}));
						}
						else
						{
							weights_biases.push_back(new tensor(0.0, std::vector<int>{m_hidden_size, 4}));
							weights_biases.push_back(new tensor(0.0, std::vector<int>{output}));
						}
						for (int i = 0; i < seq_length; ++i)
							cells.push_back(new LSTMCell(input, m_hidden_size, output));
					}
				}

			}

			std::tuple<tensor*, tensor*, tensor*> LSTM::forward(tensor* x) {
				const auto input_shape = x->getShape();
				x->reshape(std::vector<int>{input_shape[0], 1, m_vocab_size});
				cache.push_back(x);
				cache.push_back(new tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
				cache.push_back(new tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
				for (int l = 0; l < m_num_layers; ++l) {
					int input = l == 0 ? m_vocab_size : m_hidden_size;
					int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;
					cache.push_back(new tensor(0.0, std::vector<int>{m_seq_length, m_directions, output}));
					cache.push_back(new tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
					cache.push_back(new tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
				}

				for (int dir = 0; dir < m_directions; ++dir) {
					for (int l = 0; l < m_num_layers; ++l) {
						for (int i = 0; i < input_shape[0]; ++i) {
							int weight_bias_idx = dir * m_num_layers * 5 + l * 5;
							int cache_idx = l * 3;
							int direction = dir == 1 ? input_shape[0] - i - 1 : i;

							int input_offset = direction * cache[cache_idx]->getShape()[2];
							int weight_offset = direction * m_hidden_size;
							int output_offset = direction * cache[cache_idx + 3]->getShape()[2];

							if (dir == 1) {
								if (cache_idx != 0)
									input_offset += m_seq_length * cache[cache_idx]->getShape()[2];
								weight_offset += m_seq_length * m_hidden_size;
								output_offset += m_seq_length * cache[cache_idx + 3]->getShape()[2];
							}
							cells[dir * m_num_layers * m_seq_length + l * m_seq_length + i]->forward(
								cache[cache_idx + 0],
								cache[cache_idx + 1],
								cache[cache_idx + 2],
								cache[cache_idx + 3],
								cache[cache_idx + 4],
								cache[cache_idx + 5],
								weights_biases[weight_bias_idx + 0],
								weights_biases[weight_bias_idx + 1],
								weights_biases[weight_bias_idx + 2],
								weights_biases[weight_bias_idx + 3],
								weights_biases[weight_bias_idx + 4],
								input_offset, weight_offset, output_offset
							);
						}
					}
				}


				for (auto l : cells) {
					layers.push_back(l);
					set_io(l);
				}
				add_module(this);
				return std::make_tuple(cache[cache.size() - 3], cache[cache.size() - 2], cache[cache.size() - 1]);
			}
			GRU::GRU(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size, float dropout, bool bias, std::string nonlinearity) :
				m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_seq_length(seq_length), m_output_size(output_size), m_directions(1), USE_BIAS(bias)
			{
				if (bidirectional)
					m_directions = 2;
				if (output_size == 0)
					m_output_size = vocab_size;

				for (int dir = 0; dir < m_directions; ++dir) {
					for (int l = 0; l < m_num_layers; ++l) {
						int input = l == 0 ? m_vocab_size : m_hidden_size;
						int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;

						weights_biases.push_back(new tensor(1.0, std::vector<int>{m_hidden_size, input, 3}));
						weights_biases.push_back(new tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size, 3}));
						weights_biases.push_back(new tensor(1.0, std::vector<int>{output, m_hidden_size, 3}));
						if (USE_BIAS)
						{
							weights_biases.push_back(new tensor(1.0, std::vector<int>{m_hidden_size, 3}));
							weights_biases.push_back(new tensor(1.0, std::vector<int>{output}));
						}
						else
						{
							weights_biases.push_back(new tensor(0.0, std::vector<int>{m_hidden_size, 3}));
							weights_biases.push_back(new tensor(0.0, std::vector<int>{output}));
						}
						for (int i = 0; i < seq_length; ++i)
							cells.push_back(new GRUCell(input, m_hidden_size, output));
					}
				}

			}

			std::tuple<tensor*, tensor*> GRU::forward(tensor* x) {
				const auto input_shape = x->getShape();
				x->reshape(std::vector<int>{input_shape[0], 1, m_vocab_size});
				cache.push_back(x);
				cache.push_back(new tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
				for (int l = 0; l < m_num_layers; ++l) {
					int input = l == 0 ? m_vocab_size : m_hidden_size;
					int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;
					cache.push_back(new tensor(0.0, std::vector<int>{m_seq_length, m_directions, output}));
					cache.push_back(new tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
				}

				for (int dir = 0; dir < m_directions; ++dir) {
					for (int l = 0; l < m_num_layers; ++l) {
						for (int i = 0; i < input_shape[0]; ++i) {
						int weight_bias_idx = dir * m_num_layers * 5 + l * 5;
							int cache_idx = l * 2;
							int direction = dir == 1 ? input_shape[0] - i - 1 : i;								
							
							int input_offset = direction * cache[cache_idx]->getShape()[2];
							int weight_offset = direction * m_hidden_size;
							int output_offset = direction * cache[cache_idx + 2]->getShape()[2];
														
							if (dir == 1) {
								if (cache_idx != 0)
									input_offset += m_seq_length * cache[cache_idx]->getShape()[2];
								weight_offset += m_seq_length * m_hidden_size;
								output_offset += m_seq_length * cache[cache_idx + 2]->getShape()[2];
							}

							cells[dir * m_num_layers * m_seq_length + l * m_seq_length + i]->forward(
								cache[cache_idx + 0],
								cache[cache_idx + 1],
								cache[cache_idx + 2],
								cache[cache_idx + 3],
								weights_biases[weight_bias_idx + 0],
								weights_biases[weight_bias_idx + 1],
								weights_biases[weight_bias_idx + 2],
								weights_biases[weight_bias_idx + 3],
								weights_biases[weight_bias_idx + 4],
								input_offset, weight_offset, output_offset
							);
						}
					}
				}

				for (auto l : cells) {
					layers.push_back(l);
					set_io(l);
				}
				add_module(this);
				return std::make_tuple(cache[cache.size() - 2], cache[cache.size() - 1]);
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
			RNNCell::RNNCell(int vocab_size, int hidden_size, int output_size) : m_param({ vocab_size, hidden_size, output_size, 0, 0 })
			{
				if (output_size == 0)
					m_param.output_size = vocab_size;

				initVulkanThing(9);
				m_type = "RNNCell";
			}

			void RNNCell::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.hidden_size, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > maxComputeWorkGroupCount)
					m_group_x = maxComputeWorkGroupCount;
				m_group_y = static_cast<int>(alignSize(m_param.output_size, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
				if (m_group_y > maxComputeWorkGroupCount)
					m_group_y = maxComputeWorkGroupCount;
				m_group_z = 1;
			}

			void RNNCell::forward(tensor* x, tensor* h, tensor* y, tensor* hn, tensor* U, tensor* W, tensor* V, tensor* b1, tensor* b2, int input_offset, int weight_offset, int output_offset)
			{
				const auto input_shape = x->getShape();//seq_len, input_size
				const auto hidden_shape = h->getShape(); //num_layers * num_directions, hidden_size
				
				m_param.input_offset = input_offset;
				m_param.weight_offset = weight_offset;
				m_param.output_offset = output_offset;

				m_input.push_back(x->getId());
				m_input.push_back(h->getId());
				m_input.push_back(U->getId());
				m_input.push_back(W->getId());
				m_input.push_back(V->getId());
				m_input.push_back(b1->getId());
				m_input.push_back(b2->getId());
				m_output.push_back(y->getId());
				m_output.push_back(hn->getId());

				if (m_pipeline == nullptr)
				{
					computeGroupCount();
					createShaderModule(shaders::rnnCell_spv, sizeof(shaders::rnnCell_spv));
					createPipeline(sizeof(RNN_cell_param));
				}

				bindTensor(m_device, U, 0, m_descriptor_set);
				bindTensor(m_device, V, 1, m_descriptor_set);
				bindTensor(m_device, W, 2, m_descriptor_set);
				bindTensor(m_device, x, 3, m_descriptor_set);
				bindTensor(m_device, h, 4, m_descriptor_set);
				bindTensor(m_device, b1, 5, m_descriptor_set);
				bindTensor(m_device, b2, 6, m_descriptor_set);
				bindTensor(m_device, y, 7, m_descriptor_set);
				bindTensor(m_device, hn, 8, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(RNN_cell_param));
				layers.push_back(this);
			}

			LSTMCell::LSTMCell(int vocab_size, int hidden_size, int output_size) : m_param({ vocab_size, hidden_size, output_size, 0, 0 })
			{
				if (output_size == 0)
					m_param.output_size = vocab_size;

				initVulkanThing(11);
				m_type = "LSTMCell";
			}

			void LSTMCell::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.hidden_size, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > maxComputeWorkGroupCount)
					m_group_x = maxComputeWorkGroupCount;
				m_group_y = static_cast<int>(alignSize(m_param.output_size, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
				if (m_group_y > maxComputeWorkGroupCount)
					m_group_y = maxComputeWorkGroupCount;
				m_group_z = 1;
			}

			void LSTMCell::forward(tensor* x, tensor* h, tensor* c, tensor* y, tensor*  hn, tensor* cn, tensor* U, tensor* W, tensor* V, tensor* b1, tensor* b2, int input_offset, int weight_offset, int output_offset)
			{
				const auto input_shape = x->getShape();//seq_len, input_size
				const auto hidden_shape = h->getShape(); //num_layers * num_directions, hidden_size
				const auto cell_shape = c->getShape();

				m_param.input_offset = input_offset;
				m_param.weight_offset = weight_offset;
				m_param.output_offset = output_offset;

				m_input.push_back(x->getId());
				m_input.push_back(h->getId());
				m_input.push_back(c->getId());
				m_input.push_back(U->getId());
				m_input.push_back(W->getId());
				m_input.push_back(V->getId());
				m_input.push_back(b1->getId());
				m_input.push_back(b2->getId());
				m_output.push_back(y->getId());
				m_output.push_back(hn->getId());
				m_output.push_back(cn->getId());

				if (m_pipeline == nullptr)
				{
					computeGroupCount();
					createShaderModule(shaders::lstmCell_spv, sizeof(shaders::lstmCell_spv));
					createPipeline(sizeof(RNN_cell_param));
				}

				bindTensor(m_device, U, 0, m_descriptor_set);
				bindTensor(m_device, V, 1, m_descriptor_set);
				bindTensor(m_device, W, 2, m_descriptor_set);
				bindTensor(m_device, x, 3, m_descriptor_set);
				bindTensor(m_device, h, 4, m_descriptor_set);
				bindTensor(m_device, c, 5, m_descriptor_set);
				bindTensor(m_device, b1, 6, m_descriptor_set);
				bindTensor(m_device, b2, 7, m_descriptor_set);
				bindTensor(m_device, y, 8, m_descriptor_set);
				bindTensor(m_device, hn, 9, m_descriptor_set);
				bindTensor(m_device, cn, 10, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(RNN_cell_param));
				layers.push_back(this);
				
			}

			GRUCell::GRUCell(int vocab_size, int hidden_size, int output_size) : m_param({ vocab_size, hidden_size, output_size, 0, 0 })
			{
				if (output_size == 0)
					m_param.output_size = vocab_size;

				initVulkanThing(9);
				m_type = "GRUCell";
			}

			void GRUCell::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.hidden_size, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > maxComputeWorkGroupCount)
					m_group_x = maxComputeWorkGroupCount;
				m_group_y = static_cast<int>(alignSize(m_param.output_size, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
				if (m_group_y > maxComputeWorkGroupCount)
					m_group_y = maxComputeWorkGroupCount;
				m_group_z = 1;
			}

			void GRUCell::forward(tensor* x, tensor* h, tensor* y, tensor* hn, tensor* U, tensor* W, tensor* V, tensor* b1, tensor* b2, int input_offset, int weight_offset, int output_offset)
			{
				const auto input_shape = x->getShape();//seq_len, input_size
				const auto hidden_shape = h->getShape(); //num_layers * num_directions, hidden_size

				m_param.input_offset = input_offset;
				m_param.weight_offset = weight_offset;
				m_param.output_offset = output_offset;

				
				m_input.push_back(x->getId());
				m_input.push_back(h->getId());
				m_input.push_back(U->getId());
				m_input.push_back(W->getId());
				m_input.push_back(V->getId());
				m_input.push_back(b1->getId());
				m_input.push_back(b2->getId());
				m_output.push_back(y->getId());
				
				if (m_pipeline == nullptr)
				{
					computeGroupCount();
					createShaderModule(shaders::gruCell_spv, sizeof(shaders::gruCell_spv));
					createPipeline(sizeof(RNN_cell_param));
				}

				bindTensor(m_device, U, 0, m_descriptor_set);
				bindTensor(m_device, V, 1, m_descriptor_set);
				bindTensor(m_device, W, 2, m_descriptor_set);
				bindTensor(m_device, x, 3, m_descriptor_set);
				bindTensor(m_device, h, 4, m_descriptor_set);
				bindTensor(m_device, b1, 5, m_descriptor_set);
				bindTensor(m_device, b2, 6, m_descriptor_set);
				bindTensor(m_device, y, 7, m_descriptor_set);
				bindTensor(m_device, hn, 8, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(RNN_cell_param));
				layers.push_back(this);
			}
		}
	}
}
