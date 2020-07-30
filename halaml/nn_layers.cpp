#include "common.h"
#include "utils.h"
#include "nn_layers.h"
#include <numeric>
#define MAX_COMPUTE_WORK_GROUP_COUNT 1024
#define LOCAL_SZ_X 32
#define LOCAL_SZ_Y 32

namespace kernel
{
	namespace layers
	{
		namespace nn
		{
			dense::dense(int size, bool use_bias) : m_size(size), USE_BIAS(use_bias)
			{
				m_type = "dense";
				update_id();
				add_module(this);
				requires_sub_graph = true;
			}

			std::shared_ptr<tensor>dense::forward(std::shared_ptr<tensor>x)
			{
				set_sub_graph();
				inputs.push_back(x->getId());
				auto input_shape = x->getShape();
				auto mm = new matmul();
				sub_graph.push_back(mm);
				std::shared_ptr<tensor> w = std::make_shared<tensor>(tensor(1.0, std::vector<int>{input_shape[1], m_size}));
				weights.push_back(w->getId());
				std::shared_ptr<tensor> y = mm->forward(x, w);

				if (USE_BIAS)
				{
					std::shared_ptr<tensor> b = std::make_shared<tensor>(tensor(1.0, y->getShape()));
					auto bias = new math::add();
					sub_graph.push_back(bias);
					biases.push_back(b->getId());
					y = bias->forward(y, b);
				}

				outputs.push_back(y->getId());
				unset_sub_graph();
				parents.push_back(get_input_id(x->getId()));
				return y;
			}

			void dense::fwd_callback()
			{
				for (auto g : sub_graph)
				{
					g->fwd_callback();
				}
			}

			void dense::backward()
			{
				// MxK KxN = MxN
				// dw =  dy * x.T
				// db = mean(dy)
				// dx = W.T * dy
			}

			conv::conv(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
				bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size), m_stride(stride),
				m_padding(padding), m_dilation(dilation), USE_BIAS(use_bias)
			{
				m_type = "conv";
				update_id();
				add_module(this);
				requires_sub_graph = true;
			}

			std::shared_ptr<tensor>conv::forward(std::shared_ptr<tensor>x)
			{
				set_sub_graph();
				inputs.push_back(x->getId());
				auto input_shape = x->getShape();
				auto kernel = new vol2col(input_shape[0], m_kernel_size, m_padding, m_stride, m_dilation);
				sub_graph.push_back(kernel);
				auto mm = new matmul();
				sub_graph.push_back(mm);
				auto w = std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_num_filters, input_shape[0] * m_kernel_size.d* m_kernel_size.h* m_kernel_size.w}));
				weights.push_back(w->getId());
				auto ir_vol2col = kernel->forward(x); //27 9
				temporaries.push_back(ir_vol2col->getId());
				auto y = mm->forward(w, ir_vol2col);

				if (USE_BIAS)
				{
					std::shared_ptr<tensor> b = std::make_shared<tensor>(tensor(1.0, y->getShape()));
					auto bias = new math::add();
					sub_graph.push_back(bias);
					biases.push_back(b->getId());
					y = bias->forward(y, b);
				}

				outputs.push_back(y->getId());

				auto out = kernel->output_shape();
				y->reshape(std::vector<int>{m_num_filters, out[0], out[1], out[2]}); //8,9
				parents.push_back(get_input_id(x->getId()));
				unset_sub_graph();
				return y;
			}

			void conv::fwd_callback()
			{
				for (auto g : sub_graph)
				{
					g->fwd_callback();
				}
			}

			void conv::backward()
			{
				// MxK KxN = MxN
				// dw =  dy * x.T
				// db = mean(dy)
				// dx = W.T * dy
			}

			convTranspose::convTranspose(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation,
				int padding_type,
				bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size),
				m_stride(stride), m_padding(padding), m_dilation(dilation),
				USE_BIAS(use_bias)
			{
				m_type = "convT";
				update_id();
				add_module(this);
				requires_sub_graph = true;
			}

			std::shared_ptr<tensor>convTranspose::forward(std::shared_ptr<tensor>x)
			{
				set_sub_graph();
				inputs.push_back(x->getId());
				auto input_shape = x->getShape();
				auto kernel = new col2vol(input_shape[0], m_kernel_size, m_padding, m_stride, m_dilation);
				sub_graph.push_back(kernel);
				auto mm = new matmul();
				sub_graph.push_back(mm);
				auto w = std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_num_filters, input_shape[0] * m_kernel_size.d* m_kernel_size.h* m_kernel_size.w}));
				weights.push_back(w->getId());
				auto ir_col2vol = kernel->forward(x);
				temporaries.push_back(ir_col2vol->getId());
				auto y = mm->forward(w, ir_col2vol);

				if (USE_BIAS)
				{
					std::shared_ptr<tensor> b = std::make_shared<tensor>(tensor(1.0, y->getShape()));
					auto bias = new math::add();
					sub_graph.push_back(bias);
					biases.push_back(b->getId());
					y = bias->forward(y, b);
				}

				outputs.push_back(y->getId());

				auto out = kernel->output_shape();
				y->reshape(std::vector<int>{m_num_filters, out[0], out[1], out[2]}); //8,9
				unset_sub_graph();
				parents.push_back(get_input_id(x->getId()));
				return y;
			}

			void convTranspose::fwd_callback()
			{
				for (auto g : sub_graph)
				{
					g->fwd_callback();
				}
			}

			void convTranspose::backward()
			{
				// MxK KxN = MxN
				// dw =  dy * x.T
				// db = mean(dy)
				// dx = W.T * dy
			}

			RNN::RNN(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size,
				float dropout, bool bias, std::string nonlinearity) :
				m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_directions(1),
				m_output_size(output_size), m_seq_length(seq_length), USE_BIAS(bias)
			{
				m_type = "RNN";
				update_id();
				add_module(this);
				requires_sub_graph = true;
				if (bidirectional)
					m_directions = 2;
				if (output_size == 0)
					m_output_size = vocab_size;

				for (int dir = 0; dir < m_directions; ++dir)
				{
					for (int l = 0; l < m_num_layers; ++l)
					{
						const int input = l == 0 ? m_vocab_size : m_hidden_size;
						const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;

						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, input})));
						weights.push_back(weights_biases.back()->getId());
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size})));
						weights.push_back(weights_biases.back()->getId());
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output, m_hidden_size})));
						weights.push_back(weights_biases.back()->getId());
						if (USE_BIAS)
						{
							weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size})));
							biases.push_back(weights_biases.back()->getId());
							weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output})));
							biases.push_back(weights_biases.back()->getId());
						}
						else
						{
							weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_hidden_size})));
							biases.push_back(weights_biases.back()->getId());
							weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{output})));
							biases.push_back(weights_biases.back()->getId());
						}
						set_sub_graph();
						for (int i = 0; i < seq_length; ++i)
							cells.push_back(new rnn::RNNCell(input, m_hidden_size, output));
						sub_graph.insert(sub_graph.end(), cells.begin(), cells.end());
						unset_sub_graph();
					}
				}
			}

			std::tuple<std::shared_ptr<tensor>, std::shared_ptr<tensor>> RNN::forward(std::shared_ptr<tensor>x)
			{
				auto h = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
				return forward(x, h);
			}

			std::tuple<std::shared_ptr<tensor>, std::shared_ptr<tensor>> RNN::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>h)
			{
				inputs.push_back(x->getId());
				weights.push_back(h->getId());
				const auto input_shape = x->getShape();
				if (x->getShape().size() == 2)
					x->reshape(std::vector<int>{input_shape[0], 1, m_vocab_size});
				cache.push_back(x);
				cache.push_back(h);

				for (int l = 0; l < m_num_layers; ++l)
				{
					const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;
					cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, output})));
					cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
				}

				for (int dir = 0; dir < m_directions; ++dir)
				{
					for (int l = 0; l < m_num_layers; ++l)
					{
						for (int i = 0; i < input_shape[0]; ++i)
						{
							const uint64_t weight_bias_idx = static_cast<uint64_t>(m_num_layers) * dir * 5 + static_cast<uint64_t>(l) * 5;
							const uint64_t cache_idx = static_cast<uint64_t>(l) * 2;
							const uint64_t direction = dir == 1 ? input_shape[0] - i - 1 : i;

							uint64_t input_offset = cache[cache_idx]->getShape()[2] * direction;
							uint64_t weight_offset = direction * m_hidden_size;
							uint64_t output_offset = direction * cache[2 + cache_idx]->getShape()[2];

							if (dir == 1)
							{
								if (cache_idx != 0)
									input_offset += static_cast<uint64_t>(m_seq_length) * cache[cache_idx]->getShape()[2];
								weight_offset += static_cast<uint64_t>(m_hidden_size) * m_seq_length;
								output_offset += static_cast<uint64_t>(m_seq_length) * cache[2 + cache_idx]->getShape()[2];
							}
							const uint64_t cell_idx = static_cast<uint64_t>(m_num_layers) * dir * m_seq_length + static_cast<uint64_t>(m_seq_length) * l + i;
							cells[cell_idx]->forward(
								cache[0 + cache_idx],
								cache[1 + cache_idx],
								cache[2 + cache_idx],
								cache[3 + cache_idx],
								weights_biases[0 + weight_bias_idx],
								weights_biases[1 + weight_bias_idx],
								weights_biases[2 + weight_bias_idx],
								weights_biases[3 + weight_bias_idx],
								weights_biases[4 + weight_bias_idx],
								static_cast<int>(input_offset), static_cast<int>(weight_offset), static_cast<int>(output_offset)
							);
						}
					}
				}

				outputs.push_back(cache[cache.size() - 2]->getId());
				outputs.push_back(cache.back()->getId());
				parents.push_back(get_input_id(x->getId()));
				return std::make_tuple(cache[cache.size() - 2], cache[cache.size() - 1]);
			}

			void RNN::fwd_callback()
			{
				for (auto g : cells)
				{
					g->fwd_callback();
				}
			}

			void RNN::backward()
			{
				// MxK KxN = MxN
				// dw =  dy * x.T
				// db = mean(dy)
				// dx = W.T * dy
			}

			LSTM::LSTM(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size,
				float dropout, bool bias, std::string nonlinearity) :
				m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_directions(1),
				m_output_size(output_size), m_seq_length(seq_length), USE_BIAS(bias), nonlinearity_(std::move(nonlinearity))
			{
				m_type = "LSTM";
				update_id();
				add_module(this);
				requires_sub_graph = true;
				if (bidirectional)
					m_directions = 2;
				if (output_size == 0)
					m_output_size = vocab_size;

				for (int dir = 0; dir < m_directions; ++dir)
				{
					for (int l = 0; l < m_num_layers; ++l)
					{
						const int input = l == 0 ? m_vocab_size : m_hidden_size;
						const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;

						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, input, 4})));
						weights.push_back(weights_biases.back()->getId());
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size, 4})));
						weights.push_back(weights_biases.back()->getId());
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output, m_hidden_size, 4})));
						weights.push_back(weights_biases.back()->getId());

						if (USE_BIAS)
						{
							weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, 4})));
							biases.push_back(weights_biases.back()->getId());
							weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output})));
							biases.push_back(weights_biases.back()->getId());
						}
						else
						{
							weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_hidden_size, 4})));
							biases.push_back(weights_biases.back()->getId());
							weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{output})));
							biases.push_back(weights_biases.back()->getId());
						}
						set_sub_graph();
						for (int i = 0; i < seq_length; ++i)
							cells.push_back(new rnn::LSTMCell(input, m_hidden_size, output));
						sub_graph.insert(sub_graph.end(), cells.begin(), cells.end());
						unset_sub_graph();
					}
				}
			}

			std::tuple<std::shared_ptr<tensor>, std::shared_ptr<tensor>, std::shared_ptr<tensor>> LSTM::forward(std::shared_ptr<tensor>x)
			{
				auto h = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
				auto c = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
				return forward(x, h, c);
			}

			std::tuple<std::shared_ptr<tensor>, std::shared_ptr<tensor>, std::shared_ptr<tensor>> LSTM::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>h, std::shared_ptr<tensor>c)
			{
				inputs.push_back(x->getId());
				weights.push_back(h->getId());
				weights.push_back(c->getId());
				const auto input_shape = x->getShape();
				if (x->getShape().size() == 2)
					x->reshape(std::vector<int>{input_shape[0], 1, m_vocab_size});
				cache.push_back(x);
				cache.push_back(h);
				cache.push_back(c);

				for (int l = 0; l < m_num_layers; ++l)
				{
					const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;
					cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, output})));
					cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
					cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
				}

				for (int dir = 0; dir < m_directions; ++dir)
				{
					for (int l = 0; l < m_num_layers; ++l)
					{
						for (int i = 0; i < input_shape[0]; ++i)
						{
							const uint64_t weight_bias_idx = static_cast<uint64_t>(m_num_layers) * dir * 5 + static_cast<uint64_t>(l) * 5;
							const uint64_t cache_idx = static_cast<uint64_t>(l) * 2;
							const uint64_t direction = dir == 1 ? input_shape[0] - i - 1 : i;

							uint64_t input_offset = direction * cache[cache_idx]->getShape()[2];
							uint64_t weight_offset = direction * m_hidden_size;
							uint64_t output_offset = direction * cache[3 + cache_idx]->getShape()[2];

							if (dir == 1)
							{
								if (cache_idx != 0)
									input_offset += static_cast<uint64_t>(m_seq_length) * cache[cache_idx]->getShape()[2];
								weight_offset += static_cast<uint64_t>(m_hidden_size) * m_seq_length;
								output_offset += static_cast<uint64_t>(m_seq_length) * cache[2 + cache_idx]->getShape()[2];
							}
							const uint64_t cell_idx = static_cast<uint64_t>(m_num_layers) * dir * m_seq_length + static_cast<uint64_t>(m_seq_length) * l + i;
							cells[cell_idx]->forward(
								cache[0 + cache_idx],
								cache[1 + cache_idx],
								cache[2 + cache_idx],
								cache[3 + cache_idx],
								cache[4 + cache_idx],
								cache[5 + cache_idx],
								weights_biases[0 + weight_bias_idx],
								weights_biases[1 + weight_bias_idx],
								weights_biases[2 + weight_bias_idx],
								weights_biases[3 + weight_bias_idx],
								weights_biases[4 + weight_bias_idx],
								static_cast<int>(input_offset), static_cast<int>(weight_offset), static_cast<int>(output_offset)
							);
						}
					}
				}

				inputs.push_back(x->getId());
				outputs.push_back(cache[cache.size() - 3]->getId());
				outputs.push_back(cache[cache.size() - 2]->getId());
				outputs.push_back(cache.back()->getId());
				parents.push_back(get_input_id(x->getId()));
				return std::make_tuple(cache[cache.size() - 3], cache[cache.size() - 2], cache[cache.size() - 1]);
			}

			void LSTM::fwd_callback()
			{
				for (auto g : cells)
				{
					g->fwd_callback();
				}
			}

			void LSTM::backward()
			{
				// MxK KxN = MxN
				// dw =  dy * x.T
				// db = mean(dy)
				// dx = W.T * dy
			}

			GRU::GRU(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size,
				float dropout, bool bias, std::string nonlinearity) :
				m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_directions(1),
				m_output_size(output_size), m_seq_length(seq_length), USE_BIAS(bias), nonlinearity_(std::move(nonlinearity))
			{
				m_type = "GRU";
				update_id();
				add_module(this);
				requires_sub_graph = true;
				if (bidirectional)
					m_directions = 2;
				if (output_size == 0)
					m_output_size = vocab_size;

				for (int dir = 0; dir < m_directions; ++dir)
				{
					for (int l = 0; l < m_num_layers; ++l)
					{
						const int input = l == 0 ? m_vocab_size : m_hidden_size;
						const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;

						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, input, 3})));
						weights.push_back(weights_biases.back()->getId());
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size, 3})));
						weights.push_back(weights_biases.back()->getId());
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output, m_hidden_size, 3})));
						weights.push_back(weights_biases.back()->getId());

						if (USE_BIAS)
						{
							weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, 3})));
							biases.push_back(weights_biases.back()->getId());
							weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output})));
							biases.push_back(weights_biases.back()->getId());
						}
						else
						{
							weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_hidden_size, 3})));
							biases.push_back(weights_biases.back()->getId());
							weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{output})));
							biases.push_back(weights_biases.back()->getId());
						}
						set_sub_graph();
						for (int i = 0; i < seq_length; ++i)
							cells.push_back(new rnn::GRUCell(input, m_hidden_size, output));
						sub_graph.insert(sub_graph.end(), cells.begin(), cells.end());
						unset_sub_graph();
					}
				}
			}

			std::tuple<std::shared_ptr<tensor>, std::shared_ptr<tensor>> GRU::forward(std::shared_ptr<tensor>x)
			{
				auto h = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
				return forward(x, h);
			}

			std::tuple<std::shared_ptr<tensor>, std::shared_ptr<tensor>> GRU::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>h)
			{
				inputs.push_back(x->getId());
				weights.push_back(h->getId());
				const auto input_shape = x->getShape();
				if (x->getShape().size() == 2)
					x->reshape(std::vector<int>{input_shape[0], 1, m_vocab_size});
				cache.push_back(x);
				cache.push_back(h);

				for (int l = 0; l < m_num_layers; ++l)
				{
					const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;
					cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, output})));
					cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
				}

				for (int dir = 0; dir < m_directions; ++dir)
				{
					for (int l = 0; l < m_num_layers; ++l)
					{
						for (int i = 0; i < input_shape[0]; ++i)
						{
							const uint64_t weight_bias_idx = static_cast<uint64_t>(m_num_layers) * dir * 5 + static_cast<uint64_t>(l) * 5;
							const uint64_t cache_idx = static_cast<uint64_t>(l) * 2;
							const uint64_t direction = dir == 1 ? input_shape[0] - i - 1 : i;

							uint64_t input_offset = direction * cache[cache_idx]->getShape()[2];
							uint64_t weight_offset = direction * m_hidden_size;
							uint64_t output_offset = direction * cache[3 + cache_idx]->getShape()[2];

							if (dir == 1)
							{
								if (cache_idx != 0)
									input_offset += static_cast<uint64_t>(m_seq_length) * cache[cache_idx]->getShape()[2];
								weight_offset += static_cast<uint64_t>(m_hidden_size) * m_seq_length;
								output_offset += static_cast<uint64_t>(m_seq_length) * cache[cache_idx + 2]->getShape()[2];
							}

							const uint64_t cell_idx = static_cast<uint64_t>(m_num_layers) * dir * m_seq_length + static_cast<uint64_t>(m_seq_length) * l + i;
							cells[cell_idx]->forward(
								cache[0 + cache_idx],
								cache[1 + cache_idx],
								cache[2 + cache_idx],
								cache[3 + cache_idx],
								weights_biases[0 + weight_bias_idx],
								weights_biases[1 + weight_bias_idx],
								weights_biases[2 + weight_bias_idx],
								weights_biases[3 + weight_bias_idx],
								weights_biases[4 + weight_bias_idx],
								static_cast<int>(input_offset), static_cast<int>(weight_offset), static_cast<int>(output_offset)
							);
						}
					}
				}

				outputs.push_back(cache[cache.size() - 2]->getId());
				outputs.push_back(cache.back()->getId());
				parents.push_back(get_input_id(x->getId()));
				return std::make_tuple(cache[cache.size() - 2], cache[cache.size() - 1]);
			}

			void GRU::fwd_callback()
			{
				for (auto g : cells)
				{
					g->fwd_callback();
				}
			}

			void GRU::backward()
			{
				// MxK KxN = MxN
				// dw =  dy * x.T
				// db = mean(dy)
				// dx = W.T * dy
			}
		}
	}
}