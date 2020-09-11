#include "common.h"
#include "utils.h"
#include "nn_layers.h"
#include <numeric>
#include <cmath>

namespace layers
{
	namespace nn
	{
		dense::dense(int size, bool use_bias) : m_size(size), USE_BIAS(use_bias)
		{
			m_type = "dense";
			update_id();
			add_module(this);

			set_sub_graph();
			mm = new matmul();
			if (USE_BIAS)
				bias = new math::add();
			unset_sub_graph();
		}

		std::shared_ptr<tensor>& dense::operator()(const std::shared_ptr<tensor>& _x)
		{
			this->x = _x;
			set_sub_graph();
			auto input_shape = x->getShape();
			if (!w)
				w = std::make_shared<tensor>(tensor(1.0, std::vector<int>{input_shape[input_shape.size() - 1], m_size}));
			y = mm->operator()(x, w);

			if (USE_BIAS)
			{
				if (!b)
					b = std::make_shared<tensor>(tensor(1.0, y->getShape()));
				y = bias->operator()(y, b);
			}

			unset_sub_graph();
			if (!m1)
				m1 = get_input_id(x->getId());
			return y;

			// MxK KxN = MxN
			// dw =  dy * x.T
			// db = mean(dy)
			// dx = W.T * dy
		}

		int dense::set_backward()
		{
			if (USE_BIAS)
			{
				bias->dy = dy;
				bias->is_bias = true;
				bias->set_backward();
				db = bias->dw;

				mm->dy = bias->dx;
				mm->set_backward();
				dx = mm->dx;
				dw = mm->dw;
			}
			else
			{
				mm->dy = dy;
				mm->set_backward();
				dx = mm->dx;
				dw = mm->dw;
			}

			return 1;
		}

		void dense::update_weight()
		{
		}

		conv::conv(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
			bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size), m_stride(stride),
			m_padding(padding), m_dilation(dilation), USE_BIAS(use_bias)
		{
			m_type = "conv";
			update_id();
			add_module(this);
			set_sub_graph();
			mm = new matmul();
			if (USE_BIAS)
				bias = new math::add();
			trans = new transpose(std::vector<int>{1, 0, 2, 3, 4});
			unset_sub_graph();
		}

		std::shared_ptr<tensor>& conv::operator()(const std::shared_ptr<tensor>& x_)
		{
			x = x_;
			set_sub_graph();
			auto input_shape = x->getShape();

			int channels = input_shape[1];
			int batch_size = input_shape[0];

			if (!kernel)
				kernel = new vol2col(channels, m_kernel_size, m_padding, m_stride, m_dilation);
			if (!w)
			{
				int c = static_cast<int>(channels * m_kernel_size.d * m_kernel_size.h * m_kernel_size.w);
				w = std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_num_filters, c}));
			}

			t1 = kernel->operator()(x); //27 9
			y = mm->operator()(w, t1);
			auto out = kernel->output_shape();

			if (USE_BIAS)
			{
				if (!b)
					b = std::make_shared<tensor>(tensor(1.0, y->getShape()));
				t2 = bias->operator()(y, b);
				t2->reshape(std::vector<int>{m_num_filters, batch_size, out[0], out[1], out[2]}); //8,9
				t4 = trans->operator()(t2);
			}
			else
			{
				y->reshape(std::vector<int>{m_num_filters, batch_size, out[0], out[1], out[2]}); //8,9
				t4 = trans->operator()(y);
			}

			if (!m1)
				m1 = get_input_id(x->getId());
			unset_sub_graph();
			return t4;
		}

		int conv::set_backward()
		{
			if (USE_BIAS)
			{
				bias->dy = dy;
				bias->is_bias = true;
				bias->set_backward();
				db = bias->dw;

				mm->dy = bias->dx;
				mm->set_backward();
				dx = mm->dx;
				dw = mm->dw;
			}
			else
			{
				mm->dy = dy;
				mm->set_backward();
				dx = mm->dx;
				dw = mm->dw;
			}

			return 1;
		}

		void conv::update_weight()
		{
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
			set_sub_graph();
			mm = new matmul();
			if (USE_BIAS)
				bias = new math::add();
			trans = new transpose(std::vector<int>{1, 0, 2, 3, 4});

			unset_sub_graph();
		}

		std::shared_ptr<tensor>& convTranspose::operator()(const std::shared_ptr<tensor>& x_)
		{
			x = x_;
			set_sub_graph();
			auto input_shape = x->getShape();

			int channels = input_shape[1];
			int batch_size = input_shape[0];
			if (!kernel)
			{
				//TODO dilation broken
				m_padding.d = (m_kernel_size.d - 1) * m_dilation.d - m_padding.d;
				m_padding.h = (m_kernel_size.h - 1) * m_dilation.h - m_padding.h;
				m_padding.w = (m_kernel_size.w - 1) * m_dilation.w - m_padding.w;
				m_stride.d = m_stride.d != 0 && m_stride.d > 1 ? 1 / m_stride.d : 1;
				m_stride.h = m_stride.h != 0 && m_stride.h > 1 ? 1 / m_stride.h : 1;
				m_stride.w = m_stride.w != 0 && m_stride.w > 1 ? 1 / m_stride.w : 1;
				kernel = new vol2col(channels, m_kernel_size, m_padding, m_stride, m_dilation);
			}
			if (!w)
			{
				int c = static_cast<int>(channels * m_kernel_size.d * m_kernel_size.h * m_kernel_size.w);
				w = std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_num_filters, c}));
			}

			t1 = kernel->operator()(x);
			y = mm->operator()(w, t1);
			auto out = kernel->output_shape();
			if (USE_BIAS)
			{
				if (!b)
					b = std::make_shared<tensor>(tensor(1.0, y->getShape()));
				t2 = bias->operator()(y, b);
				t2->reshape(std::vector<int>{m_num_filters, batch_size, out[0], out[1], out[2]});
				t4 = trans->operator()(t2);
			}
			else
			{
				y->reshape(std::vector<int>{m_num_filters, batch_size, out[0], out[1], out[2]});
				t4 = trans->operator()(y);
			}

			if (!m1)
				m1 = get_input_id(x->getId());
			unset_sub_graph();
			return t4;
		}

		int convTranspose::set_backward()
		{
			if (USE_BIAS)
			{
				bias->dy = dy;
				bias->is_bias = true;
				bias->set_backward();
				db = bias->dw;

				mm->dy = bias->dx;
				mm->set_backward();
				dx = mm->dx;
				dw = mm->dw;

				//col2im
			}
			else
			{
				mm->dy = dy;
				mm->set_backward();
				dx = mm->dx;
				dw = mm->dw;
			}

			return 1;
		}

		void convTranspose::update_weight()
		{
		}

		//TODO rnn needs dynamic graph

		RNN::RNN(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size,
			float dropout, bool bias, std::string nonlinearity) :
			m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_directions(1),
			m_output_size(output_size), m_seq_length(seq_length), USE_BIAS(bias)
		{
			m_type = "RNN";
			update_id();
			add_module(this);

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
					weights_biases.push_back(
						std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size})));
					weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output, m_hidden_size})));
					if (USE_BIAS)
					{
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size})));
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output})));
					}
					else
					{
						weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_hidden_size})));
						weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{output})));
					}
					set_sub_graph();
					cells.push_back(new rnn::RNNCell(input, m_hidden_size, output));
					unset_sub_graph();
				}
			}
		}

		std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> RNN::operator()(const std::shared_ptr<tensor>& x)
		{
			h = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
			return operator()(x, h);
		}

		std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> RNN::operator()(
			const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h)
		{
			this->x = x;
			this->h = h;
			const auto input_shape = x->getShape();
			if (x->getShape().size() == 2)
				x->reshape(std::vector<int>{input_shape[0], 1, m_vocab_size});
			cache.push_back(x);
			cache.push_back(h);

			for (int l = 0; l < m_num_layers; ++l)
			{
				const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;
				cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, output})));
				cache.push_back(
					std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
			}

			for (int dir = 0; dir < m_directions; ++dir)
			{
				for (int i = 0; i < input_shape[0]; ++i)
				{
					for (int l = 0; l < m_num_layers; ++l)
					{
						const uint64_t weight_bias_idx = static_cast<uint64_t>(m_num_layers) * dir * 5 + static_cast<
							uint64_t>(l) * 5;
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
						const uint64_t cell_idx = static_cast<uint64_t>(m_num_layers) * dir * m_seq_length + static_cast<
							uint64_t>(m_seq_length) * l + i;
						cells[cell_idx]->operator()(
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

			auto m = get_input_id(x->getId());
			return std::forward_as_tuple(cache[cache.size() - 2], cache[cache.size() - 1]);
		}

		int RNN::set_backward()
		{
			if (USE_BIAS)
			{
			}
			else
			{
			}

			return 1;
		}

		void RNN::update_weight()
		{
		}

		LSTM::LSTM(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size,
			float dropout, bool bias, std::string nonlinearity) :
			m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_directions(1),
			m_output_size(output_size), m_seq_length(seq_length), USE_BIAS(bias), nonlinearity_(std::move(nonlinearity))
		{
			m_type = "LSTM";
			update_id();
			add_module(this);

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
					weights_biases.push_back(
						std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size, 4})));
					weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output, m_hidden_size, 4})));

					if (USE_BIAS)
					{
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, 4})));
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output})));
					}
					else
					{
						weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_hidden_size, 4})));
						weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{output})));
					}
					set_sub_graph();
					cells.push_back(new rnn::LSTMCell(input, m_hidden_size, output));
					unset_sub_graph();
				}
			}
		}

		std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> LSTM::operator()(
			const std::shared_ptr<tensor>& x)
		{
			h = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
			c = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
			return operator()(x, h, c);
		}

		std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> LSTM::operator()(
			const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h, const std::shared_ptr<tensor>& c)
		{
			this->x = x;
			this->h = h;
			this->c = c;
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
				cache.push_back(
					std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
				cache.push_back(
					std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
			}

			for (int dir = 0; dir < m_directions; ++dir)
			{
				for (int i = 0; i < input_shape[0]; ++i)
				{
					for (int l = 0; l < m_num_layers; ++l)
					{
						const uint64_t weight_bias_idx = static_cast<uint64_t>(m_num_layers) * dir * 5 + static_cast<
							uint64_t>(l) * 5;
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
						const uint64_t cell_idx = static_cast<uint64_t>(m_num_layers) * dir * m_seq_length + static_cast<
							uint64_t>(m_seq_length) * l + i;
						cells[cell_idx]->operator()(
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

			auto m = get_input_id(x->getId());
			return std::forward_as_tuple(cache[cache.size() - 3], cache[cache.size() - 2], cache[cache.size() - 1]);
		}

		int LSTM::set_backward()
		{
			if (USE_BIAS)
			{
			}
			else
			{
			}

			return 1;
		}

		void LSTM::update_weight()
		{
		}

		GRU::GRU(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size,
			float dropout, bool bias, std::string nonlinearity) :
			m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_directions(1),
			m_output_size(output_size), m_seq_length(seq_length), USE_BIAS(bias), nonlinearity_(std::move(nonlinearity))
		{
			m_type = "GRU";
			update_id();
			add_module(this);

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
					weights_biases.push_back(
						std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size, 3})));
					weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output, m_hidden_size, 3})));

					if (USE_BIAS)
					{
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, 3})));
						weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output})));
					}
					else
					{
						weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_hidden_size, 3})));
						weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{output})));
					}
					set_sub_graph();
					cells.push_back(new rnn::GRUCell(input, m_hidden_size, output));
					unset_sub_graph();
				}
			}
		}

		std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> GRU::operator()(const std::shared_ptr<tensor>& x)
		{
			h = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
			return operator()(x, h);
		}

		std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> GRU::operator()(
			const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h)
		{
			this->x = x;
			this->h = h;
			const auto input_shape = x->getShape();
			if (x->getShape().size() == 2)
				x->reshape(std::vector<int>{input_shape[0], 1, m_vocab_size});
			cache.push_back(x);
			cache.push_back(h);

			for (int l = 0; l < m_num_layers; ++l)
			{
				const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;
				cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, output})));
				cache.push_back(
					std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
			}

			for (int dir = 0; dir < m_directions; ++dir)
			{
				for (int i = 0; i < input_shape[0]; ++i)
				{
					for (int l = 0; l < m_num_layers; ++l)
					{
						const uint64_t weight_bias_idx = static_cast<uint64_t>(m_num_layers) * dir * 5 + static_cast<
							uint64_t>(l) * 5;
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

						const uint64_t cell_idx = static_cast<uint64_t>(m_num_layers) * dir * m_seq_length + static_cast<
							uint64_t>(m_seq_length) * l + i;
						cells[cell_idx]->operator()(
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
			auto m = get_input_id(x->getId());
			return std::forward_as_tuple(cache[cache.size() - 2], cache[cache.size() - 1]);
		}

		int GRU::set_backward()
		{
			if (USE_BIAS)
			{
			}
			else
			{
			}

			return 1;
		}

		void GRU::update_weight()
		{
		}
	}
}