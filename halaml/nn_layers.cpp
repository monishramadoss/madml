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
		struct gradientParam
		{
			int total;
			float lr;
		};

		std::vector<Module*>* gradient::get_module()
		{
			return &Module::module_list;
		}

		gradient::gradient(float lr) : m_lr(lr)
		{
			initVulkanThing(3);
			m_type = "gradient";
			m_total = 0;
		}

		void gradient::reshapeOutTensor(tensor* x, tensor* z)
		{
			Shape shape = x->getShape();
			*z = z->reshape(nullptr, shape);
		}

		bool gradient::forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs)
		{
			return forward(ins[0], ins[1], outs[0]);
		}

		bool gradient::forward(tensor* x, tensor* y, tensor* z)
		{
			if (m_pipeline == nullptr)
			{
				m_total = x->count();
				computeGroupCount();
				createShaderModule(shaders::gradient_spv, sizeof(shaders::gradient_spv));
				createPipeline(sizeof(gradientParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);
			bindTensor(m_device, z, 2, m_descriptor_set);

			gradientParam param = { m_total, m_lr };
			recordCommandBuffer(static_cast<void*>(&param), sizeof(gradientParam));
			return true;
		}

		bool gradient::computeGroupCount()
		{
			m_group_x = static_cast<int>(alignSize(m_total, LOCAL_SZ_X)) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}

namespace kernel
{
	namespace layers
	{
		namespace nn
		{
			std::vector<Module*>* dense::get_module()
			{
				return &Module::module_list;
			}

			dense::dense(int size, bool use_bias)
				: size(size), USE_BIAS(use_bias)
			{
				m_mm = new matmul();
				if (USE_BIAS)
					m_bias_op = new operators(0);
				else
					m_bias_op = nullptr;
				m_input = nullptr;
				m_weight = nullptr;
				m_bias = nullptr;
				m_output = nullptr;
				d_weight = nullptr;
				d_bias = nullptr;
			}

			bool dense::operator()(tensor* x, tensor* y)
			{
				m_input = x;
				std::vector<int> input_shape = x->getShape();
				std::vector<int> weight_shape = std::vector<int>{ input_shape[1], size };
				std::vector<int> bias_shape = std::vector<int>{ input_shape[0], size };
				std::vector<int> output_shape = std::vector<int>{ input_shape[0], size };

				m_weight = new tensor(1.0, weight_shape);
				m_output = new tensor(0.0, output_shape);
				*y = *m_output;
				if (USE_BIAS)
					m_bias = new tensor(1.0, bias_shape);

				m_mm->forward(x, m_weight, y);
				forward_layers.push_back(m_mm);

				if (USE_BIAS)
				{
					m_bias_op->forward(y, m_bias, y);
					forward_layers.push_back(m_bias_op);
				}

				add_layer(this);
				return true;
			}

			void dense::backward(tensor* d_output, tensor* d_input)
			{
				// _loss
				//
			}

			void dense::update_weight()
			{
				std::cout << "Backward Dense Layer" << std::endl;
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
			std::vector<Module*>* conv::get_module()
			{
				return &Module::module_list;
			}

			conv::conv(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type, bool use_bias)
				: m_kernel_size(kernel_size), m_stride(stride), m_padding(padding),
				m_dilation(dilation), m_padding_type(padding_type), m_num_filters(num_filters),
				USE_BIAS(use_bias)
			{
				m_mm = new matmul();
				if (USE_BIAS)
					m_bias_op = new operators(0);
				else
					m_bias_op = nullptr;
				m_input = nullptr;
				m_weight = nullptr;
				m_bias = nullptr;
				m_output = nullptr;
				m_input_t = nullptr;
				m_kernel = nullptr;
				d_weight = nullptr;
				d_bias = nullptr;
			}

			bool conv::operator()(tensor* x, tensor* y)
			{
				m_input = x;
				auto input_shape = x->getShape(); //cdhw
				const int depth_col = (input_shape[1] + 2 * m_padding.d - (m_dilation.d * (m_kernel_size.d - 1) + 1)) /
					m_stride.d + 1;
				const int height_col = (input_shape[2] + 2 * m_padding.h - (m_dilation.h * (m_kernel_size.h - 1) + 1)) /
					m_stride.h + 1;
				const int width_col = (input_shape[3] + 2 * m_padding.w - (m_dilation.w * (m_kernel_size.w - 1) + 1)) /
					m_stride.w + 1;

				const int n_output_plane = input_shape[0] * m_kernel_size.w * m_kernel_size.h * m_kernel_size.d;
				const int output_length = depth_col * height_col * width_col;

				m_kernel = new vol2col(input_shape[0], m_kernel_size, m_padding, m_stride, m_dilation);

				const auto input_t_shape = std::vector<int>{ n_output_plane, output_length };
				const auto weight_shape = std::vector<int>{ m_num_filters, n_output_plane }; // T
				const auto bias_shape = std::vector<int>{ m_num_filters, depth_col, height_col, width_col };
				const auto output_shape = std::vector<int>{ m_num_filters, output_length };

				m_input_t = new tensor(0.0, input_t_shape);
				m_weight = new tensor(1.0, weight_shape);
				m_output = new tensor(0.0, output_shape);
				*y = *m_output;
				if (USE_BIAS)
					m_bias = new tensor(1.0, bias_shape);

				m_kernel->forward(x, m_input_t);
				forward_layers.push_back(m_kernel);
				m_mm->forward(m_weight, m_input_t, m_output);
				forward_layers.push_back(m_mm);

				if (USE_BIAS)
				{
					m_bias_op->forward(y, m_bias, y);
					forward_layers.push_back(m_bias_op);
				}

				y->reshape(bias_shape);

				add_layer(this);
				return true;
			}

			void conv::backward(tensor* x, tensor* y)
			{
			}

			void conv::update_weight()
			{
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
			RNNCell::RNNCell(int vocab_size, int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias, std::string nonlinearity)
				: m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_dropout(dropout), m_bidirectional(bidirectional),
				USE_BIAS(bias)
			{
				tensor* m_input = nullptr;
				tensor* m_output = nullptr;
				tensor* m_input_hidden = nullptr;

				m_input_hidden_layer = new matmul();
				m_hidden_hidden_layer = new matmul();

				if (USE_BIAS) {
					m_input_bias_layer = new operators(0);
					m_hidden_bias_layer = new operators(0);
				}
				else
				{
					m_input_bias_layer = nullptr;
					m_hidden_bias_layer = nullptr;
				}

				m_input_output_add_layer = new operators(0);

				if (nonlinearity == "relu")
				{
					m_tanh = nullptr;
					m_relu = new activation::relu();
				}
				else
				{
					m_tanh = new activation::tanh();
					m_relu = nullptr;
				}
			}

			bool RNNCell::operator()(tensor* x, tensor* h, tensor* y)
			{
				m_input = x;
				auto input_shape = x->getShape(); //cdhw
				int num_directions = 1;
				if (m_bidirectional)
					num_directions = 2;

				m_input_hidden = new tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size});
				m_hidden_hidden = new tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size});
				m_WIH = new tensor(1.0, std::vector<int>{m_hidden_size, input_shape[0]});
				m_WHH = new tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size});
				m_output = new tensor(1.0, std::vector<int>{m_hidden_size});

				if (USE_BIAS)
				{
					m_bh = new tensor(1.0, std::vector<int>{m_hidden_size});
					m_bi = new tensor(1.0, std::vector<int>{m_hidden_size});
				}

				m_input_hidden_layer->forward(m_WIH, x, m_input_hidden);
				m_hidden_hidden_layer->forward(m_WHH, h, m_hidden_hidden);

				if (USE_BIAS)
				{
					m_input_bias_layer->forward(m_input_hidden, m_bi, m_input_hidden);
					m_hidden_bias_layer->forward(m_hidden_hidden, m_bh, m_hidden_hidden);
				}

				m_input_output_add_layer->forward(m_input_hidden, m_hidden_hidden, y);

				return true;
			}
		}
	}
}
