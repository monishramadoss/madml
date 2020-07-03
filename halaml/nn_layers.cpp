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

			RNN::RNN(int vocab_size, int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias, std::string nonlinearity) :
				m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), USE_BIAS(bias)
			{
			}

			std::tuple<tensor*, tensor*> RNN::forward(tensor* x, tensor* h0) {
				auto* y = new tensor(0.0, std::vector<int>{x->getShape()[0], m_directions* m_hidden_size});
				auto* hn = new tensor(0.0, std::vector<int>{x->getShape()[0], m_num_layers* m_directions, m_hidden_size});

				return std::make_tuple(y, hn);
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
			RNNCell::RNNCell(int vocab_size, int hidden_size, int out_size, int num_layers) : m_param({ vocab_size, hidden_size, out_size, 0, 0 })
			{
				if (out_size == 0)
					m_param.out_size = vocab_size;

				initVulkanThing(9);
				m_type = "RNNCell";
			}

			void RNNCell::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.hidden_size, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > maxComputeWorkGroupCount)
					m_group_x = maxComputeWorkGroupCount;
				m_group_y = static_cast<int>(alignSize(m_param.out_size, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
				if (m_group_y > maxComputeWorkGroupCount)
					m_group_y = maxComputeWorkGroupCount;
				m_group_z = 1;
			}

			tensor* RNNCell::forward(tensor* x, tensor* h, tensor* y)
			{
				const auto input_shape = x->getShape();//seq_len, input_size
				const auto hidden_shape = h->getShape(); //num_layers * num_directions, hidden_size

				auto* U = new tensor(1.0, std::vector<int>{m_param.hidden_size, m_param.vocab_size});
				auto* W = new tensor(1.0, std::vector<int>{m_param.hidden_size, m_param.hidden_size});
				auto* V = new tensor(1.0, std::vector<int>{m_param.out_size, m_param.hidden_size});

				auto* b1 = new tensor(0.0, std::vector<int>{m_param.hidden_size});
				auto* b2 = new tensor(0.0, std::vector<int>{m_param.out_size});

				auto* y = new tensor(0.0, std::vector<int>{input_shape[0], m_param.out_size});
				auto* hn = new tensor(0.0, hidden_shape);

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
					createPipeline(sizeof(RNN_cell_param))
				}
				bindTensor(m_device, U, 0, m_descriptor_set);
				bindTensor(m_device, V, 1, m_descriptor_set);
				bindTensor(m_device, W, 2, m_descriptor_set);
				bindTensor(m_device, x, 3, m_descriptor_set);
				bindTensor(m_device, h, 4, m_descriptor_set);
				bindTensor(m_device, b1, 5, m_descriptor_set);
				bindTensor(m_device, b2, 6, m_descriptor_set);
				bindTensor(m_device, b1, 7, m_descriptor_set);
				bindTensor(m_device, b2, 8, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(RNN_cell_param));
				layers.push_back(this);
				return y;
			}
		}
	}
}
