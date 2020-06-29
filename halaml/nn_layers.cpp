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
				auto* ir_vol2col = m_kernel->forward(x);
				layers.push_back(m_kernel);
				m_input.insert(m_input.end(), m_kernel->m_input.begin(), m_kernel->m_input.end());
				m_output.insert(m_output.end(), m_kernel->m_output.begin(), m_kernel->m_output.end());

				auto* w = new tensor(1.0, std::vector<int> {ir_vol2col->getShape()[1], m_num_filters});
				auto* y = m_mm->forward(ir_vol2col, w);
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

				auto out = m_kernel->output_shape();
				y->reshape(std::vector<int>{m_num_filters, out[0], out[1], out[2]});

				return y;
			}

			convTranspose::convTranspose(int num_filters, dhw kernel_size, dhw stride, dhw padding, dhw dilation, int padding_type,
				bool use_bias) : m_num_filters(num_filters), m_kernel_size(kernel_size), m_stride(stride), m_padding(padding), m_dilation(dilation), USE_BIAS(use_bias)
			{
				m_kernel = nullptr;
				m_layer = new dense(num_filters, use_bias);
			}

			tensor* convTranspose::forward(tensor* x)
			{
				auto input_shape = x->getShape();
				m_kernel = new col2vol(input_shape[0], m_kernel_size, m_padding, m_stride, m_dilation);
				auto* ir_col2vol = m_kernel->forward(x);
				add_module(m_kernel);
				layers.push_back(m_kernel);
				m_input.insert(m_input.end(), m_kernel->m_input.begin(), m_kernel->m_input.end());
				m_output.insert(m_output.end(), m_kernel->m_output.begin(), m_kernel->m_output.end());
				auto out = m_kernel->output_shape();

				auto* y = m_layer->forward(ir_col2vol);
				add_module(m_layer);
				m_input.insert(m_input.end(), m_layer->m_input.begin(), m_layer->m_input.end());
				m_output.insert(m_output.end(), m_layer->m_output.begin(), m_layer->m_output.end());
				y->reshape(std::vector<int>{m_num_filters, out[0], out[1], out[2]});

				return y;
			}
		}
	}
}
/*
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
 */
