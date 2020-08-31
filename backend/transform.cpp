#include "common.h"
#include "utils.h"
#include "transform.h"

constexpr int local_sz_x_conv = 16;
constexpr int local_sz_y_conv = 64;

namespace layers
{
	vol2col::vol2col(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation) : Base_Layer<vol2col_param>(2)

	{
		m_type = "vol2col";
		m_param = {
			0, 1, channels, kernel.h, kernel.w, kernel.d, pad.h, pad.w, pad.d, stride.h, stride.w, stride.d, dilation.h,
			dilation.w, dilation.d, 0, 0, 0, 0, 0, 0
		};
	}

	void vol2col::computeGroupCount()
	{
		size_t tmp = m_param.channels;
		tmp *= static_cast<size_t>(m_param.kernel_h);
		tmp *= static_cast<size_t>(m_param.kernel_w);
		tmp *= static_cast<size_t>(m_param.kernel_d);

		m_group_x = static_cast<int>(alignSize(tmp, local_sz_x_conv)) / local_sz_x_conv;
		if (m_group_x > max_compute_work_group_count)
			m_group_x = max_compute_work_group_count;
		m_group_y = static_cast<int>(alignSize(m_param.batchsize, local_sz_y_conv)) / local_sz_y_conv;
		if (m_group_y > max_compute_work_group_count)
			m_group_y = max_compute_work_group_count;
		m_group_z = 1;
	}

	std::shared_ptr<tensor>& vol2col::operator()(const std::shared_ptr<tensor>& x_)
	{
		if (m_pipeline == nullptr)
		{
			const float depth = static_cast<float>(x_->getShape()[x_->getShape().size() - 3]);
			const float height = static_cast<float>(x_->getShape()[x_->getShape().size() - 2]);
			const float width = static_cast<float>(x_->getShape()[x_->getShape().size() - 1]);
			m_param.batchsize = x_->getShape()[0];
			m_param.channels = x_->getShape()[1];
			m_param.depth_vol = depth;
			m_param.height_vol = height;
			m_param.width_vol = width;
			m_param.depth_col = (depth + 2 * m_param.pad_d - (m_param.dilation_d * (m_param.kernel_d - 1) + 1)) / m_param.
				stride_d + 1;
			m_param.height_col = (height + 2 * m_param.pad_h - (m_param.dilation_h * (m_param.kernel_h - 1) + 1)) / m_param.
				stride_h + 1;
			m_param.width_col = (width + 2 * m_param.pad_w - (m_param.dilation_w * (m_param.kernel_w - 1) + 1)) / m_param.
				stride_w + 1;
		}
		const int n_out_plane = static_cast<int>(m_param.channels * m_param.kernel_d * m_param.kernel_h * m_param.kernel_w);
		const int output_length = static_cast<int>(m_param.batchsize * m_param.depth_col * m_param.height_col * m_param.width_col);
		layer_construct_forward(kernel::shaders::vol2col_spv, sizeof(kernel::shaders::vol2col_spv), x_, Format::kFormatFp32, std::vector<int>{n_out_plane, output_length});
		return y;
	}

	std::vector<int> vol2col::output_shape() const
	{
		int d = static_cast<int>(m_param.depth_col);
		int h = static_cast<int>(m_param.height_col);
		int w = static_cast<int>(m_param.width_col);
		return std::vector<int>{d, h, w};
	}

	col2vol::col2vol(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation) : Base_Layer<vol2col_param>(2)
	{
		m_type = "col2vol";
		m_param = {
			0, 1, channels, kernel.h, kernel.w, kernel.d, pad.h, pad.w, pad.d, stride.h, stride.w, stride.d, dilation.h,
			dilation.w, dilation.d, 0, 0, 0, 0, 0, 0
		};
	}

	void col2vol::computeGroupCount()
	{
		size_t tmp = m_param.channels;
		tmp *= static_cast<size_t>(m_param.kernel_h);
		tmp *= static_cast<size_t>(m_param.kernel_w);
		tmp *= static_cast<size_t>(m_param.kernel_d);

		m_group_x = static_cast<int>(alignSize(tmp, local_sz_x_conv)) / local_sz_x_conv;
		if (m_group_x > max_compute_work_group_count)
			m_group_x = max_compute_work_group_count;
		m_group_y = static_cast<int>(alignSize(m_param.batchsize, local_sz_y_conv)) / local_sz_y_conv;
		if (m_group_y > max_compute_work_group_count)
			m_group_y = max_compute_work_group_count;
		m_group_z = 1;
	}

	std::shared_ptr<tensor>& col2vol::operator()(const std::shared_ptr<tensor>& x_)
	{
		if (m_pipeline == nullptr)
		{
			const float depth = static_cast<float>(x_->getShape()[x_->getShape().size() - 3]);
			const float height = static_cast<float>(x_->getShape()[x_->getShape().size() - 2]);
			const float width = static_cast<float>(x_->getShape()[x_->getShape().size() - 1]);
			m_param.batchsize = x_->getShape()[0];

			m_param.depth_col = depth;
			m_param.height_col = height;
			m_param.width_col = width;
			m_param.depth_vol = (depth - 1) * m_param.stride_d - 2 * m_param.pad_d + m_param.dilation_d * (m_param.kernel_d - 1) + m_param.pad_d + 1;
			m_param.height_vol = (height - 1) * m_param.stride_h - 2 * m_param.pad_h + m_param.dilation_h * (m_param.kernel_h - 1) + m_param.pad_h + 1;
			m_param.width_vol = (width - 1) * m_param.stride_w - 2 * m_param.pad_w + m_param.dilation_w * (m_param.kernel_w - 1) + m_param.pad_w + 1;
		}
		const int n_out_plane = static_cast<int>(m_param.channels * m_param.kernel_d * m_param.kernel_h * m_param.kernel_w);
		const int output_length = static_cast<int>(m_param.batchsize * m_param.depth_vol * m_param.height_vol * m_param.width_vol);
		layer_construct_forward(kernel::shaders::col2vol_spv, sizeof(kernel::shaders::col2vol_spv), x_, Format::kFormatFp32,
			std::vector<int>{n_out_plane, output_length	});

		float* t = (float*)y->toHost();
		std::cout << std::endl;
		for (int i = 0; i < n_out_plane; ++i)
		{
			std::cout << "[ ";
			for (int j = 0; j < output_length; ++j)
			{
				std::cout << t[i * output_length + j] << ", ";
			}
			std::cout << "]" << std::endl;
		}

		return y;
	}

	std::vector<int> col2vol::output_shape() const
	{
		int d = static_cast<int>(m_param.depth_vol);
		int h = static_cast<int>(m_param.height_vol);
		int w = static_cast<int>(m_param.width_vol);
		return std::vector<int>{d, h, w};
	}

	copy::copy() : Base_Layer<>(2)
	{
		m_type = "copy";
	}

	std::shared_ptr<tensor>& copy::operator()(const std::shared_ptr<tensor>& x)
	{
		return layer_construct_forward(kernel::shaders::unary_operator_spv, sizeof(kernel::shaders::unary_operator_spv), x);
	}

	void copy::computeGroupCount()
	{
		m_group_x = static_cast<int>(alignSize(m_param.total, local_sz_x)) / local_sz_x;
		if (m_group_x > max_compute_work_group_count)
			m_group_x = max_compute_work_group_count;
		m_group_y = 1;
		m_group_z = 1;
	}

	std::vector<int> prepareStrides(const Shape& shape_before, const Shape& shape_after, Shape& stride)
	{
		size_t dims = shape_before.size();
		stride[2 * dims - 1] = 1;
		stride[3 * dims - 1] = 1;

		for (int64_t i = dims - 2; i >= 0; i--)
		{
			stride[dims * 2 + i] = stride[dims * 2 + i + 1] * shape_before[i + 1];
			stride[dims + i] = stride[dims + i + 1] * shape_after[i + 1];
		}
		return stride;
	}

	transpose::transpose(const std::vector<int> order) : Base_Layer<transpose_param>(2)
	{
		m_type = "transpose";
		m_param.num_axes = static_cast<int>(order.size());
		stride.resize(order.size() * 3);
		d_stride.resize(order.size() * 3);
		for (size_t i = 0; i < m_param.num_axes; ++i)
			stride[i] = order[i];
		for (size_t i = 0; i < m_param.num_axes; ++i)
			d_stride[i] = order[i];
		bck_shader = kernel::shaders::transpose_spv;
		bck_codeSize = sizeof(kernel::shaders::transpose_spv);
	}

	std::shared_ptr<tensor>& transpose::operator()(const std::shared_ptr<tensor>& _x)
	{
		if (!w || !new_shape.size())
		{
			new_shape.resize(stride.size() / 3);
			for (size_t i = 0; i < m_param.num_axes; ++i)
				new_shape[i] = _x->getShape()[stride[i]];
			old_shape = _x->getShape();;
			stride = prepareStrides(old_shape, new_shape, stride);
			w = std::make_shared<tensor>(tensor((char*)stride.data(), std::vector<int>{m_param.num_axes * 3}, Format::kFormatInt32));
		}
		return layer_construct_forward(kernel::shaders::transpose_spv, sizeof(kernel::shaders::transpose_spv), _x, w, Format::kFormatFp32, new_shape);
	}

	int transpose::set_backward()
	{
		if (!dw)
		{
			d_stride = prepareStrides(new_shape, old_shape, d_stride);
			dw = std::make_shared<tensor>(tensor((char*)d_stride.data(), std::vector<int>{m_param.num_axes * 3}, Format::kFormatInt32));
		}
		dx = derivative->layer_construct_forward(bck_shader, bck_codeSize, dy, dw, Format::kFormatFp32, old_shape);
		return dy->getId();
	}

	void transpose::computeGroupCount()
	{
		m_group_x = static_cast<int>(alignSize(m_param.total, local_sz_x)) / local_sz_x;
		if (m_group_x > max_compute_work_group_count)
			m_group_x = max_compute_work_group_count;
		m_group_y = 1;
		m_group_z = 1;
	}
}