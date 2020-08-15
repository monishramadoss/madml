#include "common.h"
#include "utils.h"
#include "transform.h"

#define LOCAL_SZ_X 16
#define LOCAL_SZ_Y 64
#define MAX_COMPUTE_WORK_GROUP_COUNT 65535

namespace layers
{
	vol2col::vol2col(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation) : Base_Layer<vol2col_param>(2)

	{
		m_type = "vol2col";
		m_param = {
			0, 1, channels, kernel.h, kernel.w, kernel.d, pad.h, pad.w, pad.d, stride.h, stride.w, stride.d, dilation.h,
			dilation.w, dilation.d, 0, 0, 0, 0, 0, 0
		};
		bck_shader = kernel::shaders::col2vol_spv;
		bck_codeSize = sizeof(kernel::shaders::col2vol_spv);
	}

	void vol2col::computeGroupCount()
	{
		size_t tmp = m_param.channels;
		tmp *= m_param.kernel_h;
		tmp *= m_param.kernel_w;
		tmp *= m_param.kernel_d;

		m_group_x = static_cast<int>(alignSize(tmp, LOCAL_SZ_X)) / LOCAL_SZ_X;
		if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
			m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
		m_group_y = static_cast<int>(alignSize(m_param.batchsize, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
		if (m_group_y > MAX_COMPUTE_WORK_GROUP_COUNT)
			m_group_y = MAX_COMPUTE_WORK_GROUP_COUNT;
		m_group_z = 1;
	}

	std::shared_ptr<tensor>& vol2col::operator()(const std::shared_ptr<tensor>& x)
	{
		if (m_pipeline == nullptr)
		{
			const int depth = x->getShape()[x->getShape().size() - 3];
			const int height = x->getShape()[x->getShape().size() - 2];
			const int width = x->getShape()[x->getShape().size() - 1];
			m_param.batchsize = 1;
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
		const int n_out_plane = m_param.channels * m_param.kernel_d * m_param.kernel_h * m_param.kernel_w;
		const int output_length = m_param.depth_col * m_param.height_col * m_param.width_col;
		y = layer_construct_forward(kernel::shaders::vol2col_spv, sizeof(kernel::shaders::vol2col_spv), x, Format::kFormatFp32, std::vector<int>{output_length* n_out_plane});
		y->reshape(std::vector<int>{n_out_plane, output_length});
		return y;
	}

	std::vector<int> vol2col::output_shape() const
	{
		return std::vector<int>{m_param.depth_col, m_param.height_col, m_param.width_col};
	}

	col2vol::col2vol(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation) : Base_Layer<vol2col_param>(2)
	{
		m_type = "col2vol";
		m_param = {
			0, 1, channels, kernel.h, kernel.w, kernel.d, pad.h, pad.w, pad.d, stride.h, stride.w, stride.d, dilation.h,
			dilation.w, dilation.d, 0, 0, 0, 0, 0, 0
		};
		bck_shader = kernel::shaders::vol2col_spv;
		bck_codeSize = sizeof(kernel::shaders::vol2col_spv);
	}

	void col2vol::computeGroupCount()
	{
		size_t tmp = m_param.channels;
		tmp *= m_param.kernel_h;
		tmp *= m_param.kernel_w;
		tmp *= m_param.kernel_d;

		m_group_x = static_cast<int>(alignSize(tmp, LOCAL_SZ_X)) / LOCAL_SZ_X;
		if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
			m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
		m_group_y = static_cast<int>(alignSize(m_param.batchsize, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
		if (m_group_y > MAX_COMPUTE_WORK_GROUP_COUNT)
			m_group_y = MAX_COMPUTE_WORK_GROUP_COUNT;
		m_group_z = 1;
	}

	std::shared_ptr<tensor>& col2vol::operator()(const std::shared_ptr<tensor>& x)
	{
		if (m_pipeline == nullptr)
		{
			const int depth = x->getShape()[x->getShape().size() - 3];
			const int height = x->getShape()[x->getShape().size() - 2];
			const int width = x->getShape()[x->getShape().size() - 1];
			m_param.batchsize = 1;
			m_param.depth_col = depth;
			m_param.height_col = height;
			m_param.width_col = width;
			m_param.depth_vol = (depth - 1) * m_param.stride_d - 2 * m_param.pad_d + m_param.dilation_d * (m_param.kernel_d - 1)
				+ m_param.pad_d + 1;
			m_param.height_vol = (height - 1) * m_param.stride_h - 2 * m_param.pad_h + m_param.dilation_h * (m_param.kernel_h -
				1) + m_param.pad_h + 1;
			m_param.width_vol = (width - 1) * m_param.stride_w - 2 * m_param.pad_w + m_param.dilation_w * (m_param.kernel_w - 1)
				+ m_param.pad_w + 1;
		}
		const int n_out_plane = x->getShape()[0] * (m_param.kernel_d * m_param.kernel_h * m_param.kernel_w);
		const int output_length = m_param.depth_vol * m_param.height_vol * m_param.width_vol;

		y = layer_construct_forward(kernel::shaders::col2vol_spv, sizeof(kernel::shaders::col2vol_spv), x, Format::kFormatFp32,
			std::vector<int>{
			n_out_plane* (m_param.depth_vol* m_param.height_vol* m_param.width_vol)
		});
		y->reshape(std::vector<int>{n_out_plane, output_length});
		return y;
	}

	std::vector<int> col2vol::output_shape() const
	{
		return std::vector<int>{m_param.depth_vol, m_param.height_vol, m_param.width_vol};
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
		m_group_x = static_cast<int>(alignSize(m_param.total, 1024)) / 1024;
		if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
			m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
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

	transpose::transpose(const std::vector<int> order) : Base_Layer<transpose_param>(4)
	{
		m_type = "transpose";
		m_param.num_axes = static_cast<int>(order.size());
		stride.resize(order.size() * 3);
		for (size_t i = 0; i < m_param.num_axes; ++i)
			stride[i] = order[i];
		bck_shader = kernel::shaders::transpose_spv;
		bck_codeSize = sizeof(kernel::shaders::transpose_spv);
	}

	std::shared_ptr<tensor>& transpose::operator()(const std::shared_ptr<tensor>& _x)
	{
		if (!w || !new_shape.size())
		{
			new_shape.resize(stride.size());
			for (size_t i = 0; i < m_param.num_axes; ++i)
				new_shape[i] = _x->getShape()[stride[i]];
			old_shape = _x->getShape();
			stride = prepareStrides(old_shape, new_shape, stride);
			w = std::make_shared<tensor>(tensor((char*)stride.data(), std::vector<int>{m_param.num_axes * 3}, Format::kFormatInt32));
		}
		return layer_construct_forward(kernel::shaders::transpose_spv, sizeof(kernel::shaders::transpose_spv), x, w, Format::kFormatFp32, new_shape);
	}

	void transpose::computeGroupCount()
	{
		m_group_x = static_cast<int>(alignSize(m_param.total, 1024)) / 1024;
		if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
			m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
		m_group_y = 1;
		m_group_z = 1;
	}
}