#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vector>
#include <utility>
#include "backend.h"
#include "layer.h"

namespace layers
{
	struct dhw
	{
		float d;
		float h;
		float w;
	};

	struct vol2col_param
	{
		int total;
		int batchsize;
		int channels;
		float kernel_h;
		float kernel_w;
		float kernel_d;
		float pad_h;
		float pad_w;
		float pad_d;
		float stride_h;
		float stride_w;
		float stride_d;
		float dilation_h;
		float dilation_w;
		float dilation_d;
		float height_col; // height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
		float width_col; // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
		float depth_col; // depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1
		float height_vol;
		float width_vol;
		float depth_vol;
	};

	class vol2col : public Base_Layer<vol2col_param>
	{
	private:
		void computeGroupCount() override;
	public:
		vol2col(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation);
		std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
		std::vector<int> output_shape() const;
	};

	class col2vol : public Base_Layer<vol2col_param>
	{
	private:
		void computeGroupCount() override;
	public:
		col2vol(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation);
		std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
		std::vector<int> output_shape() const;
	};

	class copy : public Base_Layer<>
	{
	private:
		void computeGroupCount() override;
	public:
		copy();
		std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
	};

	struct transpose_param
	{
		int total;
		int num_axes;
	};

	class transpose : public Base_Layer<transpose_param>
	{
	private:
		void computeGroupCount() override;
		std::vector<int> new_shape;
		std::vector<int> old_shape;
		std::vector<int> stride;
		std::vector<int> d_stride;
	public:
		transpose(std::vector<int> order);
		std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
		int set_backward() override;
	};
}

#endif //!TRANSFORM_H
