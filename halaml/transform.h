#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vector>
#include <utility>
#include "madml.h"
#include "layer.h"

namespace kernel
{
	namespace layers
	{
		struct dhw
		{
			int d;
			int h;
			int w;
		};

		struct vol2col_param
		{
			int total;
			int batchsize;
			int channels;
			int kernel_h;
			int kernel_w;
			int kernel_d;
			int pad_h;
			int pad_w;
			int pad_d;
			int stride_h;
			int stride_w;
			int stride_d;
			int dilation_h;
			int dilation_w;
			int dilation_d;
			int height_col; // height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
			int width_col; // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
			int depth_col; // depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1
			int height_vol;
			int width_vol;
			int depth_vol;
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
			std::shared_ptr<tensor> tensor_stride;
			std::shared_ptr<tensor> d_tensor_stride;
		public:
			transpose(std::vector<int> order);
			std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
		};
	}
}

#endif //!TRANSFORM_H
