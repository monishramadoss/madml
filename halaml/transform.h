#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vector>
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

		class vol2col : public layer, public Module
		{
		public:
			vol2col(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation);
			virtual bool forward(tensor* x, tensor* y);
			virtual void reshapeOutTensor(tensor* x, tensor* z);
			bool forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) override;

			void backward() override
			{
			}

			void update_weight() override
			{
			}

			bool operator()(tensor* x, tensor* y) override { return false; };

			std::vector<int> calc_output_shape();

		private:
			virtual bool computeGroupCount();
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

			static std::vector<Module*> module_list;
			std::vector<Module*>* get_module() override;
		};

		class col2vol : public layer, public Module
		{
		public:
			col2vol(int channels, dhw kernel, dhw pad, dhw stride, dhw dilation);
			virtual bool forward(tensor* x, tensor* y);
			virtual void reshapeOutTensor(tensor* x, tensor* z);
			bool forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) override;

			void backward() override
			{
			}

			void update_weight() override
			{
			}

			bool operator()(tensor* x, tensor* y) override { return false; };

			std::vector<int> calc_output_shape();

		private:
			virtual bool computeGroupCount();
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

			static std::vector<Module*> module_list;
			std::vector<Module*>* get_module() override;
		};
	}
}

#endif //!TRANSFORM_H
