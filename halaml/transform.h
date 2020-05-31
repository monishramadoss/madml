#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel {
	namespace layers {
		class im2col : public layer {
		public:
            im2col(int batchsize, int channels, int kernel[2], int pad[2], int stride[2], int dilation[2]);
            virtual bool forward(tensor& x, tensor& y) = 0;
            virtual void reshapeOutTensor(tensor& x, tensor& z);
            virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
		private:
            virtual bool computeGroupCount();
            int batchsize;
            int channels;
            int kernel_h;
            int kernel_w;
            int pad_h;
            int pad_w;
            int stride_h;
            int stride_w;
            int dilation_h;
            int dilation_w;

            int height_col; // height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
            int width_col;  // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
            int height_im;
            int width_im;
		};

        class col2im : public layer {
        public:
            col2im(int batchsize, int channels, int kernel[2], int pad[2], int stride[2], int dilation[2]);
            virtual bool forward(tensor& x, tensor& y) = 0;
            virtual void reshapeOutTensor(tensor& x, tensor& z);
            virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
        private:
            virtual bool computeGroupCount();
            int batchsize;
            int channels;
            int kernel_h;
            int kernel_w;
            int pad_h;
            int pad_w;
            int stride_h;
            int stride_w;
            int dilation_h;
            int dilation_w;

            int height_col; // height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
            int width_col;  // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
            int height_im;
            int width_im;
        };
	}
}



#endif //!TRANSFORM_H