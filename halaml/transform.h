#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel {
	namespace layers {
		class im2col : public layer {
		public:
            im2col(int channels, int kernel[3], int pad[3], int stride[3], int dilation[3]);
            virtual bool forward(tensor* x, tensor* y) = 0;
            virtual void reshapeOutTensor(tensor* x, tensor* z);
            virtual bool forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs);
		private:
            virtual bool computeGroupCount();
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
            int width_col;  // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
            int depth_col;  // depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1
                
            int height_im;
            int width_im;
            int depth_im;
    	};

        class col2im : public layer {
        public:
            col2im(int channels, int kernel[3], int pad[3], int stride[3], int dilation[3]);
            virtual bool forward(tensor* x, tensor* y) = 0;
            virtual void reshapeOutTensor(tensor* x, tensor* z);
            virtual bool forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs);
        private:
            virtual bool computeGroupCount();
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
            int width_col;  // width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
            int depth_col;  // depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1
                
            int height_im;
            int width_im;
            int depth_im;
        };
	}
}



#endif //!TRANSFORM_H