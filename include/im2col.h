#ifndef IM2COL
#define IM2COL

#include "madml.h"
#include "layer.h"

namespace kernel {
	namespace layers {
        class im2col : public layer
		{
		public:
			im2col();
			bool forward(tensor& x, tensor& y);
			void reshapeOutTensor(tensor& x, tensor& z);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
		};	
    }
}
#endif

#ifndef COL2IM
#define COL2IM

#include "madml.h"
#include "layer.h"

namespace kernel {
	namespace layers {
        class col2im : public layer
		{
		public:
            col2im();
			bool forward(tensor& x, tensor& y);
			void reshapeOutTensor(tensor& x, tensor& z);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
		};	
    }
}

#endif


