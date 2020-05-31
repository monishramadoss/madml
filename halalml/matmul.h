#ifndef MATMUL_H
#define MATMUL_H

#include "madml.h"
#include "layer.h"

namespace kernel {
	namespace layers {
		class matmul : public layer
		{
		public:

			matmul();
			bool forward(tensor& x, tensor& y, tensor& z);
			void reshapeOutTensor(tensor& x, tensor& z);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_m;
			int m_n;
			int m_k;
		};

	}
}

#endif