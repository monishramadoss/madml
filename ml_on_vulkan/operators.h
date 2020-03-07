#ifndef OPERATORS_H
#define OPERATORS_H

#include "kernel.h"
#include "layer.h"

namespace kernel{
	namespace layers {
		class operators : public layer
		{
		public:
			operators(int op_id);
			bool forward(tensor& in, tensor& in2, tensor& out);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
			virtual bool run();
		private:
			bool computeGroupCount();
			int m_total;
			int m_op;
		};	

	}
}

#endif
