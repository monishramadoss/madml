#ifndef OPERATORS_H
#define OPERATORS_H

#include "madml.h"
#include "layer.h"

namespace kernel{
	namespace layers {
		class operators : public layer, public Module
		{
		public:
			operators(size_t op_id);
			bool forward(tensor* x, tensor* y, tensor* z);
			bool forward(tensor* x, tensor* y);
			void reshapeOutTensor(tensor* x, tensor* z);
			bool forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs);
			void backward();
			bool operator() (tensor* x, tensor* y);
			~operators() {
				if(out != nullptr)
					delete out;
			}
		private:
			bool computeGroupCount();
			size_t m_total;
			size_t m_op;	
			tensor* out;
		};	

	}
}

#endif
