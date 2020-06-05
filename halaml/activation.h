#ifndef ACTIVATION
#define ACTIVATION

#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel {
	namespace nn {
		namespace activation {

			class Activationfn : public layer {
			public :
				Activationfn();
				virtual bool forward(tensor* x, tensor* y) = 0;
				void reshapeOutTensor(tensor* x, tensor* z);
				bool forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs);
			protected:
				virtual bool computeGroupCount();
				size_t m_total;
			};


			class elu : public Activationfn { 
			public:
				elu(float alpha);
				bool forward(tensor* x, tensor* y);
			private:
				float m_const;
			};

			class relu : public Activationfn {
			public:
				relu();
				bool forward(tensor* x, tensor* y);
			};


		}
	}
}

#endif //!activation