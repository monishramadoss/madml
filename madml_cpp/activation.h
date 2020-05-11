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
				virtual bool forward(tensor& x, tensor& y) = 0;
				virtual void reshapeOutTensor(tensor& x, tensor& z);
				virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
			private:
				virtual bool computeGroupCount();
				size_t m_total;
			};



			class elu : public Activationfn
			{
			public:
				elu(float alpha);
				bool forward(tensor& x, tensor& y);
				void reshapeOutTensor(tensor& x, tensor& z);
				virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
			private:
				bool computeGroupCount();
				size_t m_total;
				float m_const;
			};

		}
	}
}

#endif //!activation