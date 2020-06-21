#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel
{
	namespace layers
	{
		namespace nn
		{
			namespace activation
			{
				class ActivationFn : public layer, public Module
				{
				public:
					ActivationFn();
					virtual bool forward(tensor* x, tensor* y) = 0;
					void reshapeOutTensor(tensor* x, tensor* z);
					bool forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) override;
				protected:
					virtual bool computeGroupCount();
					size_t m_total;
				};

				class elu : public ActivationFn
				{
				public:
					elu(float alpha);
					bool forward(tensor* x, tensor* y) override;
				private:
					float m_const;
				};

				class relu : public ActivationFn
				{
				public:
					relu();
					bool forward(tensor* x, tensor* y) override;
				};
			}
		}
	}
}

#endif //!activation
