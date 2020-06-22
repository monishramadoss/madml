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
					bool operator()(tensor* x, tensor* y) override { return forward(x, y); };

					void backward() override
					{
					}

					void update_weight() override
					{
					};

				protected:
					virtual bool computeGroupCount();
					size_t m_total;
					static std::vector<Module*> module_list;
					std::vector<Module*>* get_module() override;
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

				class sigmoid : public ActivationFn
				{
				public:
					sigmoid();
					bool forward(tensor* x, tensor* y) override;
				};
			}
		}
	}
}

namespace kernel
{
	namespace layers
	{
		namespace nn
		{
			namespace activation
			{
				class acos : public ActivationFn
				{
				public:
					acos();
					bool forward(tensor* x, tensor* y) override;
				};

				class acosh : public ActivationFn
				{
				public:
					acosh();
					bool forward(tensor* x, tensor* y) override;
				};

				class asin : public ActivationFn
				{
				public:
					asin();
					bool forward(tensor* x, tensor* y) override;
				};

				class asinh : public ActivationFn
				{
				public:
					asinh();
					bool forward(tensor* x, tensor* y) override;
				};

				class atan : public ActivationFn
				{
				public:
					atan();
					bool forward(tensor* x, tensor* y) override;
				};

				class atanh : public ActivationFn
				{
				public:
					atanh();
					bool forward(tensor* x, tensor* y) override;
				};

				class cos : public ActivationFn
				{
				public:
					cos();
					bool forward(tensor* x, tensor* y) override;
				};

				class cosh : public ActivationFn
				{
				public:
					cosh();
					bool forward(tensor* x, tensor* y) override;
				};

				class sin : public ActivationFn
				{
				public:
					sin();
					bool forward(tensor* x, tensor* y) override;
				};

				class sinh : public ActivationFn
				{
				public:
					sinh();
					bool forward(tensor* x, tensor* y) override;
				};

				class tan : public ActivationFn
				{
				public:
					tan();
					bool forward(tensor* x, tensor* y) override;
				};

				class tanh : public ActivationFn
				{
				public:
					tanh();
					bool forward(tensor* x, tensor* y) override;
				};
			}
		}
	}
}

#endif //!activation
