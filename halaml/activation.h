#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel
{
	namespace layers
	{
		namespace activation
		{
			struct activation_param
			{
				int total;
				float alpha;
			};

			class unary_operator : public Base_Layer
			{
			protected:
				activation_param m_param;
				void computeGroupCount() override;

			public:
				unary_operator(float alpha, bool in_place, bool as_module = true);
				virtual tensor* forward(tensor* x) = 0;
			};

			class celu : public unary_operator
			{
			public:
				explicit celu(float alpha, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class elu : public unary_operator
			{
			public:
				explicit elu(float alpha, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class hardshrink : public unary_operator
			{
			public:
				explicit hardshrink(float lambda, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			struct two_param
			{
				int total;
				float alpha;
				float beta;
			};

			class hardtanh : public unary_operator
			{
				two_param m_param;
			public:
				explicit hardtanh(float min_val = -1, float max_val = 1, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class leakyrelu : public unary_operator
			{
			public:
				explicit leakyrelu(float slope = -0.01, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class logsigmoid : public unary_operator
			{
			public:
				explicit logsigmoid(float alpha = -0.01, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class prelu : public unary_operator
			{
			public:
				explicit prelu(float alpha = -0.01, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class relu : public unary_operator
			{
			public:
				explicit relu(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class relu6 : public unary_operator
			{
			public:
				explicit relu6(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class selu : public unary_operator
			{
			public:
				explicit selu(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class sigmoid : public unary_operator
			{
			public:
				explicit sigmoid(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class softplus : public unary_operator
			{
			public:
				explicit softplus(float alpha, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class softshrink : public unary_operator
			{
			public:
				explicit softshrink(float alpha, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class softsign : public unary_operator
			{
			public:
				explicit softsign(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class tanhshrink : public unary_operator
			{
			public:
				explicit tanhshrink(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};
		}
	}
}

#endif //!activation
