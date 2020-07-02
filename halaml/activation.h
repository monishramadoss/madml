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
			struct operator_param
			{
				int total;
				float alpha;
			};

			class unary_operator : public layer, public Module
			{
			protected:
				bool as_module;
				bool m_inplace;
				float m_alpha;
				operator_param m_param;
				tensor* layer_construct(const uint32_t* shader, size_t codeSize, tensor* x);
				void computeGroupCount() override;

			public:
				unary_operator(float alpha, bool in_place, bool as_module = true);
				virtual tensor* forward(tensor* x) = 0;
				virtual void update_weight() override;
			};

		

			class celu : public unary_operator
			{	
			public:
				celu(float alpha, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class elu : public unary_operator 
			{				
			public:
				elu(float alpha, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class hardshrink : public unary_operator
			{public:
				hardshrink(float lambda, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			struct two_param {
				int total;
				float alpha;
				float beta;
			};

			class hardtanh : public unary_operator 
			{
				two_param m_param;
			public:
				hardtanh(float min_val = -1, float max_val = 1, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);

			};


			class leakyrelu : public unary_operator
			{
			public:
				leakyrelu(float slope = -0.01, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);

			};


			class logsigmoid : public unary_operator
			{
			public:
				logsigmoid(float alpha = -0.01, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);

			};


			class prelu : public unary_operator
			{
			public:
				prelu(float alpha = -0.01, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);
			};

			class relu : public unary_operator
			{
			public:
				relu(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);
			};

			class relu6 : public unary_operator
			{
			public:
				relu6(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);
			};

			class selu : public unary_operator
			{
			public:
				selu(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);
			};

			class sigmoid : public unary_operator
			{
			public:
				sigmoid(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);
			};

			class softplus : public unary_operator
			{
			public:
				softplus(float alpha, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);
			};

			class softshrink : public unary_operator
			{
			public:
				softshrink(float alpha, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);
			};

			class softsign : public unary_operator
			{
			public:
				softsign(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);
			};

			class tanhshrink : public unary_operator
			{
			public:
				tanhshrink(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x);
			};

		}

	}
}

#endif //!activation