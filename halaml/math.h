#ifndef MATH_H
#define MATH_H

#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel
{
	namespace layers
	{
		namespace math
		{
			
			class unary_operator : public Base_Layer
			{
			protected:
				void computeGroupCount() override;
			public:
				unary_operator(bool in_place, bool as_module = true);
				virtual tensor* forward(tensor* x) = 0;
				void update_weight() override;
			};

			class binary_operator : public Base_Layer
			{
			protected:
				void computeGroupCount() override;
			public:
				binary_operator(bool in_place, bool as_module = true);
				virtual tensor* forward(tensor* x, tensor* w) = 0;
				void update_weight() override;
			};
		}
	}
}

namespace kernel
{
	namespace layers
	{
		namespace math
		{
			// Unary Operators
			class abs : public unary_operator
			{
			public:
				abs(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class ceil : public unary_operator
			{
			public:
				ceil(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			struct clip_operator_param
			{
				int total;
				float min;
				float max;
			};

			class clip : public unary_operator
			{
				clip_operator_param m_param;
			public:
				clip(float min = 0.0f, float max = 1.0f, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class exp : public unary_operator
			{
			public:
				exp(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class floor : public unary_operator
			{
			public:
				floor(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class ln : public unary_operator
			{
			public:
				ln(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class round : public unary_operator
			{
			public:
				round(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class sqrt : public unary_operator
			{
			public:
				sqrt(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class acos : public unary_operator
			{
			public:
				acos(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class acosh : public unary_operator
			{
			public:
				acosh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class asin : public unary_operator
			{
			public:
				asin(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class asinh : public unary_operator
			{
			public:
				asinh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class atan : public unary_operator
			{
			public:
				atan(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class atanh : public unary_operator
			{
			public:
				atanh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class cos : public unary_operator
			{
			public:
				cos(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class cosh : public unary_operator
			{
			public:
				cosh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class sin : public unary_operator
			{
			public:
				sin(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class sinh : public unary_operator
			{
			public:
				sinh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class tan : public unary_operator
			{
			public:
				tan(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			class tanh : public unary_operator
			{
			public:
				tanh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
				void back_propagate() override;
			};

			// BINARY OPERATORS
			class add : public binary_operator
			{
			public:
				add(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class sub : public binary_operator
			{
			public:
				sub(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class mul : public binary_operator
			{
			public:
				mul(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class div : public binary_operator
			{
			public:
				div(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class mod : public binary_operator
			{
			public:
				mod(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class pow : public binary_operator
			{
			public:
				pow(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class max : public binary_operator
			{
			public:
				max(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class min : public binary_operator
			{
			public:
				min(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class eq : public binary_operator
			{
			public:
				eq(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class ne : public binary_operator
			{
			public:
				ne(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class lt : public binary_operator
			{
			public:
				lt(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class le : public binary_operator
			{
			public:
				le(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class gt : public binary_operator
			{
			public:
				gt(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class ge : public binary_operator
			{
			public:
				ge(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};

			class xr : public binary_operator
			{
			public:
				xr(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
				void back_propagate() override;
			};
		}
	}
}

#endif //MATH_H
