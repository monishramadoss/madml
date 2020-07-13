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
			struct operator_param
			{
				int total;
			};

			class unary_operator : public layer, public Module
			{
			protected:
				bool as_module;
				bool m_inplace;
				operator_param m_param;
				inline tensor* layer_construct_forward(const uint32_t* shader, size_t codeSize, tensor* x);
				inline tensor* layer_construct_backward(const uint32_t* shader, size_t codeSize, tensor* x);
				void computeGroupCount() override;

			public:
				unary_operator(bool in_place, bool as_module = true);
				virtual tensor* forward(tensor* x) = 0;
				void update_weight() override;
			};

			class binary_operator : public layer, public Module
			{
			protected:
				bool as_module;
				bool m_inplace;
				operator_param m_param;
				inline tensor* layer_construct_forward(const uint32_t* shader, size_t codeSize, tensor* x, tensor* w);
				inline tensor* layer_construct_backward(const uint32_t* shader, size_t codeSize, tensor* x, tensor* w);
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
			};

			class ceil : public unary_operator
			{
			public:
				ceil(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			struct clip_operator_param
			{
				int total;
				float min;
				float max;
			};

			class clip : public unary_operator
			{
				float m_min;
				float m_max;
			public:
				clip(float min = 0.0f, float max = 1.0f, bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class exp : public unary_operator
			{
			public:
				exp(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class floor : public unary_operator
			{
			public:
				floor(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class log : public unary_operator
			{
			public:
				log(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class round : public unary_operator
			{
			public:
				round(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class sqrt : public unary_operator
			{
			public:
				sqrt(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class acos : public unary_operator
			{
			public:
				acos(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class acosh : public unary_operator
			{
			public:
				acosh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class asin : public unary_operator
			{
			public:
				asin(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class asinh : public unary_operator
			{
			public:
				asinh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class atan : public unary_operator
			{
			public:
				atan(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class atanh : public unary_operator
			{
			public:
				atanh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class cos : public unary_operator
			{
			public:
				cos(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class cosh : public unary_operator
			{
			public:
				cosh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class sin : public unary_operator
			{
			public:
				sin(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class sinh : public unary_operator
			{
			public:
				sinh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class tan : public unary_operator
			{
			public:
				tan(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			class tanh : public unary_operator
			{
			public:
				tanh(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x) override;
			};

			// BINARY OPERATORS
			class add : public binary_operator
			{
			public:
				add(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class sub : public binary_operator
			{
			public:
				sub(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class mul : public binary_operator
			{
			public:
				mul(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class div : public binary_operator
			{
			public:
				div(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class mod : public binary_operator
			{
			public:
				mod(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class pow : public binary_operator
			{
			public:
				pow(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class max : public binary_operator
			{
			public:
				max(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class min : public binary_operator
			{
			public:
				min(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class eq : public binary_operator
			{
			public:
				eq(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class ne : public binary_operator
			{
			public:
				ne(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class lt : public binary_operator
			{
			public:
				lt(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class le : public binary_operator
			{
			public:
				le(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class gt : public binary_operator
			{
			public:
				gt(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class ge : public binary_operator
			{
			public:
				ge(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};

			class xr : public binary_operator
			{
			public:
				xr(bool in_place = false, bool as_module = true);
				tensor* forward(tensor* x, tensor* y) override;
			};
		}
	}
}

#endif //MATH_H
