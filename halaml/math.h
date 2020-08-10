#ifndef MATH_H
#define MATH_H

#include <vector>
#include <future>
#include "madml.h"
#include "layer.h"

namespace kernel
{
	namespace layers
	{
		namespace math
		{
			// Unary Operators
			class abs : public Base_Layer<>
			{
			public:
				abs(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class ceil : public Base_Layer<>
			{
			public:
				ceil(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			struct clip_operator_param
			{
				int total;
				float min;
				float max;
			};

			class clip : public Base_Layer<clip_operator_param>
			{
			public:
				clip(float min = 0.0f, float max = 1.0f, bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class exp : public Base_Layer<>
			{
			public:
				exp(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class floor : public Base_Layer<>
			{
			public:
				floor(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class ln : public Base_Layer<>
			{
			public:
				ln(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class round : public Base_Layer<>
			{
			public:
				round(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class sqrt : public Base_Layer<>
			{
			public:
				sqrt(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class acos : public Base_Layer<>
			{
			public:
				acos(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class acosh : public Base_Layer<>
			{
			public:
				acosh(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class asin : public Base_Layer<>
			{
			public:
				asin(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class asinh : public Base_Layer<>
			{
			public:
				asinh(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class atan : public Base_Layer<>
			{
			public:
				atan(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class atanh : public Base_Layer<>
			{
			public:
				atanh(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class cos : public Base_Layer<>
			{
			public:
				cos(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class cosh : public Base_Layer<>
			{
			public:
				cosh(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class sin : public Base_Layer<>
			{
			public:
				sin(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class sinh : public Base_Layer<>
			{
			public:
				sinh(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class tan : public Base_Layer<>
			{
			public:
				tan(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			class tanh : public Base_Layer<>
			{
			public:
				tanh(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			};

			// BINARY OPERATORS
			class add : public Base_Layer<>
			{
			public:
				add(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class sub : public Base_Layer<>
			{
			public:
				sub(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class mul : public Base_Layer<>
			{
			public:
				mul(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class div : public Base_Layer<>
			{
			public:
				div(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class mod : public Base_Layer<>
			{
			public:
				mod(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class pow : public Base_Layer<>
			{
			public:
				pow(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class max : public Base_Layer<>
			{
			public:
				max(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class min : public Base_Layer<>
			{
			public:
				min(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class eq : public Base_Layer<>
			{
			public:
				eq(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class ne : public Base_Layer<>
			{
			public:
				ne(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class lt : public Base_Layer<>
			{
			public:
				lt(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class le : public Base_Layer<>
			{
			public:
				le(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class gt : public Base_Layer<>
			{
			public:
				gt(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class ge : public Base_Layer<>
			{
			public:
				ge(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};

			class xr : public Base_Layer<>
			{
			public:
				xr(bool in_place = false);
				std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
			};
		}
	}
}

#endif //MATH_H
