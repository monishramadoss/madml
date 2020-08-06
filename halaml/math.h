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
			// Unary Operators
			class abs : public unary_operator
			{
			public:
				abs(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class ceil : public unary_operator
			{
			public:
				ceil(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
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
				clip(float min = 0.0f, float max = 1.0f, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class exp : public unary_operator
			{
			public:
				exp(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class floor : public unary_operator
			{
			public:
				floor(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class ln : public unary_operator
			{
			public:
				ln(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class round : public unary_operator
			{
			public:
				round(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class sqrt : public unary_operator
			{
			public:
				sqrt(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class acos : public unary_operator
			{
			public:
				acos(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class acosh : public unary_operator
			{
			public:
				acosh(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class asin : public unary_operator
			{
			public:
				asin(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class asinh : public unary_operator
			{
			public:
				asinh(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class atan : public unary_operator
			{
			public:
				atan(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class atanh : public unary_operator
			{
			public:
				atanh(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class cos : public unary_operator
			{
			public:
				cos(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class cosh : public unary_operator
			{
			public:
				cosh(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class sin : public unary_operator
			{
			public:
				sin(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class sinh : public unary_operator
			{
			public:
				sinh(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class tan : public unary_operator
			{
			public:
				tan(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class tanh : public unary_operator
			{
			public:
				tanh(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			// BINARY OPERATORS
			class add : public binary_operator
			{
			public:
				add(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::add* f_prime;
				void set_derivative() override;
			};

			class sub : public binary_operator
			{
			public:
				sub(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::sub* f_prime;
			};

			class mul : public binary_operator
			{
			public:
				mul(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::mul* f_prime;
			};

			class div : public binary_operator
			{
			public:
				div(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::div* f_prime;
			};

			class mod : public binary_operator
			{
			public:
				mod(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::mod* f_prime;
			};

			class pow : public binary_operator
			{
			public:
				pow(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::pow* f_prime;
			};

			class max : public binary_operator
			{
			public:
				max(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::max* f_prime;
			};

			class min : public binary_operator
			{
			public:
				min(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::min* f_prime;
			};

			class eq : public binary_operator
			{
			public:
				eq(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::eq* f_prime;
			};

			class ne : public binary_operator
			{
			public:
				ne(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::ne* f_prime;
			};

			class lt : public binary_operator
			{
			public:
				lt(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::lt* f_prime;
			};

			class le : public binary_operator
			{
			public:
				le(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::le* f_prime;
			};

			class gt : public binary_operator
			{
			public:
				gt(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::gt* f_prime;
			};

			class ge : public binary_operator
			{
			public:
				ge(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::ge* f_prime;
			};

			class xr : public binary_operator
			{
			public:
				xr(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) override;
				derivative::math::xr* f_prime;
			};
		}
	}
}

#endif //MATH_H
