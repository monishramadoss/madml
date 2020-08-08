#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel
{
	namespace layers
	{
		struct activation_param
		{
			int total;
			float alpha;
		};

		namespace activation
		{
			class celu : public Base_Layer<activation_param>
			{
			public:
				std::shared_ptr<tensor> alpha;
				explicit celu(float alpha, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class elu : public Base_Layer<activation_param>
			{
			public:
				explicit elu(float alpha, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class hardshrink : public Base_Layer<activation_param>
			{
			public:
				explicit hardshrink(float lambda, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			struct two_param
			{
				int total;
				float alpha;
				float beta;
			};

			class hardtanh : public Base_Layer<two_param>
			{
			public:
				explicit hardtanh(float min_val = -1, float max_val = 1, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class leakyrelu : public Base_Layer<activation_param>
			{
			public:
				explicit leakyrelu(float slope = -0.01, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class logsigmoid : public Base_Layer<activation_param>
			{
			public:
				explicit logsigmoid(float alpha = -0.01, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class prelu : public Base_Layer<activation_param>
			{
			public:
				explicit prelu(float alpha = -0.01, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class relu : public Base_Layer<activation_param>
			{
			public:
				explicit relu(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class relu6 : public Base_Layer<activation_param>
			{
			public:
				explicit relu6(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class selu : public Base_Layer<activation_param>
			{
			public:
				explicit selu(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class sigmoid : public Base_Layer<activation_param>
			{
			public:
				explicit sigmoid(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class softplus : public Base_Layer<activation_param>
			{
			public:
				explicit softplus(float alpha, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class softshrink : public Base_Layer<activation_param>
			{
			public:
				explicit softshrink(float alpha, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class softsign : public Base_Layer<activation_param>
			{
			public:
				explicit softsign(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};

			class tanhshrink : public Base_Layer<activation_param>
			{
			public:
				explicit tanhshrink(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x);
			};
		}
	}
}

#endif //!activation
