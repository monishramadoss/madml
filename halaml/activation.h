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

		class activation_fn : public unary_operator
		{
		public:
			activation_fn(float alpha, bool in_place);
		protected:
			activation_param m_param;
		};

		namespace activation
		{
			class celu : public activation_fn
			{
			public:
				std::shared_ptr<tensor> alpha;
				explicit celu(float alpha, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class elu : public activation_fn
			{
			public:
				explicit elu(float alpha, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class hardshrink : public activation_fn
			{
			public:
				explicit hardshrink(float lambda, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			struct two_param
			{
				int total;
				float alpha;
				float beta;
			};

			class hardtanh : public activation_fn
			{
				two_param m_param;
			public:
				explicit hardtanh(float min_val = -1, float max_val = 1, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class leakyrelu : public activation_fn
			{
			public:
				explicit leakyrelu(float slope = -0.01, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class logsigmoid : public activation_fn
			{
			public:
				explicit logsigmoid(float alpha = -0.01, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class prelu : public activation_fn
			{
			public:
				explicit prelu(float alpha = -0.01, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class relu : public activation_fn
			{
			public:
				explicit relu(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class relu6 : public activation_fn
			{
			public:
				explicit relu6(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class selu : public activation_fn
			{
			public:
				explicit selu(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class sigmoid : public activation_fn
			{
			public:
				explicit sigmoid(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class softplus : public activation_fn
			{
			public:
				explicit softplus(float alpha, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class softshrink : public activation_fn
			{
			public:
				explicit softshrink(float alpha, bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class softsign : public activation_fn
			{
			public:
				explicit softsign(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};

			class tanhshrink : public activation_fn
			{
			public:
				explicit tanhshrink(bool in_place = false);
				std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) override;
			};
		}
	}
}

#endif //!activation
