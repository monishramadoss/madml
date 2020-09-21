#ifndef MATMUL_H
#define MATMUL_H

#include "backend.h"
#include "layer.h"

namespace layers
{
	struct gemm_param
	{
		int total;
		int batchsize;
		float alpha;
		float beta;
		bool use_bias;
		int m;
		int n;
		int k;
	};

	class gemm : public Base_Layer<gemm_param>
	{
	private:
		void computeGroupCount() override;
		std::shared_ptr<layers::transpose> t;
	public:
		explicit gemm(float alpha, float beta, bool use_bias);
		std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
		std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w, std::shared_ptr<tensor>& b);

		int set_backward() override;
	};

	namespace nn
	{
		class dense : public Module
		{
		public:
			dense(int size, bool use_bias);
			std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
			int set_backward() override;
			void update_weight() override;

		private:
			int m_size;
			bool USE_BIAS;

			std::shared_ptr<gemm> mm;
			math::add* bias;
		};
	}
}

#endif
