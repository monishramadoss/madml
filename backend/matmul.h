#ifndef MATMUL_H
#define MATMUL_H

#include "backend.h"
#include "layer.h"

namespace layers
{
	struct matmul_param
	{
		int total;
		int batchsize;
		int m;
		int n;
		int k;
	};

	class matmul : public Base_Layer<matmul_param>
	{
	private:
		void computeGroupCount() override;
		layers::transpose* t;
	public:
		explicit matmul();
		std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
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

			matmul* mm;
			math::add* bias;
		};
	}
}

#endif
