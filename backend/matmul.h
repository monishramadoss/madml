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
		int gemm_2;
	};

	class matmul : public Base_Layer<matmul_param>
	{
	private:
		void computeGroupCount() override;
	public:
		matmul();
		std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
		int set_backward() override;
	};
}

#endif
