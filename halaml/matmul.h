#ifndef MATMUL_H
#define MATMUL_H

#include "madml.h"
#include "layer.h"

namespace kernel
{
	namespace layers
	{
		struct matmul_param
		{
			int total;
			int m;
			int n;
			int k;
		};

		class matmul : public Base_Layer
		{
		private:
			void computeGroupCount() override;
			matmul_param m_param{};
		public:
			matmul();
			std::shared_ptr<tensor> forward(std::shared_ptr<tensor> x, std::shared_ptr<tensor> w);
			void back_propagate() override;
		};
	}
}

#endif
