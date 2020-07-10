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
			int m;
			int n;
			int k;
		};

		class matmul : public layer, public Module
		{
		private:
			void computeGroupCount() override;
			matmul_param m_param{};
		public:
			matmul();
			tensor* forward(tensor* x, tensor* w);

			void update_weight() override
			{
			};
		};
	}
}

#endif
