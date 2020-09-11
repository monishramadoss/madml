#include "common.h"
#include "utils.h"
#include "normalization.h"

namespace layers
{
	namespace normalization
	{
		BatchNormalization::BatchNormalization(float eps = 1e-05, float momentum = 0.1, int num_features = -1, bool affine = true) : Base_Layer<norm_param>(2)
		{
			m_type = "batch normalization";
			m_param.eps = eps;
			m_param.momentum = momentum;
		}

		void BatchNormalization::computeGroupCount()
		{
		}

		std::shared_ptr<tensor>& BatchNormalization::operator()(const std::shared_ptr<tensor>& x)
		{
			if (!y)
				y = std::make_shared<tensor>(0., x->getShape());

			return y;
		}

		int BatchNormalization::set_backward() {}

		InstanceNormalization::InstanceNormalization(float eps = 1e-05, float momentum = 0.1, int num_features = -1, bool affine = true) : Base_Layer<norm_param>(2)
		{
			m_type = "instance normalization";
			m_param.eps = eps;
			m_param.momentum = momentum;
		}

		void InstanceNormalization::computeGroupCount()
		{
		}

		std::shared_ptr<tensor>& InstanceNormalization::operator()(const std::shared_ptr<tensor>& x)
		{
			if (!y)
				y = std::make_shared<tensor>(0., x->getShape());

			return y;
		}

		int InstanceNormalization::set_backward() {}
	};
}
}