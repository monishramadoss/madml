#include "common.h"
#include "utils.h"
#include "loss.h"
#include <queue>

namespace loss
{
	Loss::Loss() : Base_Layer<loss_param>(3)
	{
		m_type = "Loss";
	}

	void Loss::hook(const uint32_t* shader, size_t codeSize, const std::shared_ptr<tensor>& _x,
		const std::shared_ptr<tensor>& _w)
	{
		layer_construct_forward(shader, codeSize, _x, _w);
		if (!m1)
		{
			m1 = get_input_id(_w->getId());
			m1->dy = y;
		}
	}

	void Loss::backward()
	{
		std::queue<Module*> traversal_order;
		traversal_order.push(m1);
		while (!traversal_order.empty())
		{
			Module* tmp = traversal_order.back();
			if (tmp != nullptr)
				tmp->set_backward();
			traversal_order.pop();
			if (tmp->m1 != nullptr)
				traversal_order.push(tmp->m1);
			if (tmp->m2 != nullptr)
				traversal_order.push(tmp->m2);
		}
	}

	MSE::MSE()
	{
		m_type = "MSE";
	}

	void MSE::operator()(const std::shared_ptr<tensor>& y_true, const std::shared_ptr<tensor>& y_pred)
	{
		return hook(kernel::shaders::d_MSE_spv, sizeof(kernel::shaders::d_MSE_spv), y_true, y_pred);
	}
}