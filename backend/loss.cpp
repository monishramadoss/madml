#include "common.h"
#include "utils.h"
#include "loss.h"

namespace loss
{
	Loss::Loss() : Base_Layer<loss_param>(3)
	{
		m_type = "Loss";
	}

	void Loss::hook(const uint32_t* shader, size_t codeSize, const std::shared_ptr<tensor>& _x, const std::shared_ptr<tensor>& _w)
	{
		layer_construct_forward(shader, codeSize, _x, _w);

		if (!m1)
		{
			m1 = get_input_id(_w->getId());
			m1->dy = y;
		}
		recordCommandBuffer(static_cast<void*>(&m_param), sizeof(loss_param));
		runCommandBuffer();
	}

	void Loss::backward()
	{
		auto M = get_module();
		for (auto it = M.rbegin(); it != M.rend(); ++it)
		{
			(*it)->set_backward();
		}
	}

	MSE::MSE()
	{
	}

	void MSE::operator()(const std::shared_ptr<tensor>& y_true, const std::shared_ptr<tensor>& y_pred)
	{
		return hook(kernel::shaders::d_MSE_spv, sizeof(kernel::shaders::d_MSE_spv), y_true, y_pred);
	}
}