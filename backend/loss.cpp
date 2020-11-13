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

    void Loss::hook(const std::shared_ptr<tensor>& _x, const std::shared_ptr<tensor>& _w)
    {
        layer_construct_forward(_x, _w);
    }

    void Loss::backward()
    {
    }

    MSE::MSE()
    {
        m_type = "MSE";
        fwd_shader = kernel::shaders::d_MSE_spv;
        fwd_codeSize = sizeof(kernel::shaders::d_MSE_spv);
    }

    void MSE::operator()(const std::shared_ptr<tensor>& y_true, const std::shared_ptr<tensor>& y_pred)
    {
        return hook(y_true, y_pred);
    }
}