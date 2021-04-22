#include "../engine/common.h"
#include "../engine/utils.h"
#include "optimizer.h"

adam::adam(std::vector<tensor>& params, float lr, float beta_a, float beta_b, float eps, float weight_decay, bool amsgrad)
{
    m_param.lr = lr;
    m_param.beta_a = beta_a;
    m_param.beta_b = beta_b;
    m_param.eps = eps;
    m_param.weight_decay = weight_decay;
    m_param.amsgrad = amsgrad;
}