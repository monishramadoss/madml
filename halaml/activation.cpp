#include "common.h"
#include "utils.h"
#include "activation.h"
#define LOCAL_SZ_X 1024
#define MAX_COMPUTE_WORK_GROUP_COUNT 65535

namespace kernel
{
	namespace layers
	{
		namespace activation
		{
			unary_operator::unary_operator(float alpha, bool in_place) : Base_Layer(2, -1, in_place), m_param({ 0, alpha })
			{
			}

			void unary_operator::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.total, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
					m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
				m_group_y = 1;
				m_group_z = 1;
			}

			celu::celu(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "celu";
			}

			tensor* celu::forward(tensor* x)
			{
				tensor* alpha = new tensor(1.0, x->getShape());
				return layer_construct_forward<activation_param>(shaders::celu_spv, sizeof(shaders::celu_spv), x, alpha, m_param);
			}

			void celu::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::d_celu_spv, sizeof(shaders::d_celu_spv), m_param);
			}

			elu::elu(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "elu";
			}

			tensor* elu::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::elu_spv, sizeof(shaders::elu_spv), x, m_param);
			}

			void elu::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::d_elu_spv, sizeof(shaders::d_elu_spv), m_param);
			}

			hardshrink::hardshrink(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "hardshrink";
			}

			tensor* hardshrink::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::hardshrink_spv, sizeof(shaders::hardshrink_spv), x, m_param);
			}

			void hardshrink::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			hardtanh::hardtanh(float min_val, float max_val, bool in_place) :
				unary_operator(0, in_place), m_param({ 0, min_val, max_val })
			{
				m_type = "hardtanh";
			}

			tensor* hardtanh::forward(tensor* x)
			{
				m_param.total = x->count();
				return layer_construct_forward<two_param>(shaders::hardshrink_spv, sizeof(shaders::hardshrink_spv), x, m_param);
			}

			void hardtanh::back_propagate()
			{
				layer_construct_backward<two_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			leakyrelu::leakyrelu(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "leakyrelu";
			}

			tensor* leakyrelu::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::leakyrelu_spv, sizeof(shaders::leakyrelu_spv), x, m_param);
			}

			void leakyrelu::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			logsigmoid::logsigmoid(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "logsigmoid";
			}

			tensor* logsigmoid::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::logsigmoid_spv, sizeof(shaders::logsigmoid_spv), x, m_param);
			}

			void logsigmoid::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			prelu::prelu(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "prelu";
			}

			tensor* prelu::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::prelu_spv, sizeof(shaders::prelu_spv), x, m_param);
			}

			void prelu::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			relu::relu(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "relu";
			}

			tensor* relu::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::relu_spv, sizeof(shaders::relu_spv), x, m_param);
			}

			void relu::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::d_relu_spv, sizeof(shaders::d_relu_spv), m_param);
			}

			relu6::relu6(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "relu6";
			}

			tensor* relu6::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::relu6_spv, sizeof(shaders::relu6_spv), x, m_param);
			}

			void relu6::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::d_relu6_spv, sizeof(shaders::d_relu6_spv), m_param);
			}

			selu::selu(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "selu";
			}

			tensor* selu::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::selu_spv, sizeof(shaders::selu_spv), x, m_param);
			}

			void selu::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			sigmoid::sigmoid(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "sigmoid";
			}

			tensor* sigmoid::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::sigmoid_spv, sizeof(shaders::sigmoid_spv), x, m_param);
			}

			void sigmoid::back_propagate()
			{
				layer_construct_backward <activation_param>(shaders::d_sigmoid_spv, sizeof(shaders::d_sigmoid_spv), m_param);
			}

			softplus::softplus(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "softplus";
			}

			tensor* softplus::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::softplus_spv, sizeof(shaders::softplus_spv), x, m_param);
			}

			void softplus::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			softshrink::softshrink(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "softshrink";
			}

			tensor* softshrink::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::softshrink_spv, sizeof(shaders::softshrink_spv), x, m_param);
			}

			void softshrink::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			softsign::softsign(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "softsign";
			}

			tensor* softsign::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::softsign_spv, sizeof(shaders::softsign_spv), x, m_param);
			}

			void softsign::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			tanhshrink::tanhshrink(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "tanhshrink";
			}

			tensor* tanhshrink::forward(tensor* x)
			{
				return layer_construct_forward<activation_param>(shaders::tanhshrink_spv, sizeof(shaders::tanhshrink_spv), x, m_param);
			}

			void tanhshrink::back_propagate()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}
		}
	}
}