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

			std::shared_ptr<tensor>celu::forward(std::shared_ptr<tensor>x)
			{
				std::shared_ptr<tensor> alpha = std::make_shared<tensor>(tensor(1.0, x->getShape(), Format::kFormatFp32));
				return layer_construct_forward<activation_param>(shaders::celu_spv, sizeof(shaders::celu_spv), x, alpha, m_param);
			}

			void celu::backward()
			{
				layer_construct_backward<activation_param>(shaders::d_celu_spv, sizeof(shaders::d_celu_spv), m_param);
			}

			elu::elu(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "elu";
			}

			std::shared_ptr<tensor>elu::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::elu_spv, sizeof(shaders::elu_spv), x, m_param);
			}

			void elu::backward()
			{
				layer_construct_backward<activation_param>(shaders::d_elu_spv, sizeof(shaders::d_elu_spv), m_param);
			}

			hardshrink::hardshrink(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "hardshrink";
			}

			std::shared_ptr<tensor>hardshrink::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::hardshrink_spv, sizeof(shaders::hardshrink_spv), x, m_param);
			}

			void hardshrink::backward()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			hardtanh::hardtanh(float min_val, float max_val, bool in_place) :
				unary_operator(0, in_place), m_param({ 0, min_val, max_val })
			{
				m_type = "hardtanh";
			}

			std::shared_ptr<tensor>hardtanh::forward(std::shared_ptr<tensor>x)
			{
				m_param.total = x->count();
				return layer_construct_forward<two_param>(shaders::hardshrink_spv, sizeof(shaders::hardshrink_spv), x, m_param);
			}

			void hardtanh::backward()
			{
				layer_construct_backward<two_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			leakyrelu::leakyrelu(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "leakyrelu";
			}

			std::shared_ptr<tensor>leakyrelu::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::leakyrelu_spv, sizeof(shaders::leakyrelu_spv), x, m_param);
			}

			void leakyrelu::backward()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			logsigmoid::logsigmoid(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "logsigmoid";
			}

			std::shared_ptr<tensor>logsigmoid::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::logsigmoid_spv, sizeof(shaders::logsigmoid_spv), x, m_param);
			}

			void logsigmoid::backward()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			prelu::prelu(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "prelu";
			}

			std::shared_ptr<tensor>prelu::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::prelu_spv, sizeof(shaders::prelu_spv), x, m_param);
			}

			void prelu::backward()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			relu::relu(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "relu";
			}

			std::shared_ptr<tensor>relu::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::relu_spv, sizeof(shaders::relu_spv), x, m_param);
			}

			void relu::backward()
			{
				layer_construct_backward<activation_param>(shaders::d_relu_spv, sizeof(shaders::d_relu_spv), m_param);
			}

			relu6::relu6(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "relu6";
			}

			std::shared_ptr<tensor>relu6::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::relu6_spv, sizeof(shaders::relu6_spv), x, m_param);
			}

			void relu6::backward()
			{
				layer_construct_backward<activation_param>(shaders::d_relu6_spv, sizeof(shaders::d_relu6_spv), m_param);
			}

			selu::selu(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "selu";
			}

			std::shared_ptr<tensor>selu::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::selu_spv, sizeof(shaders::selu_spv), x, m_param);
			}

			void selu::backward()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			sigmoid::sigmoid(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "sigmoid";
			}

			std::shared_ptr<tensor>sigmoid::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::sigmoid_spv, sizeof(shaders::sigmoid_spv), x, m_param);
			}

			void sigmoid::backward()
			{
				layer_construct_backward <activation_param>(shaders::d_sigmoid_spv, sizeof(shaders::d_sigmoid_spv), m_param);
			}

			softplus::softplus(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "softplus";
			}

			std::shared_ptr<tensor>softplus::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::softplus_spv, sizeof(shaders::softplus_spv), x, m_param);
			}

			void softplus::backward()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			softshrink::softshrink(float alpha, bool in_place) : unary_operator(alpha, in_place)
			{
				m_type = "softshrink";
			}

			std::shared_ptr<tensor>softshrink::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::softshrink_spv, sizeof(shaders::softshrink_spv), x, m_param);
			}

			void softshrink::backward()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			softsign::softsign(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "softsign";
			}

			std::shared_ptr<tensor>softsign::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::softsign_spv, sizeof(shaders::softsign_spv), x, m_param);
			}

			void softsign::backward()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}

			tanhshrink::tanhshrink(bool in_place) : unary_operator(0, in_place)
			{
				m_type = "tanhshrink";
			}

			std::shared_ptr<tensor>tanhshrink::forward(std::shared_ptr<tensor>x)
			{
				return layer_construct_forward<activation_param>(shaders::tanhshrink_spv, sizeof(shaders::tanhshrink_spv), x, m_param);
			}

			void tanhshrink::backward()
			{
				layer_construct_backward<activation_param>(shaders::unary_operator_spv, sizeof(shaders::unary_operator_spv), m_param);
			}
		}
	}
}