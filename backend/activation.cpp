#include "common.h"
#include "utils.h"
#include "activation.h"
#define LOCAL_SZ_X 1024
#define MAX_COMPUTE_WORK_GROUP_COUNT 65535

namespace layers
{
	namespace activation
	{
		celu::celu(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "celu";
			m_param.alpha = alpha;
			bck_shader = kernel::shaders::d_celu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
		}

		std::shared_ptr<tensor>& celu::operator()(const std::shared_ptr<tensor>& x)
		{
			alpha = std::make_shared<tensor>(tensor(1.0, x->getShape(), Format::kFormatFp32));
			return layer_construct_forward(kernel::shaders::celu_spv, sizeof(kernel::shaders::celu_spv), x, alpha);
		}

		elu::elu(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "elu";
			m_param.alpha = alpha;

			bck_shader = kernel::shaders::d_elu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_elu_spv);
		}

		std::shared_ptr<tensor>& elu::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::elu_spv, sizeof(kernel::shaders::elu_spv), x);
		}

		hardshrink::hardshrink(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "hardshrink";
			m_param.alpha = alpha;
			bck_shader = kernel::shaders::d_celu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
		}

		std::shared_ptr<tensor>& hardshrink::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::hardshrink_spv, sizeof(kernel::shaders::hardshrink_spv), x);
		}

		hardtanh::hardtanh(float min_val, float max_val, bool in_place) : Base_Layer<two_param>(2, in_place)
		{
			m_type = "hardtanh";
			bck_shader = kernel::shaders::d_celu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
			m_param = { 0, min_val, max_val };
		}

		std::shared_ptr<tensor>& hardtanh::operator()(const std::shared_ptr<tensor>& x)
		{
			m_param.total = x->count();
			return layer_construct_forward(kernel::shaders::hardshrink_spv, sizeof(kernel::shaders::hardshrink_spv), x);
		}

		leakyrelu::leakyrelu(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "leakyrelu";
			m_param.alpha = alpha;
			bck_shader = kernel::shaders::d_celu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
		}

		std::shared_ptr<tensor>& leakyrelu::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::leakyrelu_spv, sizeof(kernel::shaders::leakyrelu_spv), x);
		}

		logsigmoid::logsigmoid(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "logsigmoid";
			m_param.alpha = alpha;
			bck_shader = kernel::shaders::d_celu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
		}

		std::shared_ptr<tensor>& logsigmoid::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::logsigmoid_spv, sizeof(kernel::shaders::logsigmoid_spv), x);
		}

		prelu::prelu(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "prelu";
			m_param.alpha = alpha;
			bck_shader = kernel::shaders::d_celu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
		}

		std::shared_ptr<tensor>& prelu::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::prelu_spv, sizeof(kernel::shaders::prelu_spv), x);
		}

		relu::relu(bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "relu";
			m_param.alpha = 0;
			bck_shader = kernel::shaders::d_relu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_relu_spv);
		}

		std::shared_ptr<tensor>& relu::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::relu_spv, sizeof(kernel::shaders::relu_spv), x);
		}

		relu6::relu6(bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "relu6";
			m_param.alpha = 0;
			bck_shader = kernel::shaders::d_relu6_spv;
			bck_codeSize = sizeof(kernel::shaders::d_relu6_spv);
		}

		std::shared_ptr<tensor>& relu6::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::relu6_spv, sizeof(kernel::shaders::relu6_spv), x);
		}

		selu::selu(bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "selu";
			m_param.alpha = 0;
			bck_shader = kernel::shaders::d_celu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
		}

		std::shared_ptr<tensor>& selu::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::selu_spv, sizeof(kernel::shaders::selu_spv), x);
		}

		sigmoid::sigmoid(bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "sigmoid";
			m_param.alpha = 0;
			bck_shader = kernel::shaders::d_sigmoid_spv;
			bck_codeSize = sizeof(kernel::shaders::d_sigmoid_spv);
		}

		std::shared_ptr<tensor>& sigmoid::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::sigmoid_spv, sizeof(kernel::shaders::sigmoid_spv), x);
		}

		softplus::softplus(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "softplus";
			m_param.alpha = alpha;
			bck_shader = kernel::shaders::d_celu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
		}

		std::shared_ptr<tensor>& softplus::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::softplus_spv, sizeof(kernel::shaders::softplus_spv), x);
		}

		softshrink::softshrink(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "softshrink";
			m_param.alpha = alpha;
			bck_shader = kernel::shaders::d_celu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
		}

		std::shared_ptr<tensor>& softshrink::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::softshrink_spv, sizeof(kernel::shaders::softshrink_spv), x);
		}

		softsign::softsign(bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "softsign";
			m_param.alpha = 0;
			bck_shader = kernel::shaders::d_celu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
		}

		std::shared_ptr<tensor>& softsign::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::softsign_spv, sizeof(kernel::shaders::softsign_spv), x);
		}

		tanhshrink::tanhshrink(bool in_place) : Base_Layer<activation_param>(in_place)
		{
			m_type = "tanhshrink";
			m_param.alpha = 0;
			bck_shader = kernel::shaders::d_celu_spv;
			bck_codeSize = sizeof(kernel::shaders::d_celu_spv);
		}

		std::shared_ptr<tensor>& tanhshrink::operator()(const std::shared_ptr<tensor>& x)
		{
			return layer_construct_forward(kernel::shaders::tanhshrink_spv, sizeof(kernel::shaders::tanhshrink_spv), x);
		}
	}
}