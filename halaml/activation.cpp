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
			celu::celu(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "celu";
				m_param.alpha = alpha;
				bck_shader = shaders::d_celu_spv;
				bck_codeSize = sizeof(shaders::d_celu_spv);
			}

			std::shared_ptr<tensor>& celu::hook(const std::shared_ptr<tensor>& x)
			{
				alpha = std::make_shared<tensor>(tensor(1.0, x->getShape(), Format::kFormatFp32));
				return layer_construct_forward(shaders::celu_spv, sizeof(shaders::celu_spv), x, alpha);
			}

			elu::elu(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "elu";
				m_param.alpha = alpha;

				bck_shader = shaders::d_elu_spv;
				bck_codeSize = sizeof(shaders::d_elu_spv);
			}

			std::shared_ptr<tensor>& elu::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::elu_spv, sizeof(shaders::elu_spv), x);
			}

			hardshrink::hardshrink(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "hardshrink";
				m_param.alpha = alpha;
				bck_shader = shaders::d_celu_spv;
				bck_codeSize = sizeof(shaders::d_celu_spv);
			}

			std::shared_ptr<tensor>& hardshrink::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::hardshrink_spv, sizeof(shaders::hardshrink_spv), x);
			}

			hardtanh::hardtanh(float min_val, float max_val, bool in_place) : Base_Layer<two_param>(2, in_place)
			{
				m_type = "hardtanh";
				bck_shader = shaders::d_celu_spv;
				bck_codeSize = sizeof(shaders::d_celu_spv);
				m_param = { 0, min_val, max_val };
			}

			std::shared_ptr<tensor>& hardtanh::hook(const std::shared_ptr<tensor>& x)
			{
				m_param.total = x->count();
				return layer_construct_forward(shaders::hardshrink_spv, sizeof(shaders::hardshrink_spv), x);
			}

			leakyrelu::leakyrelu(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "leakyrelu";
				m_param.alpha = alpha;
				bck_shader = shaders::d_celu_spv;
				bck_codeSize = sizeof(shaders::d_celu_spv);
			}

			std::shared_ptr<tensor>& leakyrelu::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::leakyrelu_spv, sizeof(shaders::leakyrelu_spv), x);
			}

			logsigmoid::logsigmoid(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "logsigmoid";
				m_param.alpha = alpha;
				bck_shader = shaders::d_celu_spv;
				bck_codeSize = sizeof(shaders::d_celu_spv);
			}

			std::shared_ptr<tensor>& logsigmoid::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::logsigmoid_spv, sizeof(shaders::logsigmoid_spv), x);
			}

			prelu::prelu(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "prelu";
				m_param.alpha = alpha;
				bck_shader = shaders::d_celu_spv;
				bck_codeSize = sizeof(shaders::d_celu_spv);
			}

			std::shared_ptr<tensor>& prelu::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::prelu_spv, sizeof(shaders::prelu_spv), x);
			}

			relu::relu(bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "relu";
				m_param.alpha = 0;
				bck_shader = shaders::d_relu_spv;
				bck_codeSize = sizeof(shaders::d_relu_spv);
			}

			std::shared_ptr<tensor>& relu::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::relu_spv, sizeof(shaders::relu_spv), x);
			}

			relu6::relu6(bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "relu6";
				m_param.alpha = 0;
				bck_shader = shaders::d_relu6_spv;
				bck_codeSize = sizeof(shaders::d_relu6_spv);
			}

			std::shared_ptr<tensor>& relu6::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::relu6_spv, sizeof(shaders::relu6_spv), x);
			}

			selu::selu(bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "selu";
				m_param.alpha = 0;
				bck_shader = shaders::d_celu_spv;
				bck_codeSize = sizeof(shaders::d_celu_spv);
			}

			std::shared_ptr<tensor>& selu::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::selu_spv, sizeof(shaders::selu_spv), x);
			}

			sigmoid::sigmoid(bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "sigmoid";
				m_param.alpha = 0;
				bck_shader = shaders::d_sigmoid_spv;
				bck_codeSize = sizeof(shaders::d_sigmoid_spv);
			}

			std::shared_ptr<tensor>& sigmoid::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::sigmoid_spv, sizeof(shaders::sigmoid_spv), x);
			}

			softplus::softplus(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "softplus";
				m_param.alpha = alpha;
				bck_shader = shaders::d_celu_spv;
				bck_codeSize = sizeof(shaders::d_celu_spv);
			}

			std::shared_ptr<tensor>& softplus::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::softplus_spv, sizeof(shaders::softplus_spv), x);
			}

			softshrink::softshrink(float alpha, bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "softshrink";
				m_param.alpha = alpha;
				bck_shader = shaders::d_celu_spv;
				bck_codeSize = sizeof(shaders::d_celu_spv);
			}

			std::shared_ptr<tensor>& softshrink::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::softshrink_spv, sizeof(shaders::softshrink_spv), x);
			}

			softsign::softsign(bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "softsign";
				m_param.alpha = 0;
				bck_shader = shaders::d_celu_spv;
				bck_codeSize = sizeof(shaders::d_celu_spv);
			}

			std::shared_ptr<tensor>& softsign::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::softsign_spv, sizeof(shaders::softsign_spv), x);
			}

			tanhshrink::tanhshrink(bool in_place) : Base_Layer<activation_param>(in_place)
			{
				m_type = "tanhshrink";
				m_param.alpha = 0;
				bck_shader = shaders::d_celu_spv;
				bck_codeSize = sizeof(shaders::d_celu_spv);
			}

			std::shared_ptr<tensor>& tanhshrink::hook(const std::shared_ptr<tensor>& x)
			{
				return layer_construct_forward(shaders::tanhshrink_spv, sizeof(shaders::tanhshrink_spv), x);
			}
		}
	}
}