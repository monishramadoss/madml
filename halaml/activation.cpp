#include "common.h"
#include "utils.h"
#include "activation.h"
#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {

		namespace activation {

			unary_operator::unary_operator(float alpha, bool in_place, bool as_module) : m_inplace(in_place), as_module(as_module), m_param({ 0, m_alpha })
			{
				initVulkanThing(2);
			}

			void unary_operator::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.total, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > maxComputeWorkGroupCount)
					m_group_x = maxComputeWorkGroupCount;
				m_group_y = 1;
				m_group_z = 1;
			}

			tensor* unary_operator::layer_construct(const uint32_t* shader, size_t codeSize, tensor* x)
			{
				m_input.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape());
				m_output.push_back(x->getId());

				if (m_pipeline == nullptr)
				{
					m_param.total =  x->count() ;
					computeGroupCount();
					createShaderModule(shader, codeSize);
					createPipeline(sizeof(operator_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, y, 1, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(operator_param));
				layers.push_back(this);
				if (as_module)
					add_module(this);
				return y;
			}

			void unary_operator::update_weight()
			{
			}

			celu::celu(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module) {
				m_type = "celu";
			}

			tensor* celu::forward(tensor* x) {
				return layer_construct(shaders::celu_spv, sizeof(shaders::celu_spv), x);
			}
			
			elu::elu(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module) {
				m_type = "elu";
			}

			tensor* elu::forward(tensor* x) {
				return layer_construct(shaders::elu_spv, sizeof(shaders::elu_spv), x);
			}

			hardshrink::hardshrink(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module) {
				m_type = "hardshrink";
			}

			tensor* hardshrink::forward(tensor* x) {
				return layer_construct(shaders::hardshrink_spv, sizeof(shaders::hardshrink_spv), x);
			}

			hardtanh::hardtanh(float min_val, float max_val, bool in_place, bool as_module) : m_param({ 0, min_val, max_val }), unary_operator(0, in_place, as_module) {
				m_type = "hardtanh";
			}

			tensor* hardtanh::forward(tensor* x) {
				m_input.push_back(x->getId());
				tensor* y;
				if (m_inplace)
					y = x;
				else
					y = new tensor(0.0, x->getShape());
				m_output.push_back(x->getId());

				if (m_pipeline == nullptr)
				{
					m_param.total = x->count();
					computeGroupCount();
					createShaderModule(shaders::hardtanh_spv, sizeof(shaders::hardtanh_spv));
					createPipeline(sizeof(two_param));
				}

				bindTensor(m_device, x, 0, m_descriptor_set);
				bindTensor(m_device, y, 1, m_descriptor_set);

				recordCommandBuffer(static_cast<void*>(&m_param), sizeof(operator_param));
				layers.push_back(this);
				if (as_module)
					add_module(this);
				return y;
			}


			leakyrelu::leakyrelu(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module) {
				m_type = "leakyrelu";
			}

			tensor* leakyrelu::forward(tensor* x) {
				return layer_construct(shaders::leakyrelu_spv, sizeof(shaders::leakyrelu_spv), x);
			}

			logsigmoid::logsigmoid(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module) {
				m_type = "logsigmoid";
			}

			tensor* logsigmoid::forward(tensor* x) {
				return layer_construct(shaders::logsigmoid_spv, sizeof(shaders::logsigmoid_spv), x);
			}

			prelu::prelu(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module) {
				m_type = "prelu";
			}

			tensor* prelu::forward(tensor* x) {
				return layer_construct(shaders::prelu_spv, sizeof(shaders::prelu_spv), x);
			}

			relu::relu(bool in_place, bool as_module) : unary_operator(0, in_place, as_module) {
				m_type = "relu";
			}

			tensor* relu::forward(tensor* x) {
				return layer_construct(shaders::relu_spv, sizeof(shaders::relu_spv), x);
			}

			relu6::relu6(bool in_place, bool as_module) : unary_operator(0, in_place, as_module) {
				m_type = "relu6";
			}

			tensor* relu6::forward(tensor* x) {
				return layer_construct(shaders::relu6_spv, sizeof(shaders::relu6_spv), x);
			}

			selu::selu(bool in_place, bool as_module) : unary_operator(0, in_place, as_module) {
				m_type = "selu";
			}

			tensor* selu::forward(tensor* x) {
				return layer_construct(shaders::selu_spv, sizeof(shaders::selu_spv), x);
			}

			sigmoid::sigmoid(bool in_place, bool as_module) : unary_operator(0, in_place, as_module) {
				m_type = "sigmoid";
			}

			tensor* sigmoid::forward(tensor* x) {
				return layer_construct(shaders::sigmoid_spv, sizeof(shaders::sigmoid_spv), x);
			}

			softplus::softplus(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module) {
				m_type = "softplus";
			}

			tensor* softplus::forward(tensor* x) {
				return layer_construct(shaders::softplus_spv, sizeof(shaders::softplus_spv), x);
			}

			softshrink::softshrink(float alpha, bool in_place, bool as_module) : unary_operator(alpha, in_place, as_module) {
				m_type = "softshrink";
			}

			tensor* softshrink::forward(tensor* x) {
				return layer_construct(shaders::softshrink_spv, sizeof(shaders::softshrink_spv), x);
			}

			softsign::softsign( bool in_place, bool as_module) : unary_operator(0, in_place, as_module) {
				m_type = "softsign";
			}

			tensor* softsign::forward(tensor* x) {
				return layer_construct(shaders::softsign_spv, sizeof(shaders::softsign_spv), x);
			}

			tanhshrink::tanhshrink(bool in_place, bool as_module) : unary_operator(0, in_place, as_module) {
				m_type = "tanhshrink";
			}

			tensor* tanhshrink::forward(tensor* x) {
				return layer_construct(shaders::tanhshrink_spv, sizeof(shaders::tanhshrink_spv), x);
			}

			

		}
	}
}