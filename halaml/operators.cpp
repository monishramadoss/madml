#include "common.h"
#include "utils.h"
#include "operators.h"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct tensorParam {
			size_t total;
		};

		operators::operators(size_t op_id) {
			layer::initVulkanThing(3);
			m_type = "operators";
			m_op = op_id;
		}

		void operators::reshapeOutTensor(tensor* x, tensor* z) {
			Shape shape = x->getShape();
			z = &(z->reshape(nullptr, shape));
		}

		bool operators::forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) {
			if(ins.size() == 2)
				return forward(ins[0], ins[1], outs[0]);
			else
				return forward(ins[0], outs[0]);
		}

		bool operators::forward(tensor* x, tensor* y) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = x->count();
				computeGroupCount();
				switch (m_op) {
					//arithmetic
				case 0:
					createShaderModule(shaders::add_spv, sizeof(shaders::add_spv));
					break;
				case 1:
					createShaderModule(shaders::sub_spv, sizeof(shaders::sub_spv));
					break;
				case 2:
					createShaderModule(shaders::mul_spv, sizeof(shaders::mul_spv));
					break;
				case 3:
					createShaderModule(shaders::add_spv, sizeof(shaders::div_spv));
					break;
				case 4:
					createShaderModule(shaders::mod_spv, sizeof(shaders::mod_spv));
					break;

					//logical
				case 5:
					createShaderModule(shaders::equal_spv, sizeof(shaders::equal_spv));
					break;
				case 6:
					createShaderModule(shaders::nequal_spv, sizeof(shaders::nequal_spv));
					break;
				case 7:
					createShaderModule(shaders::greater_than_spv, sizeof(shaders::greater_than_spv));
					break;
				case 8:
					createShaderModule(shaders::less_than_spv, sizeof(shaders::less_than_spv));
					break;
				case 9:
					createShaderModule(shaders::greater_eq_spv, sizeof(shaders::greater_eq_spv));
					break;
				case 10:
					createShaderModule(shaders::less_eq_spv, sizeof(shaders::less_eq_spv));
					break;
				case 11:
					createShaderModule(shaders::xor_spv, sizeof(shaders::xor_spv));
					break;
				case 12:
					createShaderModule(shaders::pow_spv, sizeof(shaders::pow_spv));
					break;

					//other ops
				case 13:
					createShaderModule(shaders::min_spv, sizeof(shaders::min_spv));
					break;
				case 14:
					createShaderModule(shaders::max_spv, sizeof(shaders::max_spv));
					break;

					//trig
				case 15:
					createShaderModule(shaders::asinh_spv, sizeof(shaders::asinh_spv));
					break;
				case 16:
					createShaderModule(shaders::asin_spv, sizeof(shaders::asin_spv));
					break;
				case 17:
					createShaderModule(shaders::acosh_spv, sizeof(shaders::acosh_spv));
					break;
				case 18:
					createShaderModule(shaders::acos_spv, sizeof(shaders::acos_spv));
					break;
				case 19:
					createShaderModule(shaders::atanh_spv, sizeof(shaders::atanh_spv));
					break;
				case 20:
					createShaderModule(shaders::atan_spv, sizeof(shaders::atan_spv));
					break;

				case 21:
					createShaderModule(shaders::sin_spv, sizeof(shaders::sin_spv));
					break;
				case 22:
					createShaderModule(shaders::sinh_spv, sizeof(shaders::sinh_spv));
					break;
				case 23:
					createShaderModule(shaders::cosh_spv, sizeof(shaders::cosh_spv));
					break;
				case 24:
					createShaderModule(shaders::cos_spv, sizeof(shaders::cos_spv));
					break;
				case 25:
					createShaderModule(shaders::tanh_spv, sizeof(shaders::tanh_spv));
					break;
				case 26:
					createShaderModule(shaders::tan_spv, sizeof(shaders::tan_spv));
					break;

					//math op
				case 27:
					createShaderModule(shaders::abs_spv, sizeof(shaders::abs_spv));
					break;
				case 28:
					createShaderModule(shaders::sqrt_spv, sizeof(shaders::sqrt_spv));
					break;
				case 29:
					createShaderModule(shaders::exp_spv, sizeof(shaders::exp_spv));
					break;
				case 30:
					createShaderModule(shaders::log_spv, sizeof(shaders::log_spv));
					break;
				case 31:
					createShaderModule(shaders::clip_spv, sizeof(shaders::clip_spv));
					break;
				case 32:
					createShaderModule(shaders::round_spv, sizeof(shaders::round_spv));
					break;
				case 33:
					createShaderModule(shaders::floor_spv, sizeof(shaders::floor_spv));
					break;
				case 34:
					createShaderModule(shaders::ceil_spv, sizeof(shaders::ceil_spv));
					break;

				}
				createPipeline(sizeof(tensorParam));
			}
						
			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);
			if (m_op < 15)
				bindTensor(m_device, x, 2, m_descriptor_set);
			tensorParam param = { m_total };
			recordCommandBuffer((void*)&param, sizeof(tensorParam));
			return true;
		}

		bool operators::forward(tensor* x, tensor* y, tensor* z) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = x->count();
				computeGroupCount();
				switch (m_op) { //34
					//arithmetic
				case 0:
					createShaderModule(shaders::add_spv, sizeof(shaders::add_spv));
					break;
				case 1:
					createShaderModule(shaders::sub_spv, sizeof(shaders::sub_spv));
					break;
				case 2:
					createShaderModule(shaders::mul_spv, sizeof(shaders::mul_spv));
					break;
				case 3:
					createShaderModule(shaders::div_spv, sizeof(shaders::div_spv));
					break;
				case 4:
					createShaderModule(shaders::mod_spv, sizeof(shaders::mod_spv));
					break;

					//logical
				case 5:
					createShaderModule(shaders::equal_spv, sizeof(shaders::equal_spv));
					break;
				case 6:
					createShaderModule(shaders::nequal_spv, sizeof(shaders::nequal_spv));
					break;
				case 7:
					createShaderModule(shaders::greater_than_spv, sizeof(shaders::greater_than_spv));
					break;
				case 8:
					createShaderModule(shaders::less_than_spv, sizeof(shaders::less_than_spv));
					break;
				case 9:
					createShaderModule(shaders::greater_eq_spv, sizeof(shaders::greater_eq_spv));
					break;
				case 10:
					createShaderModule(shaders::less_eq_spv, sizeof(shaders::less_eq_spv));
					break;
				case 11:
					createShaderModule(shaders::xor_spv, sizeof(shaders::xor_spv));
					break;

				//other ops

				case 12:
					createShaderModule(shaders::pow_spv, sizeof(shaders::pow_spv));
					break;			
				case 13:
					createShaderModule(shaders::min_spv, sizeof(shaders::min_spv));
					break;
				case 14:
					createShaderModule(shaders::max_spv, sizeof(shaders::max_spv));
					break;
				}
				createPipeline(sizeof(tensorParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);
			bindTensor(m_device, z, 2, m_descriptor_set);
			tensorParam param = { m_total };
			recordCommandBuffer((void*)&param, sizeof(tensorParam));
			return true;
		}		

		bool operators::computeGroupCount() {
			m_group_x = (int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}		

		bool operators::operator()(tensor* x, tensor* y){
			if (m_op > 15) {
				if (out == nullptr && y->count() == 0) {
					char* tmp = fill_memory_shape<float>(x->getShape(), 0);
					out = new tensor(tmp, x->getShape(), x->getFormat());
					*y = *out;
				}
				else if (out == nullptr) {
					out = y;
				}
			}
				
			forward(x, y);
			return false;
		}

		void operators::backward() {

		}
	}
}