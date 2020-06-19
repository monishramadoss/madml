#include "common.h"
#include "utils.h"
#include "nn_layers.h"
#include <numeric>
#define maxComputeWorkGroupCount 1024
#define LOCAL_SZ_X 1024

namespace kernel {
	namespace layers {
		struct gradientParam {
			int total;
			float lr;
		};

		std::vector<Module*>* gradient::get_module() {
			return &Module::module_list;
		}

		gradient::gradient(float lr) : m_lr(lr) {
			initVulkanThing(3);
			m_type = "gradient";
		}

		void gradient::reshapeOutTensor(tensor* x, tensor* z) {
			Shape shape = x->getShape();
			*z = z->reshape(nullptr, shape);
		}

		bool gradient::forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) {
			return forward(ins[0], ins[1], outs[0]);
		}

		bool gradient::forward(tensor* x, tensor* y, tensor* z) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = x->count();
				computeGroupCount();
				createShaderModule(shaders::gradient_spv, sizeof(shaders::gradient_spv));
				createPipeline(sizeof(gradientParam));
			}
						
			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);			
			bindTensor(m_device, z, 2, m_descriptor_set);

			gradientParam param = { m_total, m_lr };
			recordCommandBuffer((void*)&param, sizeof(gradientParam));
			return true;
		}

		bool gradient::computeGroupCount() {
			m_group_x = (int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}


namespace kernel {
	namespace layers {
		namespace nn {
			
			std::vector<Module*>* dense::get_module() {
				return &Module::module_list;
			}

			dense::dense(int size, bool bias, bool debug) : size(size), bias(bias) {
				
				mul_op = new matmul();
				forward_layers.push_back(mul_op);

				grad_w = new gradient(1.0);
				gradient_layers.push_back(grad_w);

				backward_mul_op_dw = new matmul();
				backward_mul_op_dx = new matmul();
				backward_layers.push_back(backward_mul_op_dx);
				backward_layers.push_back(backward_mul_op_dw);

				if (bias) {
					add_op = new operators(0);
					forward_layers.push_back(add_op);
					grad_b = new gradient(1.0);
					gradient_layers.push_back(grad_b);
				}
				
				m_weight = nullptr;
				m_bias = nullptr;
				m_output = nullptr;

				input_tensor = nullptr;
				weight_tensor = nullptr;
				bias_tensor = nullptr;
				output_tensor = nullptr;
				m_debug = debug;

			}

			bool dense::operator()(tensor* x, tensor* y) {
				std::vector<layer*> layers;
				std::vector<tensor*> tensors;
				tensors.push_back(x);

				auto in_shape = x->getShape();
				
				if (m_weight == nullptr) {
					auto weight_shape = std::vector<int>{ in_shape[1], size };
					m_weight = fill_memory_shape<float>(weight_shape, 1);
					weight_tensor = new tensor(m_weight, weight_shape, kFormatFp32);
					tensors.push_back(weight_tensor);
				}

				if (m_bias == nullptr && bias) {
					auto bias_shape = std::vector<int>{ in_shape[0], size };
					m_bias = fill_memory_shape<float>(bias_shape, 1);
					bias_tensor = new tensor(m_bias, bias_shape, kFormatFp32);
					tensors.push_back(bias_tensor);
				}
				if (m_output == nullptr) {
					auto output_shape = std::vector<int>{ in_shape[0], size };
					m_output = fill_memory_shape<float>(output_shape, 0);
					output_tensor = new tensor(m_output, output_shape, kFormatFp32);
					tensors.push_back(output_tensor);
					*y = *output_tensor;
				}

				mul_op->forward(x, weight_tensor, y);
				if (bias) {
					add_op->forward(output_tensor, bias_tensor, output_tensor);
				}
				
				if (m_debug) {
					mul_op->run();
					if (bias)
						add_op->run();
					for(int i = 0; i < y->count(); ++i)
						std::cout << ((float*)y->toHost())[i] << " ";
					std::cout << std::endl;				
				}	

				add_layer(this);
				return true;
			}
					
			void dense::update_weight() {
			    
				grad_w->forward(weight_tensor, d_weight, weight_tensor);
				if (bias)
					grad_b->forward(bias_tensor, d_bias, bias_tensor);

				std::cout << "Backward Dense Layer" << std::endl;
			}

			void dense::backward(tensor* d_output, tensor* d_input) {
				/* 
					d_C, A, B, d_A, d_B

					d_L/d_X = d_L/d_Y * W.T
					d_L/d_W = X.T * d_L/d_Y 

					d_A += d_C * B.T
					d_B += A.T * d_C

					deltaoutput, inputArray, nnWeights, deltainput, deltaweights 


					acc_grad = previous grad
					
					weight_gradient = input * acc_grad * loss; if no act
					if act then act' * acc_grad * loss;

				*/

				d_bias = d_output;
				backward_mul_op_dw->forward(d_output, weight_tensor, d_weight); //weight transposed
				backward_mul_op_dx->forward(input_tensor, d_output, d_input);   //input transposed
			}

		}
	}
}