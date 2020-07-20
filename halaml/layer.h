#ifndef LAYER_H
#define LAYER_H
#include "madml.h"

#include <string>
#include <vector>

namespace kernel
{
	namespace layers
	{
		class Module;
	}

	class context;

	class layer
	{
	public:
		layer();
		virtual ~layer();
		//		virtual void forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) = 0;
		//		virtual void backward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) = 0;
		void initVulkanThing(int buffer_num_forward, int buffer_num_backward = -1);
		
		void createDescriptorSetLayoutForward(int buffer_num);
		void createDescriptorSetForward(int buffer_num);
		void createShaderModuleForward(const uint32_t* spv, size_t sz, const std::string& source = std::string());
		void createPipelineForward(size_t push_constants_size = 0, VkSpecializationInfo* specialization_info = nullptr);
		void createCommandBufferForward();
		void recordCommandBufferForward(void* push_constants = nullptr, size_t push_constants_size = 0) const;
		void runCommandBufferForward() const;

		void createDescriptorSetLayoutBackward(int buffer_num);
		void createDescriptorSetBackward(int buffer_num);
		void createShaderModuleBackward(const uint32_t* spv, size_t sz, const std::string& source = std::string());
		void createPipelineBackward(size_t push_constants_size = 0, VkSpecializationInfo* specialization_info = nullptr);
		void createCommandBufferBackward();
		void recordCommandBufferBackward(void* push_constants = nullptr, size_t push_constants_size = 0);
		void runCommandBufferBackward();
		
		virtual void computeGroupCount() = 0;
		VkDevice m_device;
		
		VkPipeline m_pipeline_forward;
		VkCommandBuffer m_cmd_buffer_forward;
		VkDescriptorPool m_descriptor_pool_forward;
		VkDescriptorSet m_descriptor_set_forward;
		VkDescriptorSetLayout m_descriptor_set_layout_forward;
		VkPipelineLayout m_pipeline_layout_forward;
		VkShaderModule m_module_forward;
				
		VkPipeline m_pipeline_backward;
		VkCommandBuffer m_cmd_buffer_backward;
		VkDescriptorPool m_descriptor_pool_backward;
		VkDescriptorSet m_descriptor_set_backward;
		VkDescriptorSetLayout m_descriptor_set_layout_backward;
		VkPipelineLayout m_pipeline_layout_backward;
		VkShaderModule m_module_backward;
				
		int m_group_x;
		int m_group_y;
		int m_group_z;
		std::string m_type;

		friend class layers::Module;
	};

	namespace layers
	{
		class Module
		{
		public:
			static void backward();
			virtual void update_weight() = 0;
			void execute();
			static void super_run();
			std::vector<int> inputs;
			std::vector<int> outputs;
			std::vector<int> weights;
			std::vector<int> biases;
			std::vector<int> temporaries;

		protected:
			std::vector<layer*> layers;
			static std::vector<Module*>& get_module();
			static std::vector<tensor*>& get_tensors();
			static std::vector<tensor*>& get_gradients();
			static tensor* get_grad(int id);

			static void zero_grad();

			static void add_tensor(tensor* T);
			static void add_module(Module* M);
			static void add_gradient(tensor* G);

			int batch_size = 0;
			float lr = 0.0001f;

			virtual void back_propagate() {};

			friend class tensor;
		};
	}
}

#endif
