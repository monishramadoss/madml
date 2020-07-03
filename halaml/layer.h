#ifndef LAYER_H
#define LAYER_H
#include "madml.h"

#include <string>
#include <vector>

namespace kernel
{
	namespace layers {
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
	protected:
		void initVulkanThing(int buffer_num);
		void createDescriptorSetLayout(int buffer_num);
		void createDescriptorSet(int buffer_num);
		void createShaderModule(const uint32_t* spv, size_t sz, const std::string& source = std::string());
		void createPipeline(size_t push_constants_size = 0, VkSpecializationInfo* specialization_info = nullptr);
		void createCommandBuffer();
		void recordCommandBuffer(void* push_constants = nullptr, size_t push_constants_size = 0);
		void runCommandBuffer();
		virtual void computeGroupCount() = 0;

		VkPipeline m_pipeline;
		VkCommandBuffer m_cmd_buffer;
		VkDescriptorPool m_descriptor_pool;
		VkDescriptorSet m_descriptor_set;
		VkDevice m_device;
		VkDescriptorSetLayout m_descriptor_set_layout;
		VkPipelineLayout m_pipeline_layout;
		VkShaderModule m_module;
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
			void backward();
			virtual void update_weight() = 0;
			void execute();
			void super_run();
			std::vector<int> m_input;
			std::vector<int> m_output;
			std::vector<int> m_weights;
			std::vector<int> m_bias;

		protected:
			std::vector<layer*> layers;
			static std::vector<Module*>& get_module();
			static std::vector<tensor*>& get_tensors();
			static void add_tensor(tensor* T);
			static void add_module(Module* M);
			inline void set_io(Module* m);

			int batch_size = 0;
			float lr = 0.0001f;

			friend class kernel::tensor;
		};
	}
}

#endif
