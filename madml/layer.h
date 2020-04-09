#ifndef LAYER_H
#define LAYER_H
#include "madml.h"

#include <string>
#include <vector>

namespace kernel {
	class context;

	class layer
	{
	public:
		layer();
		virtual ~layer();
		virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs) = 0;
		bool run() {
			runCommandBuffer();
			return true;
		}
	protected:
		void initVulkanThing(int buffer_num);
		void createDescriptorSetLayout(int buffer_num);
		void createDescriptorSet(int buffer_num);
		void createShaderModule(const uint32_t* spv, size_t sz, const std::string& source = std::string());
		void createPipeline(size_t push_constants_size = 0, VkSpecializationInfo* specialization_info = 0);
		void createCommandBuffer();
		void recordCommandBuffer(void* push_constants = 0, size_t push_constants_size = 0);
		void runCommandBuffer();

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
	};
}


#endif