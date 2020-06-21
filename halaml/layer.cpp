#include "common.h"
#include "utils.h"
#include "layer.h"

namespace kernel
{
	namespace layers
	{
		std::vector<Module*> Module::module_list;
	}

	layer::layer()
	{
		createContext();
		m_device = kDevice;
		m_pipeline = nullptr;
		m_cmd_buffer = nullptr;
		m_descriptor_pool = nullptr;
		m_descriptor_set = nullptr;
		m_descriptor_set_layout = nullptr;
		m_pipeline_layout = nullptr;
		m_module = nullptr;
		m_group_x = 1;
		m_group_y = 1;
		m_group_z = 1;
	}

	layer::~layer()
	{
		vkDestroyShaderModule(m_device, m_module, nullptr);
		vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
		vkDestroyPipeline(m_device, m_pipeline, nullptr);
		vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
	}

	void layer::initVulkanThing(int buffer_num)
	{
		createDescriptorSetLayout(buffer_num);
		createDescriptorSet(buffer_num);
		createCommandBuffer();
	}

	void layer::createDescriptorSetLayout(int buffer_num)
	{
		if (buffer_num <= 0)
			return;
		std::vector<VkDescriptorSetLayoutBinding> bindings(buffer_num);
		for (int i = 0; i < buffer_num; i++)
		{
			bindings[i].binding = i;
			bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		VkDescriptorSetLayoutCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		info.bindingCount = buffer_num;
		info.pBindings = &bindings[0];
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device, &info, 0, &m_descriptor_set_layout));
	}

	void layer::createDescriptorSet(int buffer_num)
	{
		VkDescriptorPoolSize pool_size = {};
		pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		pool_size.descriptorCount = buffer_num;

		VkDescriptorPoolCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		info.maxSets = 1;
		info.poolSizeCount = 1;
		info.pPoolSizes = &pool_size;
		VK_CHECK_RESULT(vkCreateDescriptorPool(m_device, &info, 0, &m_descriptor_pool));

		VkDescriptorSetAllocateInfo allocate_info = {};
		allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocate_info.descriptorPool = m_descriptor_pool;
		allocate_info.descriptorSetCount = 1;
		allocate_info.pSetLayouts = &m_descriptor_set_layout;
		VK_CHECK_RESULT(vkAllocateDescriptorSets(m_device, &allocate_info, &m_descriptor_set));
	}

	void layer::createShaderModule(const uint32_t* spv, size_t size, const std::string& source)
	{
		VkShaderModuleCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		if (spv)
		{
			create_info.pCode = spv;
			create_info.codeSize = size;
		}
		else
		{
			/*/
					   std::vector<uint32_t> code;
					   code = compile("shader", shaderc_compute_shader, source);
					   create_info.pCode = code.data();
					   create_info.codeSize = sizeof(uint32_t) * code.size();*/
		}
		VK_CHECK_RESULT(vkCreateShaderModule(m_device, &create_info, 0, &m_module));
	}

	void layer::createPipeline(size_t push_constants_size, VkSpecializationInfo* specialization_info)
	{
		VkPipelineShaderStageCreateInfo stage_create_info = {};
		stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		stage_create_info.module = m_module;
		stage_create_info.pName = "main";
		stage_create_info.pSpecializationInfo = specialization_info;
		VkPushConstantRange push_constant_ranges[1] = {};
		push_constant_ranges[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		push_constant_ranges[0].offset = 0;
		push_constant_ranges[0].size = static_cast<uint32_t>(push_constants_size);
		VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
		pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		if (push_constants_size != 0)
		{
			pipeline_layout_create_info.pushConstantRangeCount = 1;
			pipeline_layout_create_info.pPushConstantRanges = push_constant_ranges;
		}
		pipeline_layout_create_info.setLayoutCount = 1;
		pipeline_layout_create_info.pSetLayouts = &m_descriptor_set_layout;
		VK_CHECK_RESULT(vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, 0, &m_pipeline_layout));

		VkComputePipelineCreateInfo pipeline_create_info = {};
		pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipeline_create_info.stage = stage_create_info;
		pipeline_create_info.layout = m_pipeline_layout;
		VK_CHECK_RESULT(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipeline_create_info, 0, &m_pipeline));
	}

	void layer::createCommandBuffer()
	{
		VkCommandBufferAllocateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.commandPool = kCmdPool;
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = 1;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(m_device, &info, &m_cmd_buffer));
	}

	void layer::recordCommandBuffer(void* push_constants, size_t push_constants_size)
	{
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		kContextMtx.lock();
		// TODO: lock mutex lock(kContextMtx);
		VK_CHECK_RESULT(vkBeginCommandBuffer(m_cmd_buffer, &beginInfo));
		if (push_constants)
			vkCmdPushConstants(m_cmd_buffer, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
				static_cast<uint32_t>(push_constants_size), push_constants);
		vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
		vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 0, 1,
			&m_descriptor_set, 0, nullptr);
		vkCmdDispatch(m_cmd_buffer, m_group_x, m_group_y, m_group_z);

		VK_CHECK_RESULT(vkEndCommandBuffer(m_cmd_buffer));
		kContextMtx.unlock();
	}

	void layer::runCommandBuffer()
	{
		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &m_cmd_buffer;

		VkFence fence;
		VkFenceCreateInfo fence_create_info_ = {};
		fence_create_info_.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_create_info_.flags = 0;

		VK_CHECK_RESULT(vkCreateFence(m_device, &fence_create_info_, NULL, &fence));
		{
			kContextMtx.lock();
			//TODO: lock context cv::AutoLock lock(kContextMtx);
			VK_CHECK_RESULT(vkQueueSubmit(kQueue, 1, &submit_info, fence));
			kContextMtx.unlock();
		}
		VK_CHECK_RESULT(vkWaitForFences(m_device, 1, &fence, VK_TRUE, 100000000000));
		vkDestroyFence(m_device, fence, nullptr);
	}

	namespace layers
	{
		bool Module::forward(std::vector<tensor*>& x, std::vector<tensor*>& y)
		{
			return true;
		}

		void Module::backward()
		{
			for (int l = 0; l < module_list.size(); ++l)
			{
				module_list.at(l)->backward();
			}
		}

		void Module::backward(tensor* Cost)
		{
			tensor* grad = Cost;
		}

		void Module::execute()
		{
			for (auto layer : forward_layers)
			{
				layer->run();
			}
		}

		void Module::super_run()
		{
			auto tmp = *get_module();
			for (Module* m : tmp)
			{
				m->execute();
			}
		}
	}
}
