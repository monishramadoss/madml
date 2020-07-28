#include "common.h"
#include "utils.h"
#include "layer.h"

namespace kernel
{
	layer::layer()
	{
		createContext();
		m_device = kDevice;
		m_pipeline_forward = nullptr;
		m_cmd_buffer_forward = nullptr;
		m_descriptor_pool_forward = nullptr;
		m_descriptor_set_forward = nullptr;
		m_descriptor_set_layout_forward = nullptr;
		m_pipeline_layout_forward = nullptr;
		m_module_forward = nullptr;

		m_pipeline_backward = nullptr;
		m_cmd_buffer_backward = nullptr;
		m_descriptor_pool_backward = nullptr;
		m_descriptor_set_backward = nullptr;
		m_descriptor_set_layout_backward = nullptr;
		m_pipeline_layout_backward = nullptr;
		m_module_backward = nullptr;

		m_group_x = 1;
		m_group_y = 1;
		m_group_z = 1;
	}

	layer::~layer()
	{
		vkDestroyShaderModule(m_device, m_module_forward, nullptr);
		vkDestroyDescriptorPool(m_device, m_descriptor_pool_forward, nullptr);
		vkDestroyPipeline(m_device, m_pipeline_forward, nullptr);
		vkDestroyPipelineLayout(m_device, m_pipeline_layout_forward, nullptr);

		vkDestroyShaderModule(m_device, m_module_backward, nullptr);
		vkDestroyDescriptorPool(m_device, m_descriptor_pool_backward, nullptr);
		vkDestroyPipeline(m_device, m_pipeline_backward, nullptr);
		vkDestroyPipelineLayout(m_device, m_pipeline_layout_backward, nullptr);
	}

	void layer::initVulkanThing(int buffer_num_forward, int buffer_num_backward)
	{
		createDescriptorSetLayoutForward(buffer_num_forward);
		createDescriptorSetForward(buffer_num_forward);
		createCommandBufferForward();

		if (buffer_num_backward == -1)
			buffer_num_backward = buffer_num_forward;
		createDescriptorSetLayoutBackward(buffer_num_backward);
		createDescriptorSetBackward(buffer_num_backward);
		createCommandBufferBackward();
	}

	void layer::createDescriptorSetLayoutForward(int buffer_num)
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
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device, &info, 0, &m_descriptor_set_layout_forward));
	}

	void layer::createDescriptorSetForward(int buffer_num)
	{
		VkDescriptorPoolSize pool_size = {};
		pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		pool_size.descriptorCount = buffer_num;

		VkDescriptorPoolCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		info.maxSets = 1;
		info.poolSizeCount = 1;
		info.pPoolSizes = &pool_size;
		VK_CHECK_RESULT(vkCreateDescriptorPool(m_device, &info, 0, &m_descriptor_pool_forward));

		VkDescriptorSetAllocateInfo allocate_info = {};
		allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocate_info.descriptorPool = m_descriptor_pool_forward;
		allocate_info.descriptorSetCount = 1;
		allocate_info.pSetLayouts = &m_descriptor_set_layout_forward;
		VK_CHECK_RESULT(vkAllocateDescriptorSets(m_device, &allocate_info, &m_descriptor_set_forward));
	}

	void layer::createShaderModuleForward(const uint32_t* spv, size_t size, const std::string& source)
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
			//std::vector<uint32_t> code;
			//code = compile("shader", shaderc_compute_shader, source);
			//create_info.pCode = code.data();
			//create_info.codeSize = sizeof(uint32_t) * code.size();
		}
		VK_CHECK_RESULT(vkCreateShaderModule(m_device, &create_info, 0, &m_module_forward));
	}

	void layer::createPipelineForward(size_t push_constants_size, VkSpecializationInfo* specialization_info)
	{
		VkPipelineShaderStageCreateInfo stage_create_info = {};
		stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		stage_create_info.module = m_module_forward;
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
		pipeline_layout_create_info.pSetLayouts = &m_descriptor_set_layout_forward;
		VK_CHECK_RESULT(vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, 0, &m_pipeline_layout_forward));

		VkComputePipelineCreateInfo pipeline_create_info = {};
		pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipeline_create_info.stage = stage_create_info;
		pipeline_create_info.layout = m_pipeline_layout_forward;
		VK_CHECK_RESULT(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipeline_create_info, 0, &m_pipeline_forward));
	}

	void layer::createCommandBufferForward()
	{
		VkCommandBufferAllocateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.commandPool = kCmdPool;
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = 1;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(m_device, &info, &m_cmd_buffer_forward));
	}

	void layer::recordCommandBufferForward(void* push_constants, size_t push_constants_size) const
	{
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		kContextMtx.lock();
		// TODO: lock mutex lock(kContextMtx);
		VK_CHECK_RESULT(vkBeginCommandBuffer(m_cmd_buffer_forward, &beginInfo));
		if (push_constants)
			vkCmdPushConstants(m_cmd_buffer_forward, m_pipeline_layout_forward, VK_SHADER_STAGE_COMPUTE_BIT, 0,
				static_cast<uint32_t>(push_constants_size), push_constants);
		vkCmdBindPipeline(m_cmd_buffer_forward, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_forward);
		vkCmdBindDescriptorSets(m_cmd_buffer_forward, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout_forward, 0, 1,
			&m_descriptor_set_forward, 0, nullptr);
		vkCmdDispatch(m_cmd_buffer_forward, m_group_x, m_group_y, m_group_z);

		VK_CHECK_RESULT(vkEndCommandBuffer(m_cmd_buffer_forward));
		kContextMtx.unlock();
	}

	void layer::runCommandBufferForward() const
	{
		//TODO: generate with thread pool;
		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &m_cmd_buffer_forward;

		VkFence fence;
		VkFenceCreateInfo fence_create_info_ = {};
		fence_create_info_.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_create_info_.flags = 0;

		VK_CHECK_RESULT(vkCreateFence(m_device, &fence_create_info_, NULL, &fence));
		{
			kContextMtx.lock();
			VK_CHECK_RESULT(vkQueueSubmit(kQueue, 1, &submit_info, fence));
			kContextMtx.unlock();
		}
		VK_CHECK_RESULT(vkWaitForFences(m_device, 1, &fence, VK_TRUE, 100000000000));
		vkDestroyFence(m_device, fence, nullptr);
	}

	void layer::createDescriptorSetLayoutBackward(int buffer_num)
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
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device, &info, 0, &m_descriptor_set_layout_backward));
	}

	void layer::createDescriptorSetBackward(int buffer_num)
	{
		VkDescriptorPoolSize pool_size = {};
		pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		pool_size.descriptorCount = buffer_num;

		VkDescriptorPoolCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		info.maxSets = 1;
		info.poolSizeCount = 1;
		info.pPoolSizes = &pool_size;
		VK_CHECK_RESULT(vkCreateDescriptorPool(m_device, &info, 0, &m_descriptor_pool_backward));

		VkDescriptorSetAllocateInfo allocate_info = {};
		allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocate_info.descriptorPool = m_descriptor_pool_backward;
		allocate_info.descriptorSetCount = 1;
		allocate_info.pSetLayouts = &m_descriptor_set_layout_backward;
		VK_CHECK_RESULT(vkAllocateDescriptorSets(m_device, &allocate_info, &m_descriptor_set_backward));
	}

	void layer::createShaderModuleBackward(const uint32_t* spv, size_t size, const std::string& source)
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
			//std::vector<uint32_t> code;
			//code = compile("shader", shaderc_compute_shader, source);
			//create_info.pCode = code.data();
			//create_info.codeSize = sizeof(uint32_t) * code.size();
		}
		VK_CHECK_RESULT(vkCreateShaderModule(m_device, &create_info, 0, &m_module_backward));
	}

	void layer::createPipelineBackward(size_t push_constants_size, VkSpecializationInfo* specialization_info)
	{
		VkPipelineShaderStageCreateInfo stage_create_info = {};
		stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		stage_create_info.module = m_module_backward;
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
		pipeline_layout_create_info.pSetLayouts = &m_descriptor_set_layout_backward;
		VK_CHECK_RESULT(vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, 0, &m_pipeline_layout_backward));

		VkComputePipelineCreateInfo pipeline_create_info = {};
		pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipeline_create_info.stage = stage_create_info;
		pipeline_create_info.layout = m_pipeline_layout_backward;
		VK_CHECK_RESULT(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipeline_create_info, 0, &m_pipeline_backward));
	}

	void layer::createCommandBufferBackward()
	{
		VkCommandBufferAllocateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.commandPool = kCmdPool;
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = 1;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(m_device, &info, &m_cmd_buffer_backward));
	}

	void layer::recordCommandBufferBackward(void* push_constants, size_t push_constants_size)
	{
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		kContextMtx.lock();
		// TODO: lock mutex lock(kContextMtx);
		VK_CHECK_RESULT(vkBeginCommandBuffer(m_cmd_buffer_backward, &beginInfo));
		if (push_constants)
			vkCmdPushConstants(m_cmd_buffer_backward, m_pipeline_layout_backward, VK_SHADER_STAGE_COMPUTE_BIT, 0,
				static_cast<uint32_t>(push_constants_size), push_constants);
		vkCmdBindPipeline(m_cmd_buffer_backward, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_backward);
		vkCmdBindDescriptorSets(m_cmd_buffer_backward, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout_backward, 0, 1,
			&m_descriptor_set_backward, 0, nullptr);
		vkCmdDispatch(m_cmd_buffer_backward, m_group_x, m_group_y, m_group_z);

		VK_CHECK_RESULT(vkEndCommandBuffer(m_cmd_buffer_backward));
		kContextMtx.unlock();
	}

	void layer::runCommandBufferBackward()
	{
		//TODO: generate with thread pool;
		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &m_cmd_buffer_backward;

		VkFence fence;
		VkFenceCreateInfo fence_create_info_ = {};
		fence_create_info_.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_create_info_.flags = 0;

		VK_CHECK_RESULT(vkCreateFence(m_device, &fence_create_info_, NULL, &fence));
		{
			kContextMtx.lock();
			VK_CHECK_RESULT(vkQueueSubmit(kQueue, 1, &submit_info, fence));
			kContextMtx.unlock();
		}
		VK_CHECK_RESULT(vkWaitForFences(m_device, 1, &fence, VK_TRUE, 100000000000));
		vkDestroyFence(m_device, fence, nullptr);
	}

	namespace layers
	{
		void Module::update_weight()
		{
		}

		void Module::backward()
		{
		}

		void Module::execute()
		{
			auto& M = get_module();
			auto& T = get_tensors();
			/// <summary>
			///  weight = -2;
			///  bias = -3;
			///  input = -1;
			/// </summary>
			for (auto* m : M)
			{
				std::cout << m->id << " " << m->m_type << " ";
				for (auto p : m->parents)
					std::cout << p << " ";
				std::cout << "\n";
			}
			//std::cout << *std::max_element(sanity_check.begin(), sanity_check.end());
			//std::cout << std::endl;
		}

		std::vector<tensor*>& Module::get_tensors()
		{
			static std::vector<tensor*> T;
			return T;
		}

		std::vector<tensor*>& Module::get_gradients()
		{
			static std::vector<tensor*> G;
			return G;
		}

		std::vector<Module*>& Module::get_module()
		{
			static std::vector<Module*> M;
			return M;
		}

		tensor* Module::get_grad(int id)
		{
			auto T = get_gradients();
			return T[id];
		}

		void Module::add_tensor(tensor* T)
		{
			auto& t = get_tensors();
			t.push_back(T);
		}

		void Module::add_gradient(tensor* G)
		{
			auto& g = get_gradients();
			g.push_back(G);
		}

		void Module::add_module(Module* M)
		{
			auto& m = get_module();
			m.push_back(M);
		}

		void Module::zero_grad()
		{
			auto& G = get_gradients();
			auto& T = get_tensors();
			if (G.size() != T.size())
			{
				for (auto t : T)
				{
					G.push_back(new tensor(0.0, t->getShape()));
				}
			}
		}

		void Module::BFS(std::vector<std::vector<int>> adj, int s)
		{
			int v = adj.size();
			std::vector<bool> visited(v, false);
			std::vector<int> q;
			q.push_back(s);
			visited[s] = true;
			int vis;
			while (!q.empty())
			{
				vis = q[0];
				std::cout << vis << " -> ";
				q.erase(q.begin());
				for (int i = 0; i < v; i++)
				{
					if (adj[vis][i] == 1 && (!visited[i]))
					{
						q.push_back(i);
						visited[i] = true;
					}
				}
			}
		}

		int Module::get_input_id(int i)
		{
			auto& M = get_module();
			for (auto* m : M)
			{
				if (std::find(m->outputs.begin(), m->outputs.end(), i) != m->outputs.end())
					return m->id;
				if (std::find(m->weights.begin(), m->weights.end(), i) != m->weights.end())
					return -2;
				if (std::find(m->biases.begin(), m->biases.end(), i) != m->biases.end())
					return -3;
			}
			return -1;
		}
		int& Module::get_object_id()
		{
			static int objId;
			return objId;
		}

		void Module::update_id()
		{
			auto& objId = get_object_id();
			id = objId++;
		}
	}

	Base_Layer::Base_Layer(int forward_buffers, int backward_buffers, bool in_place) : m_in_place(in_place), m_param({ 0 })
	{
		update_id();
		add_module(this);
		initVulkanThing(forward_buffers, backward_buffers);
	}

	void Base_Layer::fwd_callback()
	{
		runCommandBufferForward();
	}

	void Base_Layer::bwd_callback()
	{
		runCommandBufferBackward();
	}
}