#include "common.h"
#include "utils.h"
#include "layer.h"

#define LOCAL_SZ_X 1024
#define MAX_COMPUTE_WORK_GROUP_COUNT 65535

namespace kernel
{
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

	void layer::initVulkanThing(int buffer_num_forward)
	{
		createDescriptorSetLayout(buffer_num_forward);
		createDescriptorSet(buffer_num_forward);
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
			//std::vector<uint32_t> code;
			//code = compile("shader", shaderc_compute_shader, source);
			//create_info.pCode = code.data();
			//create_info.codeSize = sizeof(uint32_t) * code.size();
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

	void layer::recordCommandBuffer(void* push_constants, size_t push_constants_size) const
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

	void layer::runCommandBuffer() const
	{
		//TODO: generate with thread pool;
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

		void DFS_f(size_t start, std::vector<bool>& visited, std::vector<std::vector<size_t>>& adj, std::vector<size_t>& execution_order)
		{
			execution_order.push_back(start);
			visited[start] = true;
			for (size_t i = 0; i < adj.size(); i++)
			{
				if (adj[start][i] == 1 && (!visited[i]))
				{
					DFS_f(i, visited, adj, execution_order);
				}
			}
		}

		void Module::execute()
		{
			auto& M = get_module();
			/// <summary>
			///  input = -1;
			///  weight = -2;
			///  bias = -3;
			///  temp = -4;
			/// </summary>
			if (execution_order.size() == 0)
			{
				adj_mat.resize(M.size(), std::vector<size_t>(M.size(), 0));
				visted.resize(M.size(), false);

				for (auto m : M)
				{
					for (auto p : m->parents)
					{
						adj_mat[m->id][p] = -1;
					}
				}

				for (size_t i = 0; i < M.size(); ++i)
				{
					for (size_t j = 0; j < M.size(); ++j)
					{
						if (adj_mat[i][j])
							M[j]->children.push_back(i);
					}
				}

				for (auto m : M)
				{
					for (auto c : m->children)
					{
						adj_mat[m->id][c] = 1;
					}
				}

				DFS_f(0, visted, adj_mat, execution_order);
			}

			for (size_t m_idx : execution_order)
			{
				if (M[m_idx]->requires_sub_graph)
				{
					std::cout << M[m_idx]->m_type << std::endl;
					//for (auto g : M[m_idx]->sub_graph)
						//std::cout << g->m_type << std::endl;
				}
				else
					std::cout << M[m_idx]->m_type << std::endl;

				if (M[m_idx]->x != nullptr)
					M[m_idx]->dy = std::make_shared<tensor>(tensor(0.0, M[m_idx]->x->getShape()));

				if (M[m_idx]->w != nullptr)
					M[m_idx]->dw = std::make_shared<tensor>(tensor(0.0, M[m_idx]->w->getShape()));
				if (M[m_idx]->b != nullptr)
					M[m_idx]->db = std::make_shared<tensor>(tensor(0.0, M[m_idx]->b->getShape()));

				M[m_idx]->set_derivative();
			}
			std::cout << "\n\n";
			for (auto i = execution_order.rbegin(); i != execution_order.rend(); ++i)
			{
				std::cout << M[*i]->m_type << std::endl;
			}
		}

		void Module::execute_b()
		{
		}

		std::vector<Module*>& Module::get_module()
		{
			static std::vector<Module*> M;
			return M;
		}

		bool& Module::sub_graph_bit()
		{
			static bool requires_sub_graph;
			return requires_sub_graph;
		}

		void Module::set_sub_graph()
		{
			bool& r = sub_graph_bit();
			r = true;
		}

		void Module::unset_sub_graph()
		{
			bool& r = sub_graph_bit();
			r = false;
		}

		bool& Module::train()
		{
			static bool training;
			training = true;
			return training;
		}

		void Module::eval()
		{
			auto t = train();
			t = false;
		}

		void Module::add_module(Module* M)
		{
			auto& m = get_module();
			m.push_back(M);
		}

		void Module::zero_grad()
		{
		}

		int Module::get_input_id(size_t i)
		{
			auto& M = get_module();
			for (auto m : M)
			{
				if (std::find(m->outputs.begin(), m->outputs.end(), i) != m->outputs.end())
				{
					return m->id;
				}
				if (std::find(m->weights.begin(), m->weights.end(), i) != m->weights.end())
				{
					return -2;
				}
				if (std::find(m->biases.begin(), m->biases.end(), i) != m->biases.end())
					return -3;
				if (std::find(m->temporaries.begin(), m->temporaries.end(), i) != m->temporaries.end())
					return -4;
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

		std::shared_ptr<Module> Module::getptr()
		{
			return shared_from_this();
		}
	}

	Base_Layer::Base_Layer(int forward_buffers, bool in_place) : m_in_place(in_place), m_param({ 0 })
	{
		update_id();
		bool is_derivative = m_type.find("d_") != std::string::npos;
		if (!sub_graph_bit() && train() && !is_derivative)
			add_module(this);
		else if (is_derivative)
			nullptr;

		initVulkanThing(forward_buffers);
	}

	void Base_Layer::run()
	{
		runCommandBuffer();
	}

	namespace layers
	{
		unary_operator::unary_operator(bool in_place) : Base_Layer(2, in_place)
		{
		}

		void unary_operator::computeGroupCount()
		{
			m_group_x = static_cast<int>(alignSize(m_param.total, LOCAL_SZ_X)) / LOCAL_SZ_X;
			if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
				m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
			m_group_y = 1;
			m_group_z = 1;
		}

		binary_operator::binary_operator(bool in_place) : Base_Layer(3, in_place)
		{
		}

		void binary_operator::computeGroupCount()
		{
			m_group_x = static_cast<int>(alignSize(m_param.total, LOCAL_SZ_X)) / LOCAL_SZ_X;
			if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
				m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
			m_group_y = 1;
			m_group_z = 1;
		}
	}
}