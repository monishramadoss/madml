#ifndef LAYER_H
#define LAYER_H
#include "madml.h"

#include <string>
#include <vector>

namespace kernel
{
	class context;

	class layer
	{
	public:
		layer();
		virtual ~layer();
		virtual bool forward(std::vector<tensor*>& ins, std::vector<tensor*>& outs) = 0;

		bool run()
		{
			runCommandBuffer();
			return true;
		}

	protected:
		void initVulkanThing(int buffer_num);
		void createDescriptorSetLayout(int buffer_num);
		void createDescriptorSet(int buffer_num);
		void createShaderModule(const uint32_t* spv, size_t sz, const std::string& source = std::string());
		void createPipeline(size_t push_constants_size = 0, VkSpecializationInfo* specialization_info = nullptr);
		void createCommandBuffer();
		void recordCommandBuffer(void* push_constants = nullptr, size_t push_constants_size = 0);
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

	namespace layers
	{
		class Module
		{
		protected:
			std::vector<layer*> forward_layers;
			std::vector<layer*> gradient_layers;
			std::vector<layer*> backward_layers;
			int batch_size = 0;

		public:

			static std::vector<Module*> module_list;
			virtual std::vector<Module*>* get_module() = 0;

			void add_layer(Module* obj)
			{
				get_module()->push_back(obj);
			}

			virtual bool forward(std::vector<tensor*>& x, std::vector<tensor*>& z);
			virtual bool operator()(tensor* x, tensor* y) = 0;
			virtual void backward();
			void backward(tensor* Cost);

			virtual void update_weight() = 0;
			void execute();
			void super_run();

			// static std::vector<tensor*> inputs;
			// static std::vector<tensor*> outputs;
			// static std::vector<tensor*> weights;
			// static std::vector<tensor*> biases;
		};
	}
}

#endif
