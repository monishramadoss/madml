#ifndef LAYER_H
#define LAYER_H
#include "madml.h"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <list>

namespace kernel
{
	namespace layers
	{
		class Module;
	}

	class context;
	struct operator_param
	{
		int total;
	};

	class layer
	{
	public:
		layer();
		virtual ~layer();
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
	};

	namespace layers
	{
		class Module : std::enable_shared_from_this<Module>
		{
		public:
			virtual void update_weight();
			virtual void execute();
			virtual void backward();

			std::vector<int> parents;
			std::shared_ptr<Module> getptr();

		protected:
			std::string m_type;
			std::vector<int> inputs;
			std::vector<int> outputs;
			std::vector<int> weights;
			std::vector<int> biases;
			std::vector<int> temporaries;
			bool requires_sub_graph = false;
			std::vector<Module*> sub_graph;
			static std::vector<std::shared_ptr<tensor>>& get_tensors();
			static std::vector<std::shared_ptr<tensor>>& get_gradients();
			static std::vector<Module*>& get_module();
			static bool& sub_graph_bit();
			static void set_sub_graph();
			static void unset_sub_graph();

			static std::shared_ptr<tensor> get_grad(int id);
			static void zero_grad();

			static void add_tensor(std::shared_ptr<tensor>T);
			static void add_gradient(std::shared_ptr<tensor>G);
			static void add_module(Module* M);

			int get_input_id(int i);

			int batch_size = 0;
			float lr = 0.0001f;
			virtual void back_propagate() {};

			int id = 0;
			static int& get_object_id();
			void update_id();

			friend class tensor;
		private:
			//void BFS(std::vector<std::vector<int>> adj, int s = 0);
		};
	}

	class Base_Layer : public layer, public layers::Module
	{
	public:
		Base_Layer(int forward_buffers, int backward_buffers = -1, bool in_place = false);
		virtual void fwd_callback();
		virtual void bwd_callback();
	protected:
		bool m_in_place;
		operator_param m_param;

		template <typename T = operator_param> inline std::shared_ptr<tensor>layer_construct_forward(const uint32_t* shader, size_t codeSize, std::shared_ptr<tensor>x, T m_param, Format fmt = Format::kFormatFp32, std::vector<int> output_shape = {});
		template <typename T = operator_param> inline std::shared_ptr<tensor>layer_construct_forward(const uint32_t* shader, size_t codeSize, std::shared_ptr<tensor>x, std::shared_ptr<tensor>w, T m_param, Format fmt = Format::kFormatFp32, std::vector<int> output_shape = {});
		template <typename T = operator_param> void layer_construct_backward(const uint32_t* shader, size_t codeSize, T m_param);
	};
}

namespace kernel
{
	template<class T>
	std::shared_ptr<tensor>Base_Layer::layer_construct_forward(const uint32_t* shader, size_t codeSize, std::shared_ptr<tensor>x, T m_param, Format fmt, std::vector<int> output_shape)
	{
		inputs.push_back(x->getId());
		std::shared_ptr<tensor>y;
		/*if (m_in_place && output_shape.size() == 0)
			y = x;*/
			//else {
		if (output_shape.size() != 0)
			y = std::make_shared<tensor>(tensor(0.0, output_shape, fmt));
		else
			y = std::make_shared<tensor>(tensor(0.0, x->getShape()));
		//}
		outputs.push_back(y->getId());

		if (m_pipeline_forward == nullptr)
		{
			m_param.total = x->count();
			computeGroupCount();
			createShaderModuleForward(shader, codeSize);
			createPipelineForward(sizeof(T));
		}

		bindTensor(m_device, x, 0, m_descriptor_set_forward);
		bindTensor(m_device, y, 1, m_descriptor_set_forward);

		recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(T));

		parents.push_back(get_input_id(x->getId()));
		return y;
	}

	template<class T>
	std::shared_ptr<tensor>Base_Layer::layer_construct_forward(const uint32_t* shader, size_t codeSize, std::shared_ptr<tensor>x, std::shared_ptr<tensor>w, T m_param, Format fmt, std::vector<int> output_shape)
	{
		inputs.push_back(x->getId());
		inputs.push_back(w->getId());
		std::shared_ptr<tensor>y;

		/*if (m_in_place && output_shape.size() == 0)
			y = x;*/
			//else {
		if (output_shape.size() != 0)
			y = std::make_shared<tensor>(tensor(0.0, output_shape, fmt));
		else
			y = std::make_shared<tensor>(tensor(0.0, x->getShape()));
		//}
		outputs.push_back(y->getId());
		if (m_pipeline_forward == nullptr)
		{
			m_param.total = x->count();
			computeGroupCount();
			createShaderModuleForward(shader, codeSize);
			createPipelineForward(sizeof(T));
		}

		bindTensor(m_device, x, 0, m_descriptor_set_forward);
		bindTensor(m_device, w, 1, m_descriptor_set_forward);
		bindTensor(m_device, y, 2, m_descriptor_set_forward);

		recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(T));

		parents.push_back(get_input_id(x->getId()));
		parents.push_back(get_input_id(w->getId()));
		return y;
	}

	template<class T>
	void Base_Layer::layer_construct_backward(const uint32_t* shader, size_t codeSize, T m_param)
	{
		if (m_pipeline_backward == nullptr)
		{
			computeGroupCount();
			createShaderModuleBackward(shader, codeSize);
			createPipelineBackward(sizeof(T));
		}

		int binding = 0;
		for (int o : outputs)
		{
			bindTensor(m_device, get_grad(o), binding++, m_descriptor_set_backward);
		}
		for (int w : weights)
		{
			bindTensor(m_device, get_grad(w), binding++, m_descriptor_set_backward);
		}
		for (int b : biases)
		{
			bindTensor(m_device, get_grad(b), binding++, m_descriptor_set_backward);
		}
		for (int i : inputs)
		{
			bindTensor(m_device, get_grad(i), binding++, m_descriptor_set_backward);
		}

		recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(T));
	}
}

#endif
