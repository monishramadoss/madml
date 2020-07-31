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
			void execute();
			void execute_b();
			virtual void backward() = 0;
			virtual void bck_callback() {};
			virtual void fwd_callback() = 0;

			std::vector<int> parents;
			std::vector<size_t> children;
			std::shared_ptr<Module> getptr();

		protected:
			std::string m_type;
			std::vector<size_t> inputs;
			std::vector<size_t> outputs;
			std::vector<size_t> weights;
			std::vector<size_t> biases;
			std::vector<size_t> temporaries;

			bool requires_sub_graph = false;
			std::vector<Module*> sub_graph;

			static std::vector<Module*>& get_module();
			static bool& sub_graph_bit();
			static void set_sub_graph();
			static void unset_sub_graph();

			static void zero_grad();
			static void add_module(Module* M);

			int get_input_id(size_t i);

			int batch_size = 0;
			float lr = 0.0001f;

			int id = 0;
			static int& get_object_id();
			void update_id();

			std::vector<size_t> execution_order;
			std::vector<std::vector<size_t>> adj_mat;
			std::vector<bool> visted;
			std::shared_ptr<tensor> x, y, w, b;
		private:
			//void BFS(std::vector<std::vector<int>> adj, int s = 0);
		};
	}

	class Base_Layer : public layer, public layers::Module
	{
	public:
		Base_Layer(int forward_buffers, int backward_buffers = -1, bool in_place = false);
		virtual void fwd_callback() override;
		virtual void bck_callback() override;
	protected:
		bool m_in_place;
		operator_param m_param;

		template <typename T = operator_param> inline std::shared_ptr<tensor>& layer_construct_forward(const uint32_t* shader, size_t codeSize, const std::shared_ptr<tensor>& x, T& m_param, Format fmt = Format::kFormatFp32, std::vector<int> output_shape = {});
		template <typename T = operator_param> inline std::shared_ptr<tensor>& layer_construct_forward(const uint32_t* shader, size_t codeSize, const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w, T& m_param, Format fmt = Format::kFormatFp32, std::vector<int> output_shape = {});
		template <typename T = operator_param> void layer_construct_backward(const uint32_t* shader, size_t codeSize, T m_param);
	};
}

namespace kernel
{
	template<class T>
	std::shared_ptr<tensor>& Base_Layer::layer_construct_forward(const uint32_t* shader, size_t codeSize, const std::shared_ptr<tensor>& x, T& param, Format fmt, std::vector<int> output_shape)
	{
		inputs.push_back(x->getId());
		this->x = x;
		/*if (m_in_place && output_shape.size() == 0)
			y = x;*/
			//else {
		if (output_shape.size() != 0)
			y = std::make_shared<tensor>(tensor(0.0, output_shape, fmt));
		else
			y = std::make_shared<tensor>(tensor(0.0, this->x->getShape()));
		//}
		outputs.push_back(y->getId());

		if (m_pipeline_forward == nullptr)
		{
			param.total = this->x->count();
			computeGroupCount();
			createShaderModuleForward(shader, codeSize);
			createPipelineForward(sizeof(T));
		}

		bindTensor(m_device, *this->x, 0, m_descriptor_set_forward);
		bindTensor(m_device, *y, 1, m_descriptor_set_forward);

		recordCommandBufferForward(static_cast<void*>(&param), sizeof(T));
		parents.push_back(get_input_id(this->x->getId()));
		runCommandBufferForward();
		return y;
	}

	template<class T>
	std::shared_ptr<tensor>& Base_Layer::layer_construct_forward(const uint32_t* shader, size_t codeSize, const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w, T& m_param, Format fmt, std::vector<int> output_shape)
	{
		inputs.push_back(x->getId());
		inputs.push_back(w->getId());
		this->x = x;
		this->w = w;
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
			if (m_group_x == 0 || m_group_y == 0 || m_group_z == 0)
			{
				std::cout << "GROUP DIMMS OFF" << std::endl;
			}
			createShaderModuleForward(shader, codeSize);
			createPipelineForward(sizeof(T));
		}

		bindTensor(m_device, *x, 0, m_descriptor_set_forward);
		bindTensor(m_device, *w, 1, m_descriptor_set_forward);
		bindTensor(m_device, *y, 2, m_descriptor_set_forward);

		recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(T));

		parents.push_back(get_input_id(x->getId()));
		parents.push_back(get_input_id(w->getId()));
		runCommandBufferForward();
		return y;
	}

	template<class T>
	void Base_Layer::layer_construct_backward(const uint32_t* shader, size_t codeSize, T m_param)
	{
		if (m_pipeline_backward == nullptr)
		{
			createShaderModuleBackward(shader, codeSize);
			createPipelineBackward(sizeof(T));
		}

		recordCommandBufferBackward(static_cast<void*>(&m_param), sizeof(T));
	}
}

#endif
