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
		void initVulkanThing(int buffer_num_forward);

		void createDescriptorSetLayout(int buffer_num);
		void createDescriptorSet(int buffer_num);
		void createShaderModule(const uint32_t* spv, size_t sz, const std::string& source = std::string());
		void createPipeline(size_t push_constants_size = 0, VkSpecializationInfo* specialization_info = nullptr);
		void createCommandBuffer();
		void recordCommandBuffer(void* push_constants = nullptr, size_t push_constants_size = 0) const;
		void runCommandBuffer() const;
		void bindTensor(std::shared_ptr<tensor> tensor, int binding);

		virtual void computeGroupCount() = 0;
		VkDevice m_device;

		VkPipeline m_pipeline;
		VkCommandBuffer m_cmd_buffer;
		VkDescriptorPool m_descriptor_pool;
		VkDescriptorSet m_descriptor_set;
		VkDescriptorSetLayout m_descriptor_set_layout;
		VkPipelineLayout m_pipeline_layout;
		VkShaderModule m_module;

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

			virtual void run() = 0;

			std::vector<int> parents;
			std::vector<size_t> children;
			std::shared_ptr<Module> getptr();
			std::shared_ptr<tensor> dx, dy, dw, db;

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

			static bool& train();
			static void eval();

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
		Base_Layer(int forward_buffers, bool in_place = false);

	protected:
		const uint32_t* bck_shader;
		size_t bck_codeSize;
		std::shared_ptr<Base_Layer> derivative;
		bool m_in_place;
		operator_param m_param;
		void computeGroupCount() override;
		void set_group(int x, int y, int z);
		virtual void run() override;
		template <typename T = operator_param> inline std::shared_ptr<tensor>& layer_construct_forward(const uint32_t* shader, size_t codeSize, const std::shared_ptr<tensor>& x, T& m_param, Format fmt = Format::kFormatFp32, std::vector<int> output_shape = {});
		template <typename T = operator_param> inline std::shared_ptr<tensor>& layer_construct_forward(const uint32_t* shader, size_t codeSize, const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w, T& m_param, Format fmt = Format::kFormatFp32, std::vector<int> output_shape = {});
		//template <typename T = operator_param> void layer_construct_backward(const uint32_t* shader, size_t codeSize, T m_param);
	};

	namespace layers
	{
		class unary_operator : public Base_Layer
		{
		protected:
			void computeGroupCount() override;
		public:
			unary_operator(bool in_place);
			virtual std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x) = 0;
		};

		class binary_operator : public Base_Layer
		{
		protected:
			void computeGroupCount() override;
		public:
			binary_operator(bool in_place);
			virtual std::shared_ptr<tensor>& hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w) = 0;
		};
	}
}

namespace kernel
{
	template<class T>
	std::shared_ptr<tensor>& Base_Layer::layer_construct_forward(const uint32_t* shader, size_t codeSize, const std::shared_ptr<tensor>& _x, T& param, Format fmt, std::vector<int> output_shape)
	{
		x = _x;
		inputs.push_back(x->getId());
		/*if (m_in_place && output_shape.size() == 0)
			y = x;*/
			//else {
		if (output_shape.size() != 0)
			y = std::make_shared<tensor>(tensor(0.0, output_shape, fmt));
		else
			y = std::make_shared<tensor>(tensor(0.0, x->getShape()));
		//}
		outputs.push_back(y->getId());

		if (m_pipeline == nullptr)
		{
			param.total = x->count();
			computeGroupCount();
			createShaderModule(shader, codeSize);
			createPipeline(sizeof(T));
		}

		bindTensor(_x, 0);
		bindTensor(y, 1);

		recordCommandBuffer(static_cast<void*>(&param), sizeof(T));

		parents.push_back(get_input_id(get_input_id(x->getId())));
		if (train())
		{
			auto M = get_module();
			int i = parents[0];

			if (!dy)
				dy = std::make_shared<tensor>(tensor(0.0, y->getShape()));

			derivative = std::make_shared<Base_Layer>(Base_Layer(2, false));
			derivative->set_group(m_group_x, m_group_y, m_group_z);
			dx = derivative->layer_construct_forward<T>(bck_shader, bck_codeSize, dy, param, fmt, x->getShape());

			if (i > 0)
			{
				auto m1 = M[i];
				m1->dy = dx;
			}
		}

		return y;
	}

	template<class T>
	std::shared_ptr<tensor>& Base_Layer::layer_construct_forward(const uint32_t* shader, size_t codeSize, const std::shared_ptr<tensor>& _x, const std::shared_ptr<tensor>& _w, T& param, Format fmt, std::vector<int> output_shape)
	{
		x = _x;
		w = _w;

		inputs.push_back(x->getId());
		inputs.push_back(w->getId());
		/*if (m_in_place && output_shape.size() == 0)
			y = x;*/
			//else {
		if (output_shape.size() != 0)
			y = std::make_shared<tensor>(tensor(0.0, output_shape, fmt));
		else
			y = std::make_shared<tensor>(tensor(0.0, x->getShape()));
		//}
		outputs.push_back(y->getId());
		if (m_pipeline == nullptr)
		{
			param.total = x->count();
			computeGroupCount();
			if (m_group_x == 0 || m_group_y == 0 || m_group_z == 0)
			{
				std::cout << "GROUP DIMMS OFF" << std::endl;
			}
			createShaderModule(shader, codeSize);
			createPipeline(sizeof(T));
		}

		bindTensor(x, 0);
		bindTensor(w, 1);
		bindTensor(y, 2);

		recordCommandBuffer(static_cast<void*>(&param), sizeof(T));

		parents.push_back(get_input_id(x->getId()));
		parents.push_back(get_input_id(w->getId()));
		if (train() && bck_codeSize)
		{
			int i = parents[0];
			int j = parents[1];
			auto M = get_module();

			if (!dw)
				dw = std::make_shared<tensor>(tensor(0.0, w->getShape()));
			if (!dy)
				dy = std::make_shared<tensor>(tensor(0.0, y->getShape()));

			derivative = std::make_shared<Base_Layer>(Base_Layer(3, false));
			derivative->set_group(m_group_x, m_group_y, m_group_z);
			dx = derivative->layer_construct_forward<T>(bck_shader, bck_codeSize, dy, dw, param, fmt, x->getShape());

			if (i > 0)
			{
				auto m1 = M[i];
				m1->dy = dx;
			}
			if (j > 0)
			{
				auto m2 = M[j];
				m2->dy = dw;
			}
		}
		return y;
	}
}

#endif
