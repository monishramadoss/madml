#ifndef LAYER_H
#define LAYER_H
#include "halaml.h"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <future>
#include <list>

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
	int runCommandBuffer();
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

	friend class layers::Module;

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

		std::vector<int> parents;
		std::vector<size_t> children;
		std::shared_ptr<Module> getptr();
		std::shared_ptr<tensor> dx, dy, dw, db;

		std::vector<std::future<int>>& get_futures();
		int get_id() const;

	protected:
		std::string m_type;
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

		Module* get_input_id(size_t i);

		int batch_size = 0;
		float lr = 0.0001f;

		int id = 0;
		static int& get_object_id();
		void update_id();
		virtual int set_backward() { return -1; }
		std::vector<std::future<int>> m_futures;

		std::vector<size_t> execution_order;

		std::shared_ptr<tensor> x, y, w, b;

	private:

		//void BFS(std::vector<std::vector<int>> adj, int s = 0);
	};
}

template <class T = operator_param>
class Base_Layer : public layer, public layers::Module
{
public:
	Base_Layer(int forward_buffers, bool in_place = false);

protected:
	const uint32_t* bck_shader;
	size_t bck_codeSize;
	Base_Layer* derivative;

	bool m_in_place;
	T m_param;
	void computeGroupCount() override;
	void set_group(int x, int y, int z);
	int set_backward() override;

	std::shared_ptr<tensor>& layer_construct_forward(const uint32_t* shader, size_t codeSize,
		const std::shared_ptr<tensor>& x, Format fmt = Format::kFormatFp32,
		std::vector<int> output_shape = {});
	std::shared_ptr<tensor>& layer_construct_forward(const uint32_t* shader, size_t codeSize,
		const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w,
		Format fmt = Format::kFormatFp32, std::vector<int> output_shape = {});

	//template <typename T = operator_param> void layer_construct_backward(const uint32_t* shader, size_t codeSize, T m_param);
};

template <typename T>
Base_Layer<T>::Base_Layer(int forward_buffers, bool in_place) : m_in_place(in_place), m_param({ 0 })
{
	update_id();
	if (!sub_graph_bit())
		add_module(this);
	bck_shader = nullptr;
	bck_codeSize = 0;
	derivative = nullptr;
	initVulkanThing(forward_buffers);
}

template <typename T>
void Base_Layer<T>::computeGroupCount()
{
}

template <typename T>
void Base_Layer<T>::set_group(int x, int y, int z)
{
	m_group_x = x;
	m_group_y = y;
	m_group_z = z;
}

template <typename T>
int Base_Layer<T>::set_backward()
{
	if (train() && bck_codeSize && !parents.size())
	{
		auto M = get_module();
		if (!dy)
			dy = std::make_shared<tensor>(tensor(0.0, y->getShape()));

		if (parents.size() > 1)
		{
			int j = parents[1];
			if (!dw)
				dw = std::make_shared<tensor>(tensor(0.0, w->getShape()));
			if (j > 0)
				M[j]->dy = dw;
			dx = derivative->layer_construct_forward(bck_shader, bck_codeSize, dy, dw, Format::kFormatFp32, x->getShape());
		}
		else
		{
			dx = derivative->layer_construct_forward(bck_shader, bck_codeSize, dy, Format::kFormatFp32, x->getShape());
		}

		if (parents.size() > 0)
		{
			int i = parents[0];

			if (i >= 0)
				M[i]->dy = dx;
		}

		return dy->getId();
	}
	return -2;
}

template <typename T>
std::shared_ptr<tensor>& Base_Layer<T>::layer_construct_forward(const uint32_t* shader, size_t codeSize,
	const std::shared_ptr<tensor>& _x, Format fmt,
	std::vector<int> output_shape)
{
	x = _x;
	if (!y)
	{
		if (output_shape.size() != 0)
			y = std::make_shared<tensor>(tensor(0.0, output_shape, fmt));
		else
			y = std::make_shared<tensor>(tensor(0.0, x->getShape()));
	}

	if (m_pipeline == nullptr)
	{
		m_param.total = x->count();
		computeGroupCount();
		createShaderModule(shader, codeSize);
		createPipeline(sizeof(T));
	}

	bindTensor(x, 0);
	bindTensor(y, 1);

	auto m = get_input_id(x->getId());
	parents.push_back(!m ? m->get_id() : -1);

	if (train() && bck_codeSize && !derivative)
	{
		derivative = new Base_Layer<T>(2, false);
		derivative->m_param = m_param;
		derivative->set_group(m_group_x, m_group_y, m_group_z);
		set_backward();
	}

	if (!m)
	{
		if (!m->get_futures().size())
		{
			m->get_futures().back().get();
		}
	}

	recordCommandBuffer(static_cast<void*>(&m_param), sizeof(T));
	runCommandBuffer();

	return y;
}

template <typename T>
std::shared_ptr<tensor>& Base_Layer<T>::layer_construct_forward(const uint32_t* shader, size_t codeSize,
	const std::shared_ptr<tensor>& _x,
	const std::shared_ptr<tensor>& _w, Format fmt,
	std::vector<int> output_shape)
{
	x = _x;
	w = _w;

	if (!y)
	{
		if (output_shape.size() != 0)
			y = std::make_shared<tensor>(tensor(0.0, output_shape, fmt));
		else
			y = std::make_shared<tensor>(tensor(0.0, x->getShape()));
	}

	if (m_pipeline == nullptr)
	{
		m_param.total = x->count();
		computeGroupCount();
		createShaderModule(shader, codeSize);
		createPipeline(sizeof(T));
	}

	bindTensor(x, 0);
	bindTensor(w, 1);
	bindTensor(y, 2);

	Module* m1 = get_input_id(x->getId());
	Module* m2 = get_input_id(w->getId());

	parents.push_back(!m1 ? m1->get_id() : -1);
	parents.push_back(!m2 ? m2->get_id() : -1);

	if (train() && bck_codeSize)
	{
		derivative = new Base_Layer<T>(3, false);
		derivative->m_param = m_param;
		derivative->set_group(m_group_x, m_group_y, m_group_z);
		set_backward();
	}

	recordCommandBuffer(static_cast<void*>(&m_param), sizeof(T));
	runCommandBuffer();

	return y;
}

#endif
