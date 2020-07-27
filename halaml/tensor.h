#ifndef TENSOR_H
#define TENSOR_H
#include <memory>
#include <numeric>
#include <random>
#include <vulkan/vulkan.h>
#include "madml.h"

namespace kernel
{
	class buffer;

	class tensor
	{
	public:
		tensor(Format fmt = Format::kFormatFp32);
		tensor(char* data, const std::vector<int>& shape, Format fmt = Format::kFormatFp32);
		tensor(float c, const std::vector<int>& shape, Format fmt = Format::kFormatFp32);
		void* map() const;
		void unMap() const;
		Shape getShape() const;
		int getId() const;
		int dimNum() const;
		int dimSize(int axis) const;
		int count(int start_axis = 0, int end_axis = -1) const;
		char* toHost() const;
		tensor reshape(const char* data, const std::vector<int>& shape, bool alloc = false,
		               Format fmt = Format::kFormatInvalid);
		tensor reshape(const std::vector<int>& shape);
		void set_to(float val) const;
		Format getFormat() const;
		size_t size() const { return size_in_byte; }
		bool isEmpty() const { return size_in_byte == 0; }
		void copyTo(tensor dst) const;
		std::shared_ptr<buffer>& getBuffer() { return m_buffer; }

		// init
		

	private:

		int id{};
		bool counted = false;
		VkDevice m_device;
		std::vector<int> m_shape;
		size_t size_in_byte;
		std::shared_ptr<buffer> m_buffer;
		Format format;
		static int& get_object_id();
		void update_id();
	};
}

#endif

#ifndef INIT_H
#define INIT_H

namespace kernel {
	namespace init {
		char* normal_distribution_init(std::vector<int> shape, float mean, float std); 

		template <typename dType = float>
		char* fill_memory_shape(std::vector<int> shape, dType c)
		{
			const size_t _shape = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
			auto* ret = new dType[_shape];
			for (int i = 0; i < _shape; ++i)
				ret[i] = reinterpret_cast<dType&>(c);
			return reinterpret_cast<char*>(ret);
		}
	}
}

#endif