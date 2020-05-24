#ifndef TENSOR_H
#define TENSOR_H
#include <memory>
#include <vulkan/vulkan.h>
#include "madml.h"


namespace kernel {
	class buffer;

	class tensor
	{
	public:
		tensor(Format fmt = kFormatFp32);
		tensor(const char* data, std::vector<int> shape, Format fmt = kFormatInvalid);
		void* map();
		void unMap();
		Shape getShape() const;
		int dimNum() const;
		int dimSize(const int axis) const;
		int count(const int satart_axis = 0, const int end_axis = -1) const;
		char* toHost();

		tensor reshape(const char* data, const std::vector<int>& shape, bool alloc = false, Format fmt = kFormatInvalid);

		void setTo(float val);
		int getFormat() const;
		size_t size() const { return size_in_byte; }
		bool isEmpty() { return size_in_byte == 0; }
		void copyTo(tensor& dst);
		std::shared_ptr<buffer> getBuffer() { return m_buffer; }

	private:
		VkDevice m_device;
		std::vector<int> m_shape;
		size_t size_in_byte;
		const char* m_data;
		std::shared_ptr<buffer> m_buffer;
		Format format;

		std::vector<tensor> input;
		std::vector<tensor> output;
	};
}
#endif