#ifndef BUFFER_H
#define BUFFER_H
#include <vulkan/vulkan.h>

namespace kernel {
	class buffer
	{
	public:
		buffer(VkDevice& device) : m_device(device), m_buffer(VK_NULL_HANDLE), m_memory(VK_NULL_HANDLE) {};
		buffer(VkDevice& device, size_t size_in_bytes, const char* data);
		~buffer();
		VkDeviceMemory getVkMemory() { return m_memory; }
		VkBuffer getVkBuffer() { return m_buffer; }

	private:
		buffer();
		bool init(size_t size_in_bytes, const char* data);
		VkDevice m_device;
		VkBuffer m_buffer;
		VkDeviceMemory m_memory;
	};
}

#endif