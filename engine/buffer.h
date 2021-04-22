#pragma once

#include <vulkan/vulkan.h>

class buffer
{
public:
    buffer(VkDevice& device) : m_device(device), m_buffer(nullptr), m_memory(nullptr)
    {
    };
    buffer(VkDevice& device, size_t size_in_bytes, const char* data);
    ~buffer();
    VkDeviceMemory getVkMemory() const { return m_memory; }
    VkBuffer getVkBuffer() const { return m_buffer; }

private:
    buffer();
    bool init(size_t size_in_bytes, const char* data);
    VkDevice m_device;
    VkBuffer m_buffer;
    VkDeviceMemory m_memory;
};
