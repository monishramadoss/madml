#include "common.h"
#include "utils.h"
#include "buffer.h"

static uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(kPhysicalDevices[0], &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
        if ((memoryTypeBits & (1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) ==
            properties))
            return i;
    }
    return -1;
}

bool buffer::init(size_t size_in_bytes, const char* data)
{
    if (m_buffer != nullptr)
    {
        printf("Warn: Buffer object already initiated\n");
        return false;
    }

    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size_in_bytes;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(vkCreateBuffer(m_device, &bufferCreateInfo, nullptr, &m_buffer));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(m_device, m_buffer, &memoryRequirements);
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_memory));

    if (data)
    {
        char* dst;
        VK_CHECK_RESULT(vkMapMemory(m_device, m_memory, 0, size_in_bytes, 0, (void**)&dst));
        memcpy(dst, data, size_in_bytes);
        vkUnmapMemory(m_device, m_memory);
    }

    VK_CHECK_RESULT(vkBindBufferMemory(m_device, m_buffer, m_memory, 0));
    return true;
}

buffer::buffer(VkDevice& device, size_t size_in_bytes, const char* data)
{
    m_device = device;
    m_buffer = nullptr;
    m_memory = nullptr;
    init(size_in_bytes, data);
}

buffer::~buffer()
{
    vkFreeMemory(m_device, m_memory, nullptr);
    vkDestroyBuffer(m_device, m_buffer, nullptr);
}