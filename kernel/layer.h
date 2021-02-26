#pragma once

#include "kernel.h"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstdarg>
#include <future>
#include <list>

constexpr int local_sz_x = 1024;
constexpr int max_compute_work_group_count = 1024;

class context;

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
    void bindtensor(std::shared_ptr<tensor> tensor, int binding);

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

    std::string m_type;
};
