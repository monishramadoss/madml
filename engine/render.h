#pragma once

#include "engine.h"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstdarg>
#include <future>
#include <list>

class render
{
public:
    render();
    virtual ~render();

    void initVulkanThing();

private:
    VkPhysicalDevice m_physical_device;
    VkDevice m_device;
    VkPipeline m_pipeline;
    VkCommandBuffer m_cmd_buffer;
    VkDescriptorPool m_descriptor_pool;
    VkDescriptorSet m_descriptor_set;
    VkDescriptorSetLayout m_descriptor_set_layout;
    VkPipelineLayout m_pipeline_layout;
    VkShaderModule m_shader_module;
};