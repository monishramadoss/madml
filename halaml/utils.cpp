#include "common.h"
#include "utils.h"

namespace kernel {
	void bindTensor(VkDevice& device, tensor* tensor, int binding, VkDescriptorSet descriptor_set) {
		VkDescriptorBufferInfo desc_buffer_info = {};
		desc_buffer_info.buffer = tensor->getBuffer()->getVkBuffer();
		desc_buffer_info.offset = 0;
		desc_buffer_info.range = tensor->size();

		VkWriteDescriptorSet write_descriptor_set = {};
		write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_descriptor_set.dstSet = descriptor_set;
		write_descriptor_set.dstBinding = binding;
		write_descriptor_set.descriptorCount = 1;
		write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		write_descriptor_set.pBufferInfo = &desc_buffer_info;

		vkUpdateDescriptorSets(device, 1, &write_descriptor_set, 0, NULL);
	}

    std::vector<uint32_t> compile(const std::string& name, shaderc_shader_kind kind, const std::string& data){
        std::vector<uint32_t> result;
#ifdef USE_SHADERC

        shaderc::Compiler compiler;
        shaderc::CompileOptions options;


        options.SetGenerateDebugInfo();
        options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
        shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(data.c_str(), data.size(), kind, name.c_str(), options);

        if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
            std::cerr << module.GetErrorMessage();
        }
        result.assign(module.cbegin(), module.cend());
        return result;
#else
        return result;
#endif
    }


}