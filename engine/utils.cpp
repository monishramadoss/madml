#include "common.h"
#include "utils.h"

std::vector<uint32_t> compile(const std::string& name, const std::string& data)
{
    std::vector<uint32_t> result;
#ifdef USE_SHADERC

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    options.SetGenerateDebugInfo();
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(data.c_str(), data.size(), shaderc_glsl_compute_shader, name.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        std::cerr << module.GetErrorMessage();
    }
    result.assign(module.cbegin(), module.cend());
    return result;
#else
    return result;
#endif
}