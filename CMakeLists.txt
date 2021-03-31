# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project ("madml")

find_package(Vulkan REQUIRED FATAL_ERROR)
message(STATUS "Attempting to locate Vulkan SDK using manual path......")
set(VULKAN_SDK "C:/VulkanSDK")
set(VULKAN_VERSION "1.2.154.1")
set(VULKAN_PATH "${VULKAN_SDK}/${VULKAN_VERSION}")
message(STATUS "Using manual specified path: ${VULKAN_PATH}")

add_subdirectory("engine")
add_subdirectory("vknn")