# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project ("madml")

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(BACKEND_INCLUDE ${INCLUDE_DIR}/backend)
set(BACKEND_SRC ${SRC_DIR}/backend)
set(VKNN_INCLUDE ${INCLUDE_DIR}/vknn)
set(VKNN_SRC ${SRC_DIR}/vknn)

find_package(Vulkan REQUIRED FATAL_ERROR)
message(STATUS "Attempting to locate Vulkan SDK using manual path......")
set(VULKAN_SDK "C:/VulkanSDK")
set(VULKAN_VERSION "1.2.154.1")
set(VULKAN_PATH "${VULKAN_SDK}/${VULKAN_VERSION}")
message(STATUS "Using manual specified path: ${VULKAN_PATH}")

# Include sub-projects.
#add_subdirectory("cmake_madml")
add_subdirectory("backend")
add_subdirectory("vknn")
